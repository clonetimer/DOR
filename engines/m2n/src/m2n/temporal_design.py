from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import math
import random

import numpy as np
import torch
from torch import nn

from .model_family import ModelingContract, ModelingObjective, InputRepresentation, MechanismUsage
from .selection import ArchitectureSelectionPolicy, select_architecture


class TemporalModelKind(str, Enum):
    SUMMARY_MLP = 'SUMMARY_MLP'
    WAVEFORM_FEATURE_LINEAR = 'WAVEFORM_FEATURE_LINEAR'
    WAVEFORM_ORDER_LINEAR = 'WAVEFORM_ORDER_LINEAR'
    LINEAR_AR = 'LINEAR_AR'
    K1_INTEGRATOR = 'K1_INTEGRATOR'
    BOUNDED_INTEGRATOR = 'BOUNDED_INTEGRATOR'
    QUATERNION_KINEMATICS = 'QUATERNION_KINEMATICS'
    AFFINE_STATE_SPACE = 'AFFINE_STATE_SPACE'
    TCN = 'TCN'
    GRU = 'GRU'
    TRANSFORMER = 'TRANSFORMER'


@dataclass(frozen=True)
class TemporalCandidate:
    id: str
    kind: TemporalModelKind
    task: str
    reason: str
    config: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d['kind'] = self.kind.value
        if self.config is None:
            d.pop('config', None)
        return d


@dataclass(frozen=True)
class TemporalQualification:
    status: str  # READY | AMBIGUOUS | UNSUPPORTED
    context_column: str | None
    time_column: str | None
    split_column: str | None
    train_contexts: int
    validation_contexts: int
    min_sequence_length: int
    max_sequence_length: int
    regular_sampling: bool
    dt_relative_spread: float
    reasons: tuple[str, ...]
    candidates: tuple[TemporalCandidate, ...]
    context_support: Mapping[str, Any] | None = None
    forecast_evaluation: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            'candidates': [x.to_dict() for x in self.candidates],
        }


@dataclass(frozen=True)
class TemporalDesignSummary:
    status: str
    selected_candidate_id: str | None
    output_dir: str
    candidate_validation_risks: Mapping[str, Sequence[float]]
    selection: Mapping[str, Any] | None
    qualification: Mapping[str, Any]
    task: str
    candidate_one_step_validation_risks: Mapping[str, Sequence[float]] | None = None
    forecast_evaluation: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'status': self.status,
            'selected_candidate_id': self.selected_candidate_id,
            'output_dir': self.output_dir,
            'candidate_validation_risks': {k: list(map(float, v)) for k, v in self.candidate_validation_risks.items()},
            'selection': self.selection,
            'qualification': self.qualification,
            'task': self.task,
            **({'candidate_one_step_validation_risks': {k: list(map(float, v)) for k, v in self.candidate_one_step_validation_risks.items()}} if self.candidate_one_step_validation_risks is not None else {}),
            **({'forecast_evaluation': dict(self.forecast_evaluation)} if self.forecast_evaluation is not None else {}),
        }



def _forecast_evaluation_policy(contract: ModelingContract, min_sequence_length: int | None = None) -> dict[str, Any]:
    """Resolve the model-selection forecast semantics.

    v2.16 keeps legacy one-step ranking as the default.  Recursive rollout is
    opt-in and deliberately narrow: the initial native implementation only
    supports closed autoregressive state forecasts where input_columns and
    target_columns are identical and ordered the same.  This prevents M2M from
    silently inventing future exogenous inputs.
    """
    cfg = dict((contract.constraints or {}).get('forecast_evaluation') or {})
    if not cfg:
        return {
            'mode': 'ONE_STEP',
            'horizon_steps': 1,
            'metric': 'CONTEXT_BALANCED_STANDARDIZED_RMSE',
            'closed_loop': False,
        }
    mode = str(cfg.get('mode', 'ONE_STEP')).upper()
    if mode not in {'ONE_STEP', 'RECURSIVE_ROLLOUT'}:
        raise ValueError(f"unsupported forecast_evaluation.mode {mode!r}")
    try:
        horizon = int(cfg.get('horizon_steps', 1))
    except (TypeError, ValueError):
        raise ValueError('forecast_evaluation.horizon_steps must be an integer') from None
    if horizon < 1:
        raise ValueError('forecast_evaluation.horizon_steps must be >= 1')
    if mode == 'ONE_STEP' and horizon != 1:
        raise ValueError('forecast_evaluation mode ONE_STEP requires horizon_steps=1')
    if mode == 'RECURSIVE_ROLLOUT' and horizon < 2:
        raise ValueError('forecast_evaluation mode RECURSIVE_ROLLOUT requires horizon_steps>=2')
    out = {
        'mode': mode,
        'horizon_steps': horizon,
        'metric': 'CONTEXT_BALANCED_ENDPOINT_RMSE',
        'closed_loop': mode == 'RECURSIVE_ROLLOUT',
    }
    if min_sequence_length is not None:
        tcfg = dict((contract.constraints or {}).get('temporal_training') or {})
        default_lb = max(2, min(12, min_sequence_length // 4 if min_sequence_length >= 8 else min_sequence_length - 1))
        lookback = int(tcfg.get('lookback', default_lb))
        if lookback >= min_sequence_length:
            lookback = max(1, min_sequence_length - 1)
        out['resolved_lookback'] = int(lookback)
        out['max_supported_horizon_steps'] = max(0, int(min_sequence_length - lookback))
    return out

def _split_values(spec: Mapping[str, Any]) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    d = spec.get('data') or {}
    col = str(d.get('split_column', 'split'))
    tr = tuple(map(str, d.get('train_values') or ('train',)))
    va = tuple(map(str, d.get('validation_values') or ('val', 'validation')))
    return col, tr, va


def _context_and_time(spec: Mapping[str, Any], contract: ModelingContract) -> tuple[str | None, str | None]:
    q = spec.get('qualification') or {}
    d = spec.get('data') or {}
    context = q.get('independent_context_column') if isinstance(q, Mapping) else None
    time_col = contract.time_column or (d.get('time_column') if isinstance(d, Mapping) else None)
    return (None if context is None else str(context), None if time_col is None else str(time_col))


def _ordered_contexts(data: Mapping[str, np.ndarray], context_col: str, time_col: str) -> list[tuple[str, np.ndarray]]:
    ctx = np.asarray(data[context_col]).astype(str)
    t = np.asarray(data[time_col], dtype=float)
    out = []
    for c in np.unique(ctx):
        idx = np.where(ctx == c)[0]
        order = np.argsort(t[idx], kind='stable')
        out.append((str(c), idx[order]))
    return out


def _numeric_nonfinite_reasons(data: Mapping[str, np.ndarray], columns: Sequence[str]) -> list[str]:
    reasons: list[str] = []
    for col in dict.fromkeys(columns):
        try:
            arr = np.asarray(data[col], dtype=float)
        except (TypeError, ValueError):
            reasons.append(
                f"temporal numeric column {col!r} contains non-numeric or blank values; "
                "declare an explicit preprocessing/missing-data strategy before temporal training"
            )
            continue
        bad = ~np.isfinite(arr)
        if np.any(bad):
            reasons.append(
                f"temporal numeric column {col!r} contains {int(np.sum(bad))} non-finite values; "
                "declare an explicit preprocessing/missing-data strategy before temporal training"
            )
    return reasons


def _context_counts_without_numeric_time(
    data: Mapping[str, np.ndarray], context_col: str, split_col: str, train_values: Sequence[str], val_values: Sequence[str]
) -> tuple[int, int, int, int]:
    ctx = np.asarray(data[context_col]).astype(str)
    split = np.asarray(data[split_col]).astype(str)
    train_ctx = set(ctx[np.isin(split, list(train_values))].tolist())
    val_ctx = set(ctx[np.isin(split, list(val_values))].tolist())
    _, counts = np.unique(ctx, return_counts=True)
    return len(train_ctx), len(val_ctx), int(counts.min()) if len(counts) else 0, int(counts.max()) if len(counts) else 0


def _context_support_policy(contract: ModelingContract) -> dict[str, Any]:
    cfg = dict((contract.constraints or {}).get('temporal_context_support') or {})
    if not cfg:
        return {
            'mode': 'DIAGNOSTIC_ONLY',
            'selection_scope': 'ALL_VALIDATION',
            'dimensions': ['length', 'duration'],
            'operating_condition': None,
        }
    mode = str(cfg.get('mode', 'EMPIRICAL_TRAIN_ENVELOPE')).upper()
    scope = str(cfg.get('selection_scope', 'IN_SUPPORT_ONLY')).upper()
    dims = [str(x).lower() for x in (cfg.get('dimensions') or ('length', 'duration'))]
    bad = sorted(set(dims) - {'length', 'duration'})
    if bad:
        raise ValueError(f'unsupported temporal_context_support dimensions: {bad}')
    if mode not in {'EMPIRICAL_TRAIN_ENVELOPE', 'DIAGNOSTIC_ONLY'}:
        raise ValueError(f'unsupported temporal_context_support mode {mode!r}')
    if scope not in {'IN_SUPPORT_ONLY', 'ALL_VALIDATION'}:
        raise ValueError(f'unsupported temporal_context_support selection_scope {scope!r}')

    operating = cfg.get('operating_condition')
    op = None
    if operating is not None:
        if not isinstance(operating, Mapping):
            raise ValueError('temporal_context_support.operating_condition must be an object')
        columns = [str(x) for x in (operating.get('columns') or ())]
        if not columns:
            raise ValueError('temporal_context_support.operating_condition.columns must contain at least one observed column')
        statistic = str(operating.get('statistic', 'MEAN')).upper()
        metric = str(operating.get('metric', 'STANDARDIZED_NEAREST_NEIGHBOR')).upper()
        threshold_rule = str(operating.get('threshold_rule', 'TRAIN_LEAVE_ONE_OUT_MAX')).upper()
        if statistic not in {'MEAN', 'START'}:
            raise ValueError(f'unsupported temporal operating-condition statistic {statistic!r}')
        if metric != 'STANDARDIZED_NEAREST_NEIGHBOR':
            raise ValueError(f'unsupported temporal operating-condition metric {metric!r}')
        if threshold_rule != 'TRAIN_LEAVE_ONE_OUT_MAX':
            raise ValueError(f'unsupported temporal operating-condition threshold_rule {threshold_rule!r}')
        op = {
            'columns': columns,
            'statistic': statistic,
            'metric': metric,
            'threshold_rule': threshold_rule,
        }
    return {'mode': mode, 'selection_scope': scope, 'dimensions': dims, 'operating_condition': op}


def _context_operating_vector(data: Mapping[str, np.ndarray], idx: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    statistic = str(cfg.get('statistic', 'MEAN')).upper()
    values = []
    for col in cfg.get('columns') or ():
        if col not in data:
            raise ValueError(f'temporal operating-condition support column {col!r} is missing')
        try:
            a = np.asarray(data[col], dtype=float)[idx]
        except Exception as exc:
            raise ValueError(f'temporal operating-condition support column {col!r} must be numeric') from exc
        if not np.all(np.isfinite(a)):
            raise ValueError(f'temporal operating-condition support column {col!r} contains non-finite values')
        values.append(float(a[0] if statistic == 'START' else np.mean(a)))
    return np.asarray(values, dtype=float)


def _fit_operating_support(train_vectors: np.ndarray, policy: Mapping[str, Any]) -> dict[str, Any]:
    from scipy.spatial import cKDTree
    x = np.asarray(train_vectors, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError('temporal operating-condition support needs at least two train contexts')
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    z = (x - mean) / std
    tree = cKDTree(z)
    d, _ = tree.query(z, k=2)
    loo = np.asarray(d[:, 1], dtype=float)
    threshold = float(np.max(loo))
    if not np.isfinite(threshold):
        raise ValueError('temporal operating-condition support could not derive a finite train reference distance')
    return {
        'enabled': True,
        'columns': list(policy['columns']),
        'statistic': str(policy['statistic']),
        'metric': str(policy['metric']),
        'threshold_rule': str(policy['threshold_rule']),
        'feature_mean': mean.tolist(),
        'feature_std': std.tolist(),
        'threshold': threshold,
        'train_vectors_standardized': z.tolist(),
        'train_leave_one_out_distance': {
            'min': float(np.min(loo)),
            'median': float(np.median(loo)),
            'max': threshold,
        },
    }


def _operating_distance(model: Mapping[str, Any], raw_vector: np.ndarray) -> float:
    from scipy.spatial import cKDTree
    mean = np.asarray(model['feature_mean'], dtype=float)
    std = np.asarray(model['feature_std'], dtype=float)
    train = np.asarray(model['train_vectors_standardized'], dtype=float)
    z = (np.asarray(raw_vector, dtype=float) - mean) / std
    d, _ = cKDTree(train).query(z, k=1)
    return float(d)


def _context_support_analysis(spec: Mapping[str, Any], contract: ModelingContract, data: Mapping[str, np.ndarray]) -> tuple[dict[str, Any], set[str]]:
    context_col, time_col = _context_and_time(spec, contract)
    split_col, train_values, val_values = _split_values(spec)
    if not context_col or not time_col or context_col not in data or time_col not in data or split_col not in data:
        return {'enabled': False}, set()
    policy = _context_support_policy(contract)
    split = np.asarray(data[split_col]).astype(str)
    tt = np.asarray(data[time_col], dtype=float)
    train_rows = []
    val_rows = []
    for c, idx in _ordered_contexts(data, context_col, time_col):
        vals = set(split[idx].tolist())
        if len(vals) != 1:
            continue
        rec = {
            'id': c,
            'idx': idx,
            'length': int(len(idx)),
            'duration': float(tt[idx[-1]] - tt[idx[0]]) if len(idx) else 0.0,
        }
        v = next(iter(vals))
        if v in train_values:
            train_rows.append(rec)
        elif v in val_values:
            val_rows.append(rec)
    if not train_rows:
        return {'enabled': policy['mode'] != 'DIAGNOSTIC_ONLY', 'policy': policy, 'train_contexts': 0}, set()

    dims = policy['dimensions']
    envelope = {}
    for d in dims:
        vals = np.asarray([r[d] for r in train_rows], dtype=float)
        envelope[d] = {'min': float(vals.min()), 'max': float(vals.max())}

    operating_model = None
    op = policy.get('operating_condition')
    if op:
        train_op = np.stack([_context_operating_vector(data, r['idx'], op) for r in train_rows])
        operating_model = _fit_operating_support(train_op, op)
        for r in val_rows:
            vec = _context_operating_vector(data, r['idx'], op)
            r['operating_distance'] = _operating_distance(operating_model, vec)
            r['operating_in_support'] = bool(r['operating_distance'] <= float(operating_model['threshold']) + 1e-12)

    supported = set()
    shape_out = []
    operating_out = []
    out = []
    val_distances = []
    for r in val_rows:
        shape_ok = True
        for d in dims:
            lo, hi = envelope[d]['min'], envelope[d]['max']
            shape_ok = shape_ok and (lo <= float(r[d]) <= hi)
        op_ok = bool(r.get('operating_in_support', True))
        if operating_model:
            val_distances.append(float(r['operating_distance']))
        if not shape_ok:
            shape_out.append(r['id'])
        if not op_ok:
            operating_out.append(r['id'])
        if shape_ok and op_ok:
            supported.add(r['id'])
        else:
            out.append(r['id'])

    operating_summary = operating_model
    if operating_summary:
        operating_summary = dict(operating_summary)
        operating_summary['validation_in_support_contexts'] = int(sum(bool(r.get('operating_in_support', True)) for r in val_rows))
        operating_summary['validation_out_of_support_contexts'] = int(len(operating_out))
        if val_distances:
            operating_summary['validation_distance'] = {
                'min': float(np.min(val_distances)),
                'median': float(np.median(val_distances)),
                'max': float(np.max(val_distances)),
            }

    return {
        'enabled': policy['mode'] == 'EMPIRICAL_TRAIN_ENVELOPE',
        'policy': policy,
        'train_contexts': len(train_rows),
        'validation_contexts': len(val_rows),
        'train_envelope': envelope,
        'operating_condition_support': operating_summary,
        'validation_shape_out_of_support_contexts': len(shape_out),
        'validation_operating_out_of_support_contexts': len(operating_out),
        'validation_in_support_contexts': len(supported),
        'validation_out_of_support_contexts': len(out),
        'validation_out_of_support_fraction': float(len(out) / max(len(val_rows), 1)),
        'selection_validation_contexts': len(supported) if policy['selection_scope'] == 'IN_SUPPORT_ONLY' and policy['mode'] == 'EMPIRICAL_TRAIN_ENVELOPE' else len(val_rows),
    }, supported


def _context_support_details_from_frozen(q: Mapping[str, Any], data: Mapping[str, np.ndarray], idx: np.ndarray, length: int, duration: float) -> dict[str, Any]:
    cs = dict(q.get('context_support') or {})
    env = dict(cs.get('train_envelope') or {})
    policy = dict(cs.get('policy') or {})
    result = {
        'overall': 'NOT_ENFORCED',
        'shape': 'NOT_ENFORCED',
        'operating': 'NOT_ENFORCED',
        'operating_distance': None,
    }
    if not cs.get('enabled'):
        return result

    shape_ok = True
    for d in policy.get('dimensions') or ('length', 'duration'):
        e = env.get(d)
        if not e:
            continue
        v = float(length if d == 'length' else duration)
        if v < float(e['min']) or v > float(e['max']):
            shape_ok = False
    result['shape'] = 'IN_SUPPORT' if shape_ok else 'OUT_OF_SUPPORT'

    op_model = cs.get('operating_condition_support')
    op_ok = True
    if op_model:
        op_cfg = {
            'columns': op_model.get('columns') or (),
            'statistic': op_model.get('statistic', 'MEAN'),
        }
        raw = _context_operating_vector(data, idx, op_cfg)
        dist = _operating_distance(op_model, raw)
        op_ok = bool(dist <= float(op_model['threshold']) + 1e-12)
        result['operating'] = 'IN_SUPPORT' if op_ok else 'OUT_OF_SUPPORT'
        result['operating_distance'] = float(dist)

    result['overall'] = 'IN_SUPPORT' if shape_ok and op_ok else 'OUT_OF_SUPPORT'
    return result


def _context_support_flag_from_frozen(q: Mapping[str, Any], length: int, duration: float) -> str:
    """Backward-compatible length/duration-only support helper used by older callers/tests."""
    cs = dict(q.get('context_support') or {})
    env = dict(cs.get('train_envelope') or {})
    policy = dict(cs.get('policy') or {})
    if not cs.get('enabled') or not env:
        return 'NOT_ENFORCED'
    for d in policy.get('dimensions') or ('length', 'duration'):
        e = env.get(d)
        if not e:
            continue
        v = float(length if d == 'length' else duration)
        if v < float(e['min']) or v > float(e['max']):
            return 'OUT_OF_SUPPORT'
    return 'IN_SUPPORT'


def _waveform_feature_config_errors(wcfg: Mapping[str, Any], min_len: int) -> list[str]:
    errors: list[str] = []
    window = wcfg.get('sample_window')
    feature_set = str(wcfg.get('feature_set', 'TIME_DOMAIN_V1')).upper()
    if not isinstance(window, (list, tuple)) or len(window) != 2:
        errors.append('waveform representation requires sample_window=[start,end) declared before model selection')
        return errors
    try:
        ws, we = int(window[0]), int(window[1])
    except (TypeError, ValueError):
        ws, we = -1, -1
    if ws < 0 or we <= ws or we > min_len:
        errors.append(f'waveform sample_window must satisfy 0 <= start < end <= min_sequence_length ({min_len})')
    if feature_set != 'TIME_DOMAIN_V1':
        errors.append(f'unsupported waveform feature_set {feature_set!r}; native deterministic feature baseline supports TIME_DOMAIN_V1')
    return errors


def _waveform_order_config_errors(
    wcfg: Mapping[str, Any],
    contract: ModelingContract,
    data: Mapping[str, np.ndarray],
    contexts: Sequence[tuple[str, np.ndarray]],
    min_len: int,
) -> list[str]:
    errors = _waveform_feature_config_errors(wcfg, min_len)
    ocfg = dict(wcfg.get('order_domain') or {})
    if not ocfg:
        errors.append('WAVEFORM_ORDER_LINEAR requires waveform_representation.order_domain')
        return errors
    rot_col = str(ocfg.get('rotation_frequency_column', ''))
    signal_cols = tuple(map(str, ocfg.get('signal_columns') or ()))
    try:
        sample_rate = float(ocfg.get('sample_rate_hz'))
    except (TypeError, ValueError):
        sample_rate = math.nan
    carrier = ocfg.get('carrier_band_hz')
    orders = ocfg.get('characteristic_orders')
    try:
        harmonics = int(ocfg.get('harmonics', 1))
    except (TypeError, ValueError):
        harmonics = 0
    sidebands = ocfg.get('sidebands', [0.0])
    if not rot_col or rot_col not in contract.input_columns:
        errors.append('waveform order-domain representation requires rotation_frequency_column in modeling_contract.input_columns')
    if not signal_cols or any(c not in contract.input_columns or c == rot_col for c in signal_cols):
        errors.append('waveform order-domain signal_columns must be a non-empty subset of modeling_contract.input_columns excluding rotation_frequency_column')
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        errors.append('waveform order-domain sample_rate_hz must be finite and positive')
    if not isinstance(carrier, (list, tuple)) or len(carrier) != 2:
        errors.append('waveform order-domain carrier_band_hz must be [low, high]')
    else:
        try:
            clo, chi = float(carrier[0]), float(carrier[1])
        except (TypeError, ValueError):
            clo, chi = math.nan, math.nan
        if not (np.isfinite(clo) and np.isfinite(chi) and 0 <= clo < chi <= 0.5 * sample_rate):
            errors.append('waveform order-domain carrier_band_hz must satisfy 0 <= low < high <= Nyquist')
    try:
        order_vals = tuple(map(float, orders or ()))
    except (TypeError, ValueError):
        order_vals = ()
    if not order_vals or any((not np.isfinite(v) or v <= 0) for v in order_vals):
        errors.append('waveform order-domain characteristic_orders must contain finite positive engineering-declared orders')
    try:
        sideband_vals = tuple(map(float, sidebands or ()))
    except (TypeError, ValueError):
        sideband_vals = ()
    if harmonics < 1 or not sideband_vals or any(not np.isfinite(v) for v in sideband_vals):
        errors.append('waveform order-domain harmonics must be >=1 and sidebands must contain finite order offsets')
    if rot_col in data:
        try:
            rv = np.asarray(data[rot_col], dtype=float)
            for c, idx in contexts:
                vals = rv[idx]
                if np.any(~np.isfinite(vals)) or np.any(vals <= 0) or float(np.max(vals) - np.min(vals)) > 1e-9 * max(1.0, float(np.max(np.abs(vals)))):
                    errors.append(f'rotation frequency must be finite, positive and constant within context {c!r}')
                    break
        except (TypeError, ValueError):
            errors.append(f'rotation frequency column {rot_col!r} must be numeric')
    return errors


def _explicit_representation_candidate_specs(contract: ModelingContract) -> tuple[list[tuple[str, TemporalModelKind, dict[str, Any]]] | None, list[str]]:
    constraints = dict(contract.constraints or {})
    raw = constraints.get('representation_candidates')
    if raw is None:
        return None, []
    errors: list[str] = []
    if not isinstance(raw, (list, tuple)) or not raw:
        return [], ['constraints.representation_candidates must be a non-empty array when supplied']
    out: list[tuple[str, TemporalModelKind, dict[str, Any]]] = []
    seen: set[str] = set()
    allowed = {TemporalModelKind.WAVEFORM_FEATURE_LINEAR, TemporalModelKind.WAVEFORM_ORDER_LINEAR}
    for i, item in enumerate(raw):
        if not isinstance(item, Mapping):
            errors.append(f'representation_candidates[{i}] must be an object')
            continue
        cid = str(item.get('id', '')).strip()
        if not cid:
            errors.append(f'representation_candidates[{i}].id is required')
            continue
        cid_key = cid.casefold()
        if cid_key in seen:
            errors.append(f'duplicate representation candidate id {cid!r}')
            continue
        seen.add(cid_key)
        try:
            kind = TemporalModelKind(str(item.get('kind', '')).upper())
        except ValueError:
            errors.append(f'representation candidate {cid!r} has unknown kind {item.get("kind")!r}')
            continue
        if kind not in allowed:
            errors.append(
                f'representation candidate {cid!r} kind {kind.value!r} is not parameterized in v2.16; '
                'supported kinds are WAVEFORM_FEATURE_LINEAR and WAVEFORM_ORDER_LINEAR'
            )
            continue
        wcfg = item.get('waveform_representation')
        if not isinstance(wcfg, Mapping):
            errors.append(f'representation candidate {cid!r} requires waveform_representation object')
            continue
        out.append((cid, kind, dict(wcfg)))
    return out, errors

def qualify_temporal_design(spec: Mapping[str, Any], data: Mapping[str, np.ndarray]) -> TemporalQualification:
    contract = ModelingContract.from_mapping(spec.get('modeling_contract') or {})
    reasons: list[str] = []
    if contract.input_representation not in (InputRepresentation.MULTIVARIATE_SEQUENCE, InputRepresentation.WAVEFORM_SEQUENCE):
        return TemporalQualification('UNSUPPORTED', None, contract.time_column, None, 0, 0, 0, 0, False, math.inf,
                                     ('temporal-design requires a sequence representation: MULTIVARIATE_SEQUENCE or WAVEFORM_SEQUENCE',), ())
    context_col, time_col = _context_and_time(spec, contract)
    split_col, train_values, val_values = _split_values(spec)
    if not context_col:
        reasons.append('independent_context_column is required so temporal samples cannot leak across trajectories')
    if not time_col:
        reasons.append('a local time column is required for ordered sequence modeling')
    for col in [context_col, time_col, split_col, *contract.input_columns]:
        if col and col not in data:
            reasons.append(f'required temporal column {col!r} is missing')
    if contract.objective == ModelingObjective.CATEGORICAL_PREDICTION and contract.label_column and contract.label_column not in data:
        reasons.append(f'label column {contract.label_column!r} is missing')
    if contract.objective in (ModelingObjective.DYNAMICS_IDENTIFICATION,) and not contract.target_columns:
        reasons.append('DYNAMICS_IDENTIFICATION requires target_columns for temporal forecasting')
    for col in contract.target_columns:
        if col not in data:
            reasons.append(f'target column {col!r} is missing')
    if reasons:
        return TemporalQualification('UNSUPPORTED', context_col, time_col, split_col, 0, 0, 0, 0, False, math.inf, tuple(reasons), ())

    numeric_cols = [time_col, *contract.input_columns, *contract.target_columns]
    numeric_reasons = _numeric_nonfinite_reasons(data, [c for c in numeric_cols if c])
    if numeric_reasons:
        trn, van, mn, mx = _context_counts_without_numeric_time(data, context_col, split_col, train_values, val_values)
        return TemporalQualification(
            'AMBIGUOUS', context_col, time_col, split_col, trn, van, mn, mx, False, math.inf,
            tuple(numeric_reasons), ()
        )

    contexts = _ordered_contexts(data, context_col, time_col)
    split = np.asarray(data[split_col]).astype(str)
    lengths: list[int] = []
    dt_rels: list[float] = []
    train_ctx: set[str] = set()
    val_ctx: set[str] = set()
    for c, idx in contexts:
        lengths.append(int(len(idx)))
        vals = set(split[idx].tolist())
        if len(vals) != 1:
            reasons.append(f'context {c!r} crosses split boundaries: {sorted(vals)}')
            continue
        v = next(iter(vals))
        if v in train_values:
            train_ctx.add(c)
        elif v in val_values:
            val_ctx.add(c)
        tt = np.asarray(data[time_col], dtype=float)[idx]
        if len(tt) >= 2:
            dt = np.diff(tt)
            if not np.all(np.isfinite(dt)) or np.any(dt <= 0):
                reasons.append(f'context {c!r} time is not strictly increasing')
            else:
                m = float(np.mean(dt))
                dt_rels.append(float(np.std(dt) / max(abs(m), 1e-12)))
    if train_ctx & val_ctx:
        reasons.append('train/validation independent contexts overlap')
    min_len = min(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0
    if min_len < 2:
        reasons.append('each temporal context needs at least two ordered samples')
    rel = max(dt_rels) if dt_rels else math.inf
    tol = float((contract.constraints or {}).get('max_dt_relative_spread', 1e-3))
    regular = bool(np.isfinite(rel) and rel <= tol)
    if not regular:
        reasons.append(f'initial native temporal backends require approximately regular sampling within context (max relative dt spread={rel:.6g}, tolerance={tol:.6g})')
    if not train_ctx:
        reasons.append('no train independent contexts found')
    if not val_ctx:
        reasons.append('no validation independent contexts found')

    forecast_evaluation = None
    try:
        forecast_evaluation = _forecast_evaluation_policy(contract, min_len)
    except ValueError as exc:
        reasons.append(f'forecast_evaluation contract invalid: {exc}')
    if forecast_evaluation and forecast_evaluation.get('mode') == 'RECURSIVE_ROLLOUT':
        if contract.objective != ModelingObjective.DYNAMICS_IDENTIFICATION:
            reasons.append('forecast_evaluation RECURSIVE_ROLLOUT is currently supported only for DYNAMICS_IDENTIFICATION')
        if tuple(contract.input_columns) != tuple(contract.target_columns):
            reasons.append('forecast_evaluation RECURSIVE_ROLLOUT initial native support requires input_columns == target_columns in identical order so future exogenous inputs are never invented')
        semantics = _target_semantics(contract)
        if semantics != 'EUCLIDEAN':
            reasons.append('forecast_evaluation RECURSIVE_ROLLOUT initial native support requires EUCLIDEAN target semantics')
        h = int(forecast_evaluation.get('horizon_steps', 1))
        max_h = int(forecast_evaluation.get('max_supported_horizon_steps', 0))
        if h > max_h:
            reasons.append(f'forecast_evaluation horizon_steps={h} exceeds the minimum-sequence supported horizon {max_h} after the resolved lookback')

    candidates: list[TemporalCandidate] = []
    if (contract.constraints or {}).get('representation_candidates') is not None and contract.objective != ModelingObjective.CATEGORICAL_PREDICTION:
        reasons.append('parameterized representation_candidates are currently implemented only for CATEGORICAL_PREDICTION with WAVEFORM_SEQUENCE; other objectives must use their existing qualified paths')
    if contract.objective == ModelingObjective.CATEGORICAL_PREDICTION:
        if contract.input_representation == InputRepresentation.WAVEFORM_SEQUENCE:
            wcfg = dict((contract.constraints or {}).get('waveform_representation') or {})
            if 'order_domain_candidates' in wcfg:
                reasons.append('waveform_representation.order_domain_candidates is not a supported contract field; use constraints.representation_candidates so each representation has an explicit candidate id and full configuration')
            explicit_specs, explicit_errors = _explicit_representation_candidate_specs(contract)
            reasons.extend(explicit_errors)
            if explicit_specs is None:
                feature_errors = _waveform_feature_config_errors(wcfg, min_len)
                if feature_errors:
                    reasons.extend(feature_errors)
                else:
                    candidates.append(TemporalCandidate('waveform_feature_linear', TemporalModelKind.WAVEFORM_FEATURE_LINEAR, 'SEQUENCE_TO_CLASS', 'deterministic ridge classifier on declared-region time-domain waveform features; retained so representation can compete with raw sequence models'))
                if wcfg.get('order_domain'):
                    order_errors = _waveform_order_config_errors(wcfg, contract, data, contexts, min_len)
                    if order_errors:
                        reasons.extend(order_errors)
                    else:
                        candidates.append(TemporalCandidate('waveform_order_linear', TemporalModelKind.WAVEFORM_ORDER_LINEAR, 'SEQUENCE_TO_CLASS', 'deterministic ridge classifier on an engineering-declared envelope/order representation normalized by measured rotation frequency'))
            else:
                for cid, kind, candidate_wcfg in explicit_specs:
                    if kind is TemporalModelKind.WAVEFORM_FEATURE_LINEAR:
                        errs = _waveform_feature_config_errors(candidate_wcfg, min_len)
                        reason = 'deterministic ridge classifier on a candidate-specific declared-region time-domain waveform representation'
                    else:
                        errs = _waveform_order_config_errors(candidate_wcfg, contract, data, contexts, min_len)
                        reason = 'deterministic ridge classifier on a candidate-specific engineering-declared order-domain waveform representation'
                    if errs:
                        reasons.extend([f'representation candidate {cid!r}: {e}' for e in errs])
                    else:
                        candidates.append(TemporalCandidate(cid, kind, 'SEQUENCE_TO_CLASS', reason, {'waveform_representation': candidate_wcfg}))
        candidates.append(TemporalCandidate('summary_mlp', TemporalModelKind.SUMMARY_MLP, 'SEQUENCE_TO_CLASS', 'order-free sequence summary baseline; retained so temporal order must earn its complexity'))
        if min_len >= 3:
            candidates.append(TemporalCandidate('tcn', TemporalModelKind.TCN, 'SEQUENCE_TO_CLASS', 'causal/local convolutional temporal pattern candidate'))
            candidates.append(TemporalCandidate('gru', TemporalModelKind.GRU, 'SEQUENCE_TO_CLASS', 'recurrent latent-state temporal candidate'))
        if min_len >= 4:
            candidates.append(TemporalCandidate('transformer', TemporalModelKind.TRANSFORMER, 'SEQUENCE_TO_CLASS', 'attention-based long-range interaction candidate'))
    elif contract.objective == ModelingObjective.STATIC_REGRESSION:
        if contract.input_representation == InputRepresentation.WAVEFORM_SEQUENCE:
            wcfg=dict((contract.constraints or {}).get('waveform_representation') or {})
            window=wcfg.get('sample_window'); feature_set=str(wcfg.get('feature_set','TIME_DOMAIN_V1')).upper()
            if not contract.target_columns:
                reasons.append('STATIC_REGRESSION waveform contract requires target_columns')
            if not isinstance(window,(list,tuple)) or len(window)!=2:
                reasons.append('WAVEFORM_SEQUENCE requires constraints.waveform_representation.sample_window=[start,end) for sequence-to-vector regression')
            else:
                try: ws,we=int(window[0]),int(window[1])
                except (TypeError,ValueError): ws,we=-1,-1
                if ws < 0 or we <= ws or we > min_len:
                    reasons.append(f'waveform sample_window must satisfy 0 <= start < end <= min_sequence_length ({min_len})')
                elif feature_set != 'TIME_DOMAIN_V1':
                    reasons.append(f'unsupported waveform feature_set {feature_set!r}; v2.13 supports TIME_DOMAIN_V1')
                else:
                    candidates.append(TemporalCandidate('waveform_feature_linear', TemporalModelKind.WAVEFORM_FEATURE_LINEAR, 'SEQUENCE_TO_VECTOR', 'deterministic ridge regressor on declared-region time-domain waveform features'))
            candidates.append(TemporalCandidate('summary_mlp', TemporalModelKind.SUMMARY_MLP, 'SEQUENCE_TO_VECTOR', 'order-free sequence summary regression baseline'))
            if min_len >= 3:
                candidates.append(TemporalCandidate('tcn', TemporalModelKind.TCN, 'SEQUENCE_TO_VECTOR', 'pooled convolutional sequence-to-vector regressor'))
                candidates.append(TemporalCandidate('gru', TemporalModelKind.GRU, 'SEQUENCE_TO_VECTOR', 'recurrent sequence-to-vector regressor'))
            if min_len >= 4:
                candidates.append(TemporalCandidate('transformer', TemporalModelKind.TRANSFORMER, 'SEQUENCE_TO_VECTOR', 'attention-based sequence-to-vector regressor'))
        else:
            reasons.append('native temporal STATIC_REGRESSION currently requires WAVEFORM_SEQUENCE; tabular static regression remains outside temporal-design')
    elif contract.objective in (ModelingObjective.NORMALITY_MODELING, ModelingObjective.DYNAMICS_IDENTIFICATION):
        physics_cfg = dict((contract.constraints or {}).get('temporal_physics') or {})
        if contract.objective == ModelingObjective.DYNAMICS_IDENTIFICATION and contract.mechanism_usage in (MechanismUsage.PREFERRED, MechanismUsage.REQUIRED):
            if physics_cfg:
                kind = str(physics_cfg.get('kind', '')).upper()
                states = tuple(map(str, physics_cfg.get('state_columns') or ()))
                drives = tuple(map(str, physics_cfg.get('drive_columns') or ()))
                if kind == 'K1_INTEGRATOR':
                    if not states or len(states) != len(drives):
                        reasons.append('K1_INTEGRATOR requires equal non-empty state_columns and drive_columns')
                    elif tuple(contract.target_columns) != states:
                        reasons.append('K1_INTEGRATOR state_columns must exactly match modeling_contract.target_columns order')
                    elif any(c not in contract.input_columns for c in (*states, *drives)):
                        reasons.append('K1_INTEGRATOR state_columns and drive_columns must all be present in modeling_contract.input_columns')
                    else:
                        try: dt = float(physics_cfg.get('dt'))
                        except (TypeError, ValueError): dt = math.nan
                        if not np.isfinite(dt) or dt <= 0:
                            reasons.append('K1_INTEGRATOR requires a finite positive temporal_physics.dt')
                        else:
                            candidates.append(TemporalCandidate('k1_integrator', TemporalModelKind.K1_INTEGRATOR, 'ONE_STEP_FORECAST', 'mechanism-preserving K1 state integrator with fixed or train-only learned drive gain'))
                elif kind == 'BOUNDED_INTEGRATOR':
                    lower_cols=tuple(map(str, physics_cfg.get('lower_bound_columns') or ()))
                    upper_cols=tuple(map(str, physics_cfg.get('upper_bound_columns') or ()))
                    lower_vals=physics_cfg.get('lower_bounds')
                    upper_vals=physics_cfg.get('upper_bounds')
                    try: dt=float(physics_cfg.get('dt'))
                    except (TypeError,ValueError): dt=math.nan
                    if not states or len(states) != len(drives):
                        reasons.append('BOUNDED_INTEGRATOR requires equal non-empty state_columns and drive_columns')
                    elif tuple(contract.target_columns) != states:
                        reasons.append('BOUNDED_INTEGRATOR state_columns must exactly match modeling_contract.target_columns order')
                    elif any(c not in contract.input_columns for c in (*states,*drives,*lower_cols,*upper_cols)):
                        reasons.append('BOUNDED_INTEGRATOR state/drive/bound columns must all be present in modeling_contract.input_columns')
                    elif not np.isfinite(dt) or dt<=0:
                        reasons.append('BOUNDED_INTEGRATOR requires a finite positive temporal_physics.dt')
                    elif lower_vals is None and len(lower_cols)!=len(states):
                        reasons.append('BOUNDED_INTEGRATOR requires lower_bounds or lower_bound_columns matching state_columns')
                    elif upper_vals is None and len(upper_cols)!=len(states):
                        reasons.append('BOUNDED_INTEGRATOR requires upper_bounds or upper_bound_columns matching state_columns')
                    else:
                        try:
                            if lower_vals is not None:
                                lv=np.asarray(lower_vals,dtype=float)
                                if lv.shape!=(len(states),) or not np.all(np.isfinite(lv)): raise ValueError
                            if upper_vals is not None:
                                uv=np.asarray(upper_vals,dtype=float)
                                if uv.shape!=(len(states),) or not np.all(np.isfinite(uv)): raise ValueError
                        except (TypeError,ValueError):
                            reasons.append('BOUNDED_INTEGRATOR fixed bounds must be finite arrays matching state_columns')
                        else:
                            candidates.append(TemporalCandidate('bounded_integrator', TemporalModelKind.BOUNDED_INTEGRATOR, 'ONE_STEP_FORECAST', 'mechanism-preserving bounded state integrator clip(x + dt*g*u, lower, upper) with declared bounds'))
                elif kind == 'QUATERNION_KINEMATICS':
                    semantics=_target_semantics(contract)
                    composition=str(physics_cfg.get('composition','LEFT')).upper()
                    try: rate_sign=float(physics_cfg.get('rate_sign',1.0)); dt=float(physics_cfg.get('dt'))
                    except (TypeError,ValueError): rate_sign=math.nan; dt=math.nan
                    if len(states)!=4 or len(drives)!=3:
                        reasons.append('QUATERNION_KINEMATICS requires four state_columns and three angular-rate drive_columns')
                    elif tuple(contract.target_columns) != states:
                        reasons.append('QUATERNION_KINEMATICS state_columns must exactly match modeling_contract.target_columns order')
                    elif any(c not in contract.input_columns for c in (*states,*drives)):
                        reasons.append('QUATERNION_KINEMATICS state_columns and drive_columns must all be present in modeling_contract.input_columns')
                    elif not np.isfinite(dt) or dt<=0:
                        reasons.append('QUATERNION_KINEMATICS requires a finite positive temporal_physics.dt')
                    elif composition not in ('LEFT','RIGHT'):
                        reasons.append('QUATERNION_KINEMATICS composition must be LEFT or RIGHT')
                    elif not np.isfinite(rate_sign) or rate_sign not in (-1.0,1.0):
                        reasons.append('QUATERNION_KINEMATICS rate_sign must be +1 or -1')
                    elif semantics != 'UNIT_QUATERNION':
                        reasons.append('QUATERNION_KINEMATICS requires constraints.temporal_target_semantics=UNIT_QUATERNION')
                    else:
                        candidates.append(TemporalCandidate('quaternion_kinematics', TemporalModelKind.QUATERNION_KINEMATICS, 'ONE_STEP_FORECAST', 'mechanism-preserving unit-quaternion propagation from admitted angular-rate kinematics'))
                elif kind == 'AFFINE_STATE_SPACE':
                    try: dt=float(physics_cfg.get('dt'))
                    except (TypeError,ValueError): dt=math.nan
                    if not states:
                        reasons.append('AFFINE_STATE_SPACE requires non-empty state_columns')
                    elif tuple(contract.target_columns) != states:
                        reasons.append('AFFINE_STATE_SPACE state_columns must exactly match modeling_contract.target_columns order')
                    elif any(c not in contract.input_columns for c in (*states,*drives)):
                        reasons.append('AFFINE_STATE_SPACE state_columns and drive_columns must all be present in modeling_contract.input_columns')
                    elif not np.isfinite(dt) or dt<=0:
                        reasons.append('AFFINE_STATE_SPACE requires a finite positive temporal_physics.dt matching the declared regular sample interval')
                    else:
                        candidates.append(TemporalCandidate('affine_state_space', TemporalModelKind.AFFINE_STATE_SPACE, 'ONE_STEP_FORECAST', 'deterministic K1 affine discrete state-space candidate x[k+1]=A x[k]+B u[k]+c fitted on train trajectories only'))
                else:
                    reasons.append(f"unsupported temporal_physics.kind {kind!r}; native adapters support K1_INTEGRATOR, BOUNDED_INTEGRATOR, QUATERNION_KINEMATICS and AFFINE_STATE_SPACE")
            elif contract.mechanism_usage == MechanismUsage.REQUIRED:
                reasons.append('mechanism_usage=REQUIRED needs an explicit native temporal_physics adapter before temporal-design can proceed')

        if contract.mechanism_usage != MechanismUsage.REQUIRED:
            candidates.append(TemporalCandidate('linear_ar', TemporalModelKind.LINEAR_AR, 'ONE_STEP_FORECAST', 'low-complexity linear autoregressive baseline'))
            if min_len >= 4:
                candidates.append(TemporalCandidate('tcn', TemporalModelKind.TCN, 'ONE_STEP_FORECAST', 'causal/local convolutional temporal predictor'))
                candidates.append(TemporalCandidate('gru', TemporalModelKind.GRU, 'ONE_STEP_FORECAST', 'recurrent latent-state temporal predictor'))
            if min_len >= 6:
                candidates.append(TemporalCandidate('transformer', TemporalModelKind.TRANSFORMER, 'ONE_STEP_FORECAST', 'attention-based temporal predictor'))
    else:
        reasons.append(f'native temporal-design does not yet implement objective {contract.objective.value}')

    candidate_ids=[c.id.casefold() for c in candidates]
    duplicate_ids=sorted({cid for cid in candidate_ids if candidate_ids.count(cid)>1})
    if duplicate_ids:
        reasons.append(f'duplicate temporal candidate ids after qualification: {duplicate_ids}')

    allowed=(contract.constraints or {}).get('temporal_candidates')
    if allowed is not None:
        wanted={str(x).upper() for x in allowed}
        known={x.value for x in TemporalModelKind}
        known_ids={c.id.upper() for c in candidates}
        bad=sorted(wanted-known-known_ids)
        if bad:
            reasons.append(f'unknown temporal candidate kinds/ids requested: {bad}')
        candidates=[c for c in candidates if c.kind.value in wanted or c.id.upper() in wanted]

    semantics=_target_semantics(contract)
    if semantics not in ('EUCLIDEAN','UNIT_QUATERNION'):
        reasons.append(f'unsupported temporal_target_semantics {semantics!r}')
    elif semantics=='UNIT_QUATERNION' and len(contract.target_columns)!=4:
        reasons.append('UNIT_QUATERNION target semantics requires exactly four target_columns')

    status = 'READY'
    if reasons:
        hard = any(('missing' in r or 'required' in r or 'requires' in r or 'crosses split' in r or 'not strictly increasing' in r or 'regular sampling' in r or 'no train' in r or 'no validation' in r or r.startswith('forecast_evaluation contract invalid') or r.startswith('forecast_evaluation horizon_steps=')) for r in reasons)
        status = 'UNSUPPORTED' if hard else 'AMBIGUOUS'
    q = spec.get('qualification') or {}
    min_train = q.get('min_train_independent_contexts') if isinstance(q, Mapping) else None
    min_val = q.get('min_validation_independent_contexts') if isinstance(q, Mapping) else None
    scope=str(q.get('evidence_scope','PRODUCTION')).upper() if isinstance(q, Mapping) else 'PRODUCTION'
    if (min_train is None) != (min_val is None):
        status='UNSUPPORTED'; reasons.append('independent-context minimums must be supplied together or both omitted')
    elif min_train is not None:
        if len(train_ctx) < int(min_train):
            status = 'AMBIGUOUS'; reasons.append(f'train independent contexts {len(train_ctx)} < required {int(min_train)}')
        if len(val_ctx) < int(min_val):
            status = 'AMBIGUOUS'; reasons.append(f'validation independent contexts {len(val_ctx)} < required {int(min_val)}')
    elif scope=='PRODUCTION':
        status='AMBIGUOUS'; reasons.append('production temporal qualification requires evidence-justified min_train_independent_contexts and min_validation_independent_contexts; qualification fails closed')
    elif scope=='BLINDED_FOR_VALIDATION' and (len(train_ctx)<2 or len(val_ctx)<2):
        status='AMBIGUOUS'; reasons.append('BLINDED_FOR_VALIDATION needs at least two train and two validation independent trajectories')
    if len(candidates) < 2 and status == 'READY' and contract.mechanism_usage != MechanismUsage.REQUIRED:
        status = 'AMBIGUOUS'; reasons.append('fewer than two qualified temporal candidates; automatic architecture comparison is not possible')
    if len(candidates) == 0 and status == 'READY':
        status = 'AMBIGUOUS'; reasons.append('no qualified temporal candidate remains after applying the modeling contract')
    try:
        context_support, supported_val_ctx = _context_support_analysis(spec, contract, data)
    except ValueError as exc:
        status = 'UNSUPPORTED'; reasons.append(str(exc)); context_support, supported_val_ctx = {'enabled': False}, set()
    policy_cs = dict((context_support or {}).get('policy') or {})
    if context_support.get('enabled') and policy_cs.get('selection_scope') == 'IN_SUPPORT_ONLY':
        nsel = int(context_support.get('selection_validation_contexts', 0))
        if nsel < 2:
            status = 'AMBIGUOUS'; reasons.append('temporal context-support leaves fewer than two in-support validation contexts; architecture comparison fails closed')
        elif int(context_support.get('validation_out_of_support_contexts', 0)) > 0:
            reasons.append(
                f"{int(context_support['validation_out_of_support_contexts'])} validation contexts fall outside the empirical train context envelope and are reserved for extrapolation audit, not architecture ranking"
            )
    return TemporalQualification(status, context_col, time_col, split_col, len(train_ctx), len(val_ctx), min_len, max_len, regular, rel, tuple(reasons), tuple(candidates), context_support, forecast_evaluation)


class _SummaryMLP(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4*n_in, hidden), nn.ReLU(), nn.Linear(hidden, n_out))
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B,L,F = x.shape
        mask = torch.arange(L, device=x.device)[None,:] < lengths[:,None]
        mf = mask[:,:,None].to(x.dtype)
        n = mf.sum(1).clamp_min(1.0)
        mean = (x*mf).sum(1)/n
        var = (((x-mean[:,None,:])**2)*mf).sum(1)/n
        big = torch.tensor(1e9, dtype=x.dtype, device=x.device)
        mn = torch.where(mask[:,:,None], x, big).amin(1)
        mx = torch.where(mask[:,:,None], x, -big).amax(1)
        return self.net(torch.cat([mean, torch.sqrt(var+1e-8), mn, mx], dim=1))



def _waveform_time_domain_features(x: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """TIME_DOMAIN_V1 feature bank on an explicitly declared sample region.

    x is already standardized per input channel using train-only statistics.
    The bank is intentionally small and deterministic: mean, std, min, max,
    RMS, peak-to-peak, line integral, kurtosis and clearance factor per channel.
    It does not select a region from labels and therefore does not own diagnosis
    semantics; the region of interest must be declared by the modeling contract.
    """
    if start < 0 or end <= start or end > x.shape[1]:
        raise ValueError('invalid waveform feature sample window')
    z=x[:,start:end,:]
    mean=z.mean(1)
    centered=z-mean[:,None,:]
    var=(centered*centered).mean(1)
    std=torch.sqrt(var+1e-12)
    mn=z.amin(1); mx=z.amax(1)
    rms=torch.sqrt((z*z).mean(1)+1e-12)
    ptp=mx-mn
    if z.shape[1] >= 2:
        line=torch.abs(z[:,1:,:]-z[:,:-1,:]).sum(1)
    else:
        line=torch.zeros_like(mean)
    kurt=(centered.pow(4).mean(1))/(var.pow(2)+1e-12)
    clearance=torch.abs(z).amax(1)/(torch.sqrt(torch.abs(z)+1e-12).mean(1).pow(2)+1e-12)
    return torch.cat([mean,std,mn,mx,rms,ptp,line,kurt,clearance],dim=1)


class _WaveformFeatureLinear(nn.Module):
    def __init__(self,n_in:int,n_out:int,start:int,end:int,feature_mean,feature_std):
        super().__init__(); self.n_in=n_in; self.start=int(start); self.end=int(end)
        self.register_buffer('feature_mean',torch.as_tensor(feature_mean,dtype=torch.float64))
        self.register_buffer('feature_std',torch.as_tensor(feature_std,dtype=torch.float64))
        self.out=nn.Linear(9*n_in,n_out)
    def forward(self,x:torch.Tensor,lengths:torch.Tensor)->torch.Tensor:
        if torch.any(lengths < self.end):
            raise ValueError('waveform prediction context is shorter than the frozen feature sample window')
        f=_waveform_time_domain_features(x,self.start,self.end)
        z=(f-self.feature_mean)/self.feature_std
        return self.out(z)


def _waveform_order_features(
    x: torch.Tensor, lengths: torch.Tensor, start: int, end: int,
    signal_indices: Sequence[int], rotation_index: int,
    input_mean: Sequence[float], input_std: Sequence[float],
    sample_rate_hz: float, carrier_band_hz: Sequence[float],
    characteristic_orders: Sequence[float], harmonics: int, sidebands: Sequence[float],
) -> torch.Tensor:
    """Envelope/order features tied to an explicitly declared rotating mechanism.

    The feature bank does not infer shaft speed or characteristic orders from labels.
    Rotation frequency is read from a context-constant input column; characteristic
    orders, carrier band, harmonics and sidebands are declared by the modeling contract.
    """
    if start < 0 or end <= start or end > x.shape[1] or torch.any(lengths < end):
        raise ValueError('invalid or unsupported waveform order sample window')
    if not signal_indices:
        raise ValueError('waveform order representation requires at least one signal channel')
    z=x[:,start:end,:].index_select(2,torch.as_tensor(list(signal_indices),dtype=torch.long,device=x.device))
    z=z-z.mean(1,keepdim=True)
    n=z.shape[1]
    freqs=torch.fft.rfftfreq(n,d=1.0/float(sample_rate_hz),device=x.device,dtype=x.dtype)
    lo,hi=float(carrier_band_hz[0]),float(carrier_band_hz[1])
    mask=((freqs>=lo)&(freqs<=hi)).to(x.dtype)[None,:,None]
    spec=torch.fft.rfft(z,dim=1)
    filtered=torch.fft.irfft(spec*mask,n=n,dim=1)
    # Analytic signal / Hilbert envelope using the standard FFT multiplier.
    F=torch.fft.fft(filtered,dim=1)
    h=torch.zeros(n,dtype=x.dtype,device=x.device)
    h[0]=1.0
    if n%2==0:
        h[n//2]=1.0; h[1:n//2]=2.0
    else:
        h[1:(n+1)//2]=2.0
    analytic=torch.fft.ifft(F*h[None,:,None],dim=1)
    env=torch.abs(analytic); env=env-env.mean(1,keepdim=True)
    win=torch.hann_window(n,periodic=True,dtype=x.dtype,device=x.device)
    amp=torch.abs(torch.fft.rfft(env*win[None,:,None],dim=1))
    norm=amp[:,1:,:].median(dim=1).values.clamp_min(1e-12)
    mu=torch.as_tensor(input_mean,dtype=x.dtype,device=x.device)
    sd=torch.as_tensor(input_std,dtype=x.dtype,device=x.device)
    rot=x[:,0,rotation_index]*sd[rotation_index]+mu[rotation_index]
    if torch.any(~torch.isfinite(rot)) or torch.any(rot<=0):
        raise ValueError('rotation frequency must be finite and positive for order-domain prediction')
    desired=[]
    for base in characteristic_orders:
        for hnum in range(1,int(harmonics)+1):
            for sb in sidebands:
                desired.append(float(hnum)*float(base)+float(sb))
    feats=[]
    df=float(sample_rate_hz)/float(n)
    nb=amp.shape[1]
    for order in desired:
        if order<=0:
            a=torch.zeros((x.shape[0],len(signal_indices)),dtype=x.dtype,device=x.device)
        else:
            pos=(rot*order/df).clamp(0.0,float(nb-1))
            i0=torch.floor(pos).to(torch.long); i1=torch.clamp(i0+1,max=nb-1); w=(pos-i0.to(x.dtype))[:,None]
            bidx=torch.arange(x.shape[0],device=x.device)[:,None]
            cidx=torch.arange(len(signal_indices),device=x.device)[None,:]
            a0=amp[bidx,i0[:,None],cidx]; a1=amp[bidx,i1[:,None],cidx]
            a=a0*(1.0-w)+a1*w
        feats.append(torch.log1p(a/norm))
    return torch.cat(feats,dim=1)


class _WaveformOrderLinear(nn.Module):
    def __init__(self,n_in:int,n_out:int,start:int,end:int,signal_indices:Sequence[int],rotation_index:int,
                 input_mean,input_std,sample_rate_hz:float,carrier_band_hz:Sequence[float],
                 characteristic_orders:Sequence[float],harmonics:int,sidebands:Sequence[float],feature_mean,feature_std):
        super().__init__(); self.n_in=n_in; self.start=int(start); self.end=int(end)
        self.signal_indices=tuple(map(int,signal_indices)); self.rotation_index=int(rotation_index)
        self.sample_rate_hz=float(sample_rate_hz); self.carrier_band_hz=tuple(map(float,carrier_band_hz))
        self.characteristic_orders=tuple(map(float,characteristic_orders)); self.harmonics=int(harmonics); self.sidebands=tuple(map(float,sidebands))
        self.register_buffer('input_mean_buf',torch.as_tensor(input_mean,dtype=torch.float64))
        self.register_buffer('input_std_buf',torch.as_tensor(input_std,dtype=torch.float64))
        self.register_buffer('feature_mean',torch.as_tensor(feature_mean,dtype=torch.float64))
        self.register_buffer('feature_std',torch.as_tensor(feature_std,dtype=torch.float64))
        nf=len(self.signal_indices)*len(self.characteristic_orders)*self.harmonics*len(self.sidebands)
        self.out=nn.Linear(nf,n_out)
    def forward(self,x:torch.Tensor,lengths:torch.Tensor)->torch.Tensor:
        f=_waveform_order_features(x,lengths,self.start,self.end,self.signal_indices,self.rotation_index,
                                   self.input_mean_buf,self.input_std_buf,self.sample_rate_hz,self.carrier_band_hz,
                                   self.characteristic_orders,self.harmonics,self.sidebands)
        z=(f-self.feature_mean)/self.feature_std
        return self.out(z)


class _TCNHead(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden: int, *, classification: bool):
        super().__init__(); self.classification=classification
        self.conv = nn.Sequential(
            nn.Conv1d(n_in, hidden, 3, padding=2, dilation=1), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=4, dilation=2), nn.ReLU(),
        )
        self.out = nn.Linear(hidden, n_out)
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        L=x.shape[1]
        z=self.conv(x.transpose(1,2))[:,:,:L].transpose(1,2)
        if self.classification:
            mask=(torch.arange(L,device=x.device)[None,:] < lengths[:,None])[:,:,None].to(x.dtype)
            h=(z*mask).sum(1)/mask.sum(1).clamp_min(1.0)
        else:
            h=z[torch.arange(z.shape[0],device=z.device), lengths-1]
        return self.out(h)


class _GRUHead(nn.Module):
    def __init__(self, n_in:int, n_out:int, hidden:int):
        super().__init__(); self.gru=nn.GRU(n_in,hidden,batch_first=True); self.out=nn.Linear(hidden,n_out)
    def forward(self,x:torch.Tensor,lengths:torch.Tensor)->torch.Tensor:
        packed=nn.utils.rnn.pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False)
        _,h=self.gru(packed)
        return self.out(h[-1])


class _TransformerHead(nn.Module):
    def __init__(self,n_in:int,n_out:int,hidden:int,max_len:int,*,classification:bool):
        super().__init__(); self.classification=classification; self.proj=nn.Linear(n_in,hidden)
        heads=4 if hidden%4==0 and hidden>=4 else 1
        layer=nn.TransformerEncoderLayer(d_model=hidden,nhead=heads,dim_feedforward=max(2*hidden,8),dropout=0.0,batch_first=True,activation='gelu')
        self.enc=nn.TransformerEncoder(layer,num_layers=1); self.out=nn.Linear(hidden,n_out)
        pos=torch.arange(max_len,dtype=torch.float32)[:,None]
        div=torch.exp(torch.arange(0,hidden,2,dtype=torch.float32)*(-math.log(10000.0)/max(hidden,1)))
        pe=torch.zeros(max_len,hidden); pe[:,0::2]=torch.sin(pos*div)
        if hidden>1: pe[:,1::2]=torch.cos(pos*div[:pe[:,1::2].shape[1]])
        self.register_buffer('pe',pe,persistent=False)
    def forward(self,x:torch.Tensor,lengths:torch.Tensor)->torch.Tensor:
        L=x.shape[1]; mask=torch.arange(L,device=x.device)[None,:] >= lengths[:,None]
        z=self.proj(x)+self.pe[:L].to(x.device,x.dtype)[None,:,:]
        z=self.enc(z,src_key_padding_mask=mask)
        if self.classification:
            keep=(~mask)[:,:,None].to(x.dtype); h=(z*keep).sum(1)/keep.sum(1).clamp_min(1.0)
        else:
            h=z[torch.arange(z.shape[0],device=z.device),lengths-1]
        return self.out(h)


class _LinearAR(nn.Module):
    def __init__(self,n_in:int,n_out:int,lookback:int):
        super().__init__(); self.lookback=lookback; self.out=nn.Linear(n_in*lookback,n_out)
    def forward(self,x:torch.Tensor,lengths:torch.Tensor)->torch.Tensor:
        return self.out(x.reshape(x.shape[0],-1))


class _K1Integrator(nn.Module):
    def __init__(self, state_indices: Sequence[int], drive_indices: Sequence[int], gains: Sequence[float], dt: float,
                 input_mean: Sequence[float], input_std: Sequence[float], target_mean: Sequence[float], target_std: Sequence[float]):
        super().__init__()
        self.register_buffer('state_indices', torch.as_tensor(list(state_indices), dtype=torch.long))
        self.register_buffer('drive_indices', torch.as_tensor(list(drive_indices), dtype=torch.long))
        self.register_buffer('gains', torch.as_tensor(list(gains), dtype=torch.float64))
        self.register_buffer('dt_value', torch.as_tensor(float(dt), dtype=torch.float64))
        self.register_buffer('input_mean_buf', torch.as_tensor(list(input_mean), dtype=torch.float64))
        self.register_buffer('input_std_buf', torch.as_tensor(list(input_std), dtype=torch.float64))
        self.register_buffer('target_mean_buf', torch.as_tensor(list(target_mean), dtype=torch.float64))
        self.register_buffer('target_std_buf', torch.as_tensor(list(target_std), dtype=torch.float64))
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        last = x[torch.arange(x.shape[0], device=x.device), lengths-1]
        raw = last * self.input_std_buf + self.input_mean_buf
        state = raw.index_select(1, self.state_indices)
        drive = raw.index_select(1, self.drive_indices)
        pred_raw = state + self.dt_value * drive * self.gains
        return (pred_raw - self.target_mean_buf) / self.target_std_buf



class _BoundedIntegrator(nn.Module):
    """Mechanism-preserving bounded discrete integrator.

    Predicts x[k+1] = clip(x[k] + dt*g*u[k], lower, upper).  Bounds may
    be fixed engineering constants or observed configuration columns.  The
    adapter does not infer why a bound changed and therefore carries no fault
    semantics.
    """
    def __init__(self, state_indices, drive_indices, gains, dt, lower_fixed, upper_fixed,
                 lower_indices, upper_indices, input_mean, input_std, target_mean, target_std):
        super().__init__()
        self.register_buffer('state_indices', torch.as_tensor(list(state_indices), dtype=torch.long))
        self.register_buffer('drive_indices', torch.as_tensor(list(drive_indices), dtype=torch.long))
        self.register_buffer('gains', torch.as_tensor(list(gains), dtype=torch.float64))
        self.register_buffer('dt_value', torch.as_tensor(float(dt), dtype=torch.float64))
        self.register_buffer('lower_fixed', torch.as_tensor(list(lower_fixed), dtype=torch.float64))
        self.register_buffer('upper_fixed', torch.as_tensor(list(upper_fixed), dtype=torch.float64))
        self.register_buffer('lower_indices', torch.as_tensor(list(lower_indices), dtype=torch.long))
        self.register_buffer('upper_indices', torch.as_tensor(list(upper_indices), dtype=torch.long))
        self.register_buffer('input_mean_buf', torch.as_tensor(list(input_mean), dtype=torch.float64))
        self.register_buffer('input_std_buf', torch.as_tensor(list(input_std), dtype=torch.float64))
        self.register_buffer('target_mean_buf', torch.as_tensor(list(target_mean), dtype=torch.float64))
        self.register_buffer('target_std_buf', torch.as_tensor(list(target_std), dtype=torch.float64))
    def forward(self,x,lengths):
        last=x[torch.arange(x.shape[0],device=x.device),lengths-1]
        raw=last*self.input_std_buf+self.input_mean_buf
        state=raw.index_select(1,self.state_indices); drive=raw.index_select(1,self.drive_indices)
        pred=state+self.dt_value*drive*self.gains
        lower=self.lower_fixed.unsqueeze(0).expand_as(pred)
        upper=self.upper_fixed.unsqueeze(0).expand_as(pred)
        if len(self.lower_indices): lower=raw.index_select(1,self.lower_indices)
        if len(self.upper_indices): upper=raw.index_select(1,self.upper_indices)
        if torch.any(lower>upper): raise ValueError('BOUNDED_INTEGRATOR lower bound exceeds upper bound')
        pred=torch.maximum(torch.minimum(pred,upper),lower)
        return (pred-self.target_mean_buf)/self.target_std_buf


class _AffineStateSpace(nn.Module):
    """Deterministic affine discrete-time state-space predictor.

    The admitted family is x[k+1] = A x[k] + B u[k] + c.  A/B/c are fit
    from training trajectories only.  This adapter does not infer state
    meaning, sparsity, stability, or physical sign constraints; those remain
    part of the upstream admitted mechanism/modeling contract.
    """
    def __init__(self, state_indices: Sequence[int], drive_indices: Sequence[int], A: Sequence[Sequence[float]],
                 B: Sequence[Sequence[float]], bias: Sequence[float],
                 input_mean: Sequence[float], input_std: Sequence[float], target_mean: Sequence[float], target_std: Sequence[float]):
        super().__init__()
        self.register_buffer('state_indices', torch.as_tensor(list(state_indices), dtype=torch.long))
        self.register_buffer('drive_indices', torch.as_tensor(list(drive_indices), dtype=torch.long))
        self.register_buffer('A', torch.as_tensor(A, dtype=torch.float64))
        self.register_buffer('B', torch.as_tensor(B, dtype=torch.float64))
        self.register_buffer('bias', torch.as_tensor(list(bias), dtype=torch.float64))
        self.register_buffer('input_mean_buf', torch.as_tensor(list(input_mean), dtype=torch.float64))
        self.register_buffer('input_std_buf', torch.as_tensor(list(input_std), dtype=torch.float64))
        self.register_buffer('target_mean_buf', torch.as_tensor(list(target_mean), dtype=torch.float64))
        self.register_buffer('target_std_buf', torch.as_tensor(list(target_std), dtype=torch.float64))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        last = x[torch.arange(x.shape[0], device=x.device), lengths-1]
        raw = last * self.input_std_buf + self.input_mean_buf
        state = raw.index_select(1, self.state_indices)
        if len(self.drive_indices):
            drive = raw.index_select(1, self.drive_indices)
            pred_raw = state @ self.A.T + drive @ self.B.T + self.bias
        else:
            pred_raw = state @ self.A.T + self.bias
        return (pred_raw - self.target_mean_buf) / self.target_std_buf


class _QuaternionKinematics(nn.Module):
    """Mechanism-preserving unit-quaternion propagation from angular rate.

    This is a bounded native temporal-physics adapter.  It does not infer frame
    conventions: composition (LEFT/RIGHT) and rate_sign must be declared by the
    admitted mechanism contract.
    """
    def __init__(self, state_indices: Sequence[int], drive_indices: Sequence[int], dt: float, composition: str, rate_sign: float,
                 input_mean: Sequence[float], input_std: Sequence[float], target_mean: Sequence[float], target_std: Sequence[float]):
        super().__init__()
        self.register_buffer('state_indices', torch.as_tensor(list(state_indices), dtype=torch.long))
        self.register_buffer('drive_indices', torch.as_tensor(list(drive_indices), dtype=torch.long))
        self.register_buffer('dt_value', torch.as_tensor(float(dt), dtype=torch.float64))
        self.register_buffer('rate_sign_value', torch.as_tensor(float(rate_sign), dtype=torch.float64))
        self.composition = str(composition).upper()
        self.register_buffer('input_mean_buf', torch.as_tensor(list(input_mean), dtype=torch.float64))
        self.register_buffer('input_std_buf', torch.as_tensor(list(input_std), dtype=torch.float64))
        self.register_buffer('target_mean_buf', torch.as_tensor(list(target_mean), dtype=torch.float64))
        self.register_buffer('target_std_buf', torch.as_tensor(list(target_std), dtype=torch.float64))

    @staticmethod
    def _qmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        lw, lv = left[:, :1], left[:, 1:]
        rw, rv = right[:, :1], right[:, 1:]
        scalar = lw * rw - torch.sum(lv * rv, dim=1, keepdim=True)
        vector = lw * rv + rw * lv + torch.cross(lv, rv, dim=1)
        return torch.cat((scalar, vector), dim=1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        last = x[torch.arange(x.shape[0], device=x.device), lengths-1]
        raw = last * self.input_std_buf + self.input_mean_buf
        q = raw.index_select(1, self.state_indices)
        q = q / torch.linalg.vector_norm(q, dim=1, keepdim=True).clamp_min(1e-12)
        omega = raw.index_select(1, self.drive_indices) * self.rate_sign_value
        norm = torch.linalg.vector_norm(omega, dim=1, keepdim=True)
        half = 0.5 * norm * self.dt_value
        vec_scale = torch.where(norm > 1e-12, torch.sin(half) / norm, 0.5 * self.dt_value * torch.ones_like(norm))
        dq = torch.cat((torch.cos(half), omega * vec_scale), dim=1)
        if self.composition == 'LEFT':
            pred_raw = self._qmul(dq, q)
        elif self.composition == 'RIGHT':
            pred_raw = self._qmul(q, dq)
        else:
            raise ValueError(f'unsupported quaternion composition {self.composition!r}')
        pred_raw = pred_raw / torch.linalg.vector_norm(pred_raw, dim=1, keepdim=True).clamp_min(1e-12)
        return (pred_raw - self.target_mean_buf) / self.target_std_buf

def _set_seed(seed:int)->None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _pad_sequences(seq: Sequence[np.ndarray], n_features: int) -> tuple[torch.Tensor, torch.Tensor]:
    lengths=np.asarray([len(x) for x in seq],dtype=np.int64)
    L=int(lengths.max())
    out=np.zeros((len(seq),L,n_features),dtype=np.float64)
    for i,a in enumerate(seq): out[i,:len(a),:]=a
    return torch.as_tensor(out,dtype=torch.float64), torch.as_tensor(lengths,dtype=torch.long)


def _context_split_sequences(spec: Mapping[str,Any], contract: ModelingContract, data: Mapping[str,np.ndarray]):
    context_col,time_col=_context_and_time(spec,contract); split_col,trvals,vavals=_split_values(spec)
    contexts=_ordered_contexts(data,context_col,time_col)
    split=np.asarray(data[split_col]).astype(str)
    Xraw=np.column_stack([np.asarray(data[c],dtype=float) for c in contract.input_columns])
    train_rows=np.concatenate([idx for _,idx in contexts if str(split[idx[0]]) in trvals])
    mu=Xraw[train_rows].mean(0); sd=Xraw[train_rows].std(0); sd=np.where(sd>1e-12,sd,1.0)
    X=(Xraw-mu)/sd
    out={'train':[], 'val':[]}
    cs, supported_val = _context_support_analysis(spec, contract, data)
    use_supported_val = bool(cs.get('enabled')) and dict(cs.get('policy') or {}).get('selection_scope') == 'IN_SUPPORT_ONLY'
    for c,idx in contexts:
        v=str(split[idx[0]])
        key='train' if v in trvals else ('val' if v in vavals else None)
        if key == 'val' and use_supported_val and c not in supported_val:
            continue
        if key: out[key].append((c,idx,X[idx]))
    return out,mu,sd


def _macro_f1(y_true: np.ndarray,y_pred: np.ndarray,n_classes:int)->float:
    fs=[]
    for c in range(n_classes):
        tp=np.sum((y_true==c)&(y_pred==c)); fp=np.sum((y_true!=c)&(y_pred==c)); fn=np.sum((y_true==c)&(y_pred!=c))
        p=tp/(tp+fp) if tp+fp else 0.0; r=tp/(tp+fn) if tp+fn else 0.0
        fs.append(2*p*r/(p+r) if p+r else 0.0)
    return float(np.mean(fs))



def _train_waveform_feature_linear(train_seq,train_y,val_seq,val_y,n_features:int,n_classes:int,contract:ModelingContract,waveform_config:Mapping[str,Any]|None=None):
    cfg=dict(waveform_config or (contract.constraints or {}).get('waveform_representation') or {})
    start,end=map(int,cfg['sample_window']); alpha=float(cfg.get('ridge_alpha',1.0))
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError('waveform_representation.ridge_alpha must be finite and non-negative')
    xtr,ltr=_pad_sequences(train_seq,n_features); xva,lva=_pad_sequences(val_seq,n_features)
    with torch.no_grad():
        ftr=_waveform_time_domain_features(xtr,start,end).cpu().numpy()
        fva=_waveform_time_domain_features(xva,start,end).cpu().numpy()
    fmu=ftr.mean(0); fsd=ftr.std(0); fsd=np.where(fsd>1e-12,fsd,1.0)
    ztr=(ftr-fmu)/fsd; zva=(fva-fmu)/fsd
    Y=-np.ones((len(train_y),n_classes),dtype=float); Y[np.arange(len(train_y)),np.asarray(train_y,dtype=int)]=1.0
    counts=np.bincount(np.asarray(train_y,dtype=int),minlength=n_classes).astype(float)
    sw=np.asarray([1.0/max(counts[int(i)],1.0) for i in train_y],dtype=float); sw=sw/max(sw.mean(),1e-12)
    A=np.column_stack([ztr,np.ones(len(ztr))]); W=sw[:,None]
    reg=np.eye(A.shape[1])*alpha; reg[-1,-1]=0.0
    lhs=A.T@(W*A)+reg; rhs=A.T@(W*Y)
    try: theta=np.linalg.solve(lhs,rhs)
    except np.linalg.LinAlgError: theta=np.linalg.lstsq(lhs,rhs,rcond=None)[0]
    model=_WaveformFeatureLinear(n_features,n_classes,start,end,fmu,fsd).double().eval()
    with torch.no_grad():
        model.out.weight.copy_(torch.as_tensor(theta[:-1].T,dtype=torch.float64))
        model.out.bias.copy_(torch.as_tensor(theta[-1],dtype=torch.float64))
        pred=model(xva,lva).argmax(1).cpu().numpy()
    f1=_macro_f1(np.asarray(val_y),pred,n_classes)
    return max(1.0-f1,1e-8),model

def _sequence_vector_risk(pred_z: np.ndarray, y_z: np.ndarray) -> float:
    pred_z=np.asarray(pred_z,dtype=float); y_z=np.asarray(y_z,dtype=float)
    return max(float(np.sqrt(np.mean((pred_z-y_z)**2))),1e-8)


def _train_waveform_order_linear(train_seq,train_y,val_seq,val_y,n_features:int,n_classes:int,contract:ModelingContract,input_mean,input_std,waveform_config:Mapping[str,Any]|None=None):
    wcfg=dict(waveform_config or (contract.constraints or {}).get('waveform_representation') or {}); ocfg=dict(wcfg.get('order_domain') or {})
    start,end=map(int,wcfg['sample_window']); alpha=float(ocfg.get('ridge_alpha',1.0))
    signal_cols=tuple(map(str,ocfg['signal_columns'])); rot_col=str(ocfg['rotation_frequency_column'])
    signal_indices=[contract.input_columns.index(c) for c in signal_cols]; rotation_index=contract.input_columns.index(rot_col)
    sample_rate=float(ocfg['sample_rate_hz']); carrier=tuple(map(float,ocfg['carrier_band_hz'])); orders=tuple(map(float,ocfg['characteristic_orders']))
    harmonics=int(ocfg.get('harmonics',1)); sidebands=tuple(map(float,ocfg.get('sidebands',[0.0])))
    if not np.isfinite(alpha) or alpha<0: raise ValueError('waveform order-domain ridge_alpha must be finite and non-negative')
    xtr,ltr=_pad_sequences(train_seq,n_features); xva,lva=_pad_sequences(val_seq,n_features)
    with torch.no_grad():
        ftr=_waveform_order_features(xtr,ltr,start,end,signal_indices,rotation_index,input_mean,input_std,sample_rate,carrier,orders,harmonics,sidebands).cpu().numpy()
        fva=_waveform_order_features(xva,lva,start,end,signal_indices,rotation_index,input_mean,input_std,sample_rate,carrier,orders,harmonics,sidebands).cpu().numpy()
    fmu=ftr.mean(0); fsd=ftr.std(0); fsd=np.where(fsd>1e-12,fsd,1.0)
    ztr=(ftr-fmu)/fsd
    Y=-np.ones((len(train_y),n_classes),dtype=float); Y[np.arange(len(train_y)),np.asarray(train_y,dtype=int)]=1.0
    counts=np.bincount(np.asarray(train_y,dtype=int),minlength=n_classes).astype(float)
    sw=np.asarray([1.0/max(counts[int(i)],1.0) for i in train_y],dtype=float); sw=sw/max(sw.mean(),1e-12)
    A=np.column_stack([ztr,np.ones(len(ztr))]); W=sw[:,None]; reg=np.eye(A.shape[1])*alpha; reg[-1,-1]=0.0
    lhs=A.T@(W*A)+reg; rhs=A.T@(W*Y)
    try: theta=np.linalg.solve(lhs,rhs)
    except np.linalg.LinAlgError: theta=np.linalg.lstsq(lhs,rhs,rcond=None)[0]
    model=_WaveformOrderLinear(n_features,n_classes,start,end,signal_indices,rotation_index,input_mean,input_std,sample_rate,carrier,orders,harmonics,sidebands,fmu,fsd).double().eval()
    with torch.no_grad():
        model.out.weight.copy_(torch.as_tensor(theta[:-1].T,dtype=torch.float64)); model.out.bias.copy_(torch.as_tensor(theta[-1],dtype=torch.float64))
        pred=model(xva,lva).argmax(1).cpu().numpy()
    return max(1.0-_macro_f1(np.asarray(val_y),pred,n_classes),1e-8),model


def _train_waveform_feature_regression(train_seq,train_y_z,val_seq,val_y_z,n_features:int,n_out:int,contract:ModelingContract):
    cfg=dict((contract.constraints or {}).get('waveform_representation') or {})
    start,end=map(int,cfg['sample_window']); alpha=float(cfg.get('ridge_alpha',1.0))
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError('waveform_representation.ridge_alpha must be finite and non-negative')
    xtr,ltr=_pad_sequences(train_seq,n_features); xva,lva=_pad_sequences(val_seq,n_features)
    with torch.no_grad():
        ftr=_waveform_time_domain_features(xtr,start,end).cpu().numpy()
        fva=_waveform_time_domain_features(xva,start,end).cpu().numpy()
    fmu=ftr.mean(0); fsd=ftr.std(0); fsd=np.where(fsd>1e-12,fsd,1.0)
    ztr=(ftr-fmu)/fsd
    A=np.column_stack([ztr,np.ones(len(ztr))])
    reg=np.eye(A.shape[1])*alpha; reg[-1,-1]=0.0
    lhs=A.T@A+reg; rhs=A.T@np.asarray(train_y_z,dtype=float)
    try: theta=np.linalg.solve(lhs,rhs)
    except np.linalg.LinAlgError: theta=np.linalg.lstsq(lhs,rhs,rcond=None)[0]
    model=_WaveformFeatureLinear(n_features,n_out,start,end,fmu,fsd).double().eval()
    with torch.no_grad():
        model.out.weight.copy_(torch.as_tensor(theta[:-1].T,dtype=torch.float64))
        model.out.bias.copy_(torch.as_tensor(theta[-1],dtype=torch.float64))
        pred=model(xva,lva).cpu().numpy()
    return _sequence_vector_risk(pred,val_y_z),model


def _train_sequence_regression_candidate(kind:TemporalModelKind, train_seq, train_y_z, val_seq, val_y_z, n_features:int, n_out:int, seed:int, cfg:Mapping[str,Any]):
    _set_seed(seed); hidden=int(cfg.get('hidden_size',24)); epochs=int(cfg.get('epochs',40)); lr=float(cfg.get('lr',0.01)); wd=float(cfg.get('weight_decay',1e-4))
    xtr,ltr=_pad_sequences(train_seq,n_features); xva,lva=_pad_sequences(val_seq,n_features); max_len=max(xtr.shape[1],xva.shape[1])
    if kind is TemporalModelKind.SUMMARY_MLP: model=_SummaryMLP(n_features,n_out,hidden).double()
    elif kind is TemporalModelKind.TCN: model=_TCNHead(n_features,n_out,hidden,classification=True).double()
    elif kind is TemporalModelKind.GRU: model=_GRUHead(n_features,n_out,hidden).double()
    elif kind is TemporalModelKind.TRANSFORMER: model=_TransformerHead(n_features,n_out,hidden,max_len,classification=True).double()
    else: raise ValueError(kind)
    Ytr=torch.as_tensor(train_y_z,dtype=torch.float64); Yva=torch.as_tensor(val_y_z,dtype=torch.float64)
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd); best=None; best_loss=math.inf
    for _ in range(epochs):
        model.train(); opt.zero_grad(); pred=model(xtr,ltr); loss=nn.functional.mse_loss(pred,Ytr); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl=float(nn.functional.mse_loss(model(xva,lva),Yva).item())
        if vl<best_loss: best_loss=vl; best={k:v.detach().clone() for k,v in model.state_dict().items()}
    if best: model.load_state_dict(best)
    model.eval()
    with torch.no_grad(): pred=model(xva,lva).cpu().numpy()
    return _sequence_vector_risk(pred,val_y_z),model

def _train_class_candidate(kind:TemporalModelKind, train_seq, train_y, val_seq, val_y, n_features:int, n_classes:int, seed:int, cfg:Mapping[str,Any]):
    _set_seed(seed); hidden=int(cfg.get('hidden_size',24)); epochs=int(cfg.get('epochs',40)); lr=float(cfg.get('lr',0.01)); wd=float(cfg.get('weight_decay',1e-4))
    xtr,ltr=_pad_sequences(train_seq,n_features); xva,lva=_pad_sequences(val_seq,n_features)
    max_len=max(xtr.shape[1],xva.shape[1])
    if kind is TemporalModelKind.SUMMARY_MLP: model=_SummaryMLP(n_features,n_classes,hidden).double()
    elif kind is TemporalModelKind.TCN: model=_TCNHead(n_features,n_classes,hidden,classification=True).double()
    elif kind is TemporalModelKind.GRU: model=_GRUHead(n_features,n_classes,hidden).double()
    elif kind is TemporalModelKind.TRANSFORMER: model=_TransformerHead(n_features,n_classes,hidden,max_len,classification=True).double()
    else: raise ValueError(kind)
    ytr=torch.as_tensor(train_y,dtype=torch.long); opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd)
    counts=np.bincount(train_y,minlength=n_classes).astype(float); weights=np.where(counts>0,1.0/np.maximum(counts,1),0.0); weights=weights/max(weights.mean(),1e-12)
    w=torch.as_tensor(weights,dtype=torch.float64)
    best=None; best_loss=math.inf
    for _ in range(epochs):
        model.train(); opt.zero_grad(); logits=model(xtr,ltr); loss=nn.functional.cross_entropy(logits,ytr,weight=w); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl=float(nn.functional.cross_entropy(model(xva,lva),torch.as_tensor(val_y,dtype=torch.long)).item())
        if vl<best_loss: best_loss=vl; best={k:v.detach().clone() for k,v in model.state_dict().items()}
    if best: model.load_state_dict(best)
    model.eval()
    with torch.no_grad(): pred=model(xva,lva).argmax(1).cpu().numpy()
    f1=_macro_f1(np.asarray(val_y),pred,n_classes)
    return max(1.0-f1,1e-8),model


def _forecast_windows(items, input_cols_n:int, yraw:np.ndarray, lookback:int):
    xs=[]; ys=[]; groups=[]
    for c,idx,x in items:
        if len(idx)<=lookback: continue
        for j in range(lookback,len(idx)):
            xs.append(x[j-lookback:j]); ys.append(yraw[idx[j]]); groups.append(c)
    if not xs: raise ValueError('no temporal forecast windows available at the requested lookback')
    return np.stack(xs),np.asarray(ys,dtype=float),np.asarray(groups,dtype=str)


def _target_semantics(contract: ModelingContract) -> str:
    constraints = dict(contract.constraints or {})
    return str(constraints.get('temporal_target_semantics', 'EUCLIDEAN')).upper()


def _context_forecast_risk(pred_z: np.ndarray, y_z: np.ndarray, groups: np.ndarray, semantics: str, ymu: np.ndarray, ysd: np.ndarray) -> float:
    pred_z=np.asarray(pred_z,dtype=float); y_z=np.asarray(y_z,dtype=float)
    if semantics == 'UNIT_QUATERNION':
        if pred_z.shape[1] != 4 or y_z.shape[1] != 4:
            raise ValueError('UNIT_QUATERNION target semantics requires exactly four target columns')
        pred=pred_z*ysd+ymu; y=y_z*ysd+ymu
        pn=np.linalg.norm(pred,axis=1,keepdims=True); yn=np.linalg.norm(y,axis=1,keepdims=True)
        if np.any(pn <= 1e-12) or np.any(yn <= 1e-12):
            raise ValueError('UNIT_QUATERNION prediction/target contains a near-zero quaternion')
        pred=pred/pn; y=y/yn
        dots=np.clip(np.abs(np.sum(pred*y,axis=1)),0.0,1.0)
        err=2.0*np.arccos(dots)
        per=[float(np.sqrt(np.mean(err[groups==c]**2))) for c in np.unique(groups)]
    else:
        per=[float(np.sqrt(np.mean((pred_z[groups==c]-y_z[groups==c])**2))) for c in np.unique(groups)]
    return max(float(np.mean(per)),1e-8)



def _recursive_closed_loop_endpoint_predictions(
    model: nn.Module,
    x_context: np.ndarray,
    lookback: int,
    horizon_steps: int,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return endpoint predictions and their context-relative row positions.

    The initial v2.16 rollout contract is intentionally closed: model inputs
    and targets have identical columns/order.  Predicted targets are converted
    back to raw engineering units, then re-normalized as the next model input.
    No future exogenous measurements are fabricated.
    """
    x = np.asarray(x_context, dtype=float)
    if x.ndim != 2:
        raise ValueError('recursive rollout expects a 2-D context sequence')
    if x.shape[1] != len(target_mean) or len(input_mean) != len(target_mean):
        raise ValueError('recursive rollout requires a closed input/target state with matching dimensions')
    preds: list[np.ndarray] = []
    positions: list[int] = []
    stop = len(x) - int(horizon_steps) + 1
    for start in range(int(lookback), stop):
        win = np.asarray(x[start-lookback:start], dtype=float).copy()
        pred_z = None
        for _ in range(int(horizon_steps)):
            tx = torch.as_tensor(win[None, :, :], dtype=torch.float64)
            lengths = torch.full((1,), int(lookback), dtype=torch.long)
            with torch.no_grad():
                pred_z = model(tx, lengths).detach().cpu().numpy()[0]
            pred_raw = pred_z * target_std + target_mean
            next_input_z = (pred_raw - input_mean) / input_std
            win = np.vstack([win[1:], next_input_z])
        preds.append(np.asarray(pred_z, dtype=float))
        positions.append(start + int(horizon_steps) - 1)
    if not preds:
        return np.empty((0, len(target_mean)), dtype=float), np.empty((0,), dtype=int)
    return np.stack(preds), np.asarray(positions, dtype=int)


def _recursive_closed_loop_endpoint_risk(
    model: nn.Module,
    validation_items,
    y_z_all: np.ndarray,
    lookback: int,
    horizon_steps: int,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> float:
    per_context: list[float] = []
    for _, idx, x_context in validation_items:
        pred_z, pos = _recursive_closed_loop_endpoint_predictions(
            model, x_context, lookback, horizon_steps,
            input_mean, input_std, target_mean, target_std,
        )
        if len(pos) == 0:
            continue
        target_z = np.asarray(y_z_all, dtype=float)[np.asarray(idx, dtype=int)[pos]]
        per_context.append(float(np.sqrt(np.mean((pred_z - target_z) ** 2))))
    if not per_context:
        raise ValueError('recursive rollout evaluation produced no validation endpoints at the requested horizon')
    return max(float(np.mean(per_context)), 1e-8)


def _train_forecast_candidate(kind:TemporalModelKind,xtr,ytr,xva,yva,gva,seed:int,cfg:Mapping[str,Any], semantics:str, ymu:np.ndarray, ysd:np.ndarray):
    hidden=int(cfg.get('hidden_size',24)); epochs=int(cfg.get('epochs',50)); lr=float(cfg.get('lr',0.01)); wd=float(cfg.get('weight_decay',1e-4))
    Xtr=torch.as_tensor(xtr,dtype=torch.float64); Xva=torch.as_tensor(xva,dtype=torch.float64); lengths=torch.full((len(Xtr),),Xtr.shape[1],dtype=torch.long); lv=torch.full((len(Xva),),Xva.shape[1],dtype=torch.long)
    n_in=xtr.shape[2]; n_out=ytr.shape[1]; L=xtr.shape[1]
    if kind is TemporalModelKind.LINEAR_AR:
        model=_LinearAR(n_in,n_out,L).double()
        flat=np.asarray(xtr,dtype=float).reshape(len(xtr),-1)
        A=np.column_stack([flat,np.ones(len(flat))])
        reg=np.eye(A.shape[1])*wd; reg[-1,-1]=0.0
        lhs=A.T@A+reg; rhs=A.T@np.asarray(ytr,dtype=float)
        try: theta=np.linalg.solve(lhs,rhs)
        except np.linalg.LinAlgError: theta=np.linalg.lstsq(lhs,rhs,rcond=None)[0]
        with torch.no_grad():
            model.out.weight.copy_(torch.as_tensor(theta[:-1].T,dtype=torch.float64))
            model.out.bias.copy_(torch.as_tensor(theta[-1],dtype=torch.float64))
        model.eval()
        with torch.no_grad(): pred=model(Xva,lv).cpu().numpy()
        return _context_forecast_risk(pred,yva,gva,semantics,ymu,ysd),model
    _set_seed(seed)
    if kind is TemporalModelKind.TCN: model=_TCNHead(n_in,n_out,hidden,classification=False).double()
    elif kind is TemporalModelKind.GRU: model=_GRUHead(n_in,n_out,hidden).double()
    elif kind is TemporalModelKind.TRANSFORMER: model=_TransformerHead(n_in,n_out,hidden,L,classification=False).double()
    else: raise ValueError(kind)
    Ytr=torch.as_tensor(ytr,dtype=torch.float64); Yva=torch.as_tensor(yva,dtype=torch.float64)
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd); best=None; best_loss=math.inf
    for _ in range(epochs):
        model.train(); opt.zero_grad(); loss=nn.functional.mse_loss(model(Xtr,lengths),Ytr); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl=float(nn.functional.mse_loss(model(Xva,lv),Yva).item())
        if vl<best_loss: best_loss=vl; best={k:v.detach().clone() for k,v in model.state_dict().items()}
    if best: model.load_state_dict(best)
    model.eval()
    with torch.no_grad(): pred=model(Xva,lv).cpu().numpy()
    return _context_forecast_risk(pred,yva,gva,semantics,ymu,ysd),model


def _train_k1_integrator_candidate(
    xtr: np.ndarray, ytr: np.ndarray, xva: np.ndarray, yva: np.ndarray, gva: np.ndarray,
    contract: ModelingContract, mu: np.ndarray, sd: np.ndarray, ymu: np.ndarray, ysd: np.ndarray, semantics: str
):
    cfg=dict((contract.constraints or {}).get('temporal_physics') or {})
    states=tuple(map(str,cfg.get('state_columns') or ())); drives=tuple(map(str,cfg.get('drive_columns') or ()))
    dt=float(cfg['dt']); state_idx=[contract.input_columns.index(c) for c in states]; drive_idx=[contract.input_columns.index(c) for c in drives]
    last=np.asarray(xtr[:,-1,:],dtype=float)*sd+mu
    target=np.asarray(ytr,dtype=float)*ysd+ymu
    state=last[:,state_idx]; drive=last[:,drive_idx]
    mode=str(cfg.get('gain_mode','LEARNED')).upper()
    if mode=='FIXED':
        gains=np.asarray(cfg.get('fixed_gains'),dtype=float)
        if gains.shape != (len(states),) or not np.all(np.isfinite(gains)):
            raise ValueError('K1_INTEGRATOR FIXED gain_mode requires finite fixed_gains matching state_columns')
    elif mode=='LEARNED':
        delta=target-state; z=dt*drive; gains=[]
        for i in range(len(states)):
            denom=float(np.dot(z[:,i],z[:,i]))
            if denom <= 1e-14:
                raise ValueError(f'K1_INTEGRATOR drive column {drives[i]!r} has insufficient excitation to identify gain')
            gains.append(float(np.dot(z[:,i],delta[:,i])/denom))
        gains=np.asarray(gains,dtype=float)
    else:
        raise ValueError(f'unsupported K1_INTEGRATOR gain_mode {mode!r}')
    model=_K1Integrator(state_idx,drive_idx,gains,dt,mu,sd,ymu,ysd).double().eval()
    Xva=torch.as_tensor(xva,dtype=torch.float64); lv=torch.full((len(Xva),),Xva.shape[1],dtype=torch.long)
    with torch.no_grad(): pred=model(Xva,lv).cpu().numpy()
    return _context_forecast_risk(pred,yva,gva,semantics,ymu,ysd),model,gains



def _bounded_cfg(contract, mu, sd, ymu, ysd):
    cfg=dict((contract.constraints or {}).get('temporal_physics') or {})
    states=tuple(map(str,cfg.get('state_columns') or ())); drives=tuple(map(str,cfg.get('drive_columns') or ()))
    n=len(states); state_idx=[contract.input_columns.index(c) for c in states]; drive_idx=[contract.input_columns.index(c) for c in drives]
    lower_cols=tuple(map(str,cfg.get('lower_bound_columns') or ())); upper_cols=tuple(map(str,cfg.get('upper_bound_columns') or ()))
    lower_idx=[contract.input_columns.index(c) for c in lower_cols] if lower_cols else []
    upper_idx=[contract.input_columns.index(c) for c in upper_cols] if upper_cols else []
    lower_fixed=np.asarray(cfg.get('lower_bounds') if cfg.get('lower_bounds') is not None else [-np.inf]*n,dtype=float)
    upper_fixed=np.asarray(cfg.get('upper_bounds') if cfg.get('upper_bounds') is not None else [np.inf]*n,dtype=float)
    return cfg,states,drives,state_idx,drive_idx,lower_idx,upper_idx,lower_fixed,upper_fixed


def _train_bounded_integrator_candidate(xtr,ytr,xva,yva,gva,contract,mu,sd,ymu,ysd,semantics):
    cfg,states,drives,state_idx,drive_idx,lower_idx,upper_idx,lower_fixed,upper_fixed=_bounded_cfg(contract,mu,sd,ymu,ysd)
    dt=float(cfg['dt']); last=np.asarray(xtr[:,-1,:],dtype=float)*sd+mu; target=np.asarray(ytr,dtype=float)*ysd+ymu
    state=last[:,state_idx]; drive=last[:,drive_idx]; mode=str(cfg.get('gain_mode','LEARNED')).upper()
    if mode=='FIXED':
        gains=np.asarray(cfg.get('fixed_gains'),dtype=float)
        if gains.shape!=(len(states),) or not np.all(np.isfinite(gains)): raise ValueError('BOUNDED_INTEGRATOR FIXED gain_mode requires finite fixed_gains matching state_columns')
    elif mode=='LEARNED':
        # Fit gain on unsaturated interior transitions only when bound information is available.
        gains=[]; delta=target-state; z=dt*drive
        if lower_idx: lower=last[:,lower_idx]
        else: lower=np.broadcast_to(lower_fixed,(len(last),len(states)))
        if upper_idx: upper=last[:,upper_idx]
        else: upper=np.broadcast_to(upper_fixed,(len(last),len(states)))
        for i in range(len(states)):
            interior=(target[:,i] > lower[:,i]+1e-9) & (target[:,i] < upper[:,i]-1e-9)
            zi=z[interior,i]; di=delta[interior,i]
            denom=float(np.dot(zi,zi))
            if denom<=1e-14: raise ValueError(f'BOUNDED_INTEGRATOR drive column {drives[i]!r} has insufficient interior excitation to identify gain')
            gains.append(float(np.dot(zi,di)/denom))
        gains=np.asarray(gains,dtype=float)
    else: raise ValueError(f'unsupported BOUNDED_INTEGRATOR gain_mode {mode!r}')
    model=_BoundedIntegrator(state_idx,drive_idx,gains,dt,lower_fixed,upper_fixed,lower_idx,upper_idx,mu,sd,ymu,ysd).double().eval()
    Xva=torch.as_tensor(xva,dtype=torch.float64); lv=torch.full((len(Xva),),Xva.shape[1],dtype=torch.long)
    with torch.no_grad(): pred=model(Xva,lv).cpu().numpy()
    return _context_forecast_risk(pred,yva,gva,semantics,ymu,ysd),model,gains


def _train_affine_state_space_candidate(
    xtr: np.ndarray, ytr: np.ndarray, xva: np.ndarray, yva: np.ndarray, gva: np.ndarray,
    contract: ModelingContract, mu: np.ndarray, sd: np.ndarray, ymu: np.ndarray, ysd: np.ndarray, semantics: str
):
    cfg=dict((contract.constraints or {}).get('temporal_physics') or {})
    states=tuple(map(str,cfg.get('state_columns') or ())); drives=tuple(map(str,cfg.get('drive_columns') or ()))
    state_idx=[contract.input_columns.index(c) for c in states]; drive_idx=[contract.input_columns.index(c) for c in drives]
    last=np.asarray(xtr[:,-1,:],dtype=float)*sd+mu
    target=np.asarray(ytr,dtype=float)*ysd+ymu
    state=last[:,state_idx]
    drive=last[:,drive_idx] if drive_idx else np.empty((len(last),0),dtype=float)
    include_bias=bool(cfg.get('include_bias',True))
    ridge=float(cfg.get('ridge',1e-8))
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError('AFFINE_STATE_SPACE ridge must be a finite non-negative number')
    parts=[state,drive]
    if include_bias: parts.append(np.ones((len(state),1),dtype=float))
    design=np.concatenate(parts,axis=1)
    reg=np.eye(design.shape[1],dtype=float)*ridge
    if include_bias: reg[-1,-1]=0.0
    lhs=design.T@design+reg; rhs=design.T@target
    try: theta=np.linalg.solve(lhs,rhs)
    except np.linalg.LinAlgError: theta=np.linalg.lstsq(lhs,rhs,rcond=None)[0]
    ns=len(states); nd=len(drives)
    A=theta[:ns,:].T
    B=theta[ns:ns+nd,:].T if nd else np.zeros((len(states),0),dtype=float)
    bias=theta[-1,:] if include_bias else np.zeros(len(states),dtype=float)
    model=_AffineStateSpace(state_idx,drive_idx,A,B,bias,mu,sd,ymu,ysd).double().eval()
    Xva=torch.as_tensor(xva,dtype=torch.float64); lv=torch.full((len(Xva),),Xva.shape[1],dtype=torch.long)
    with torch.no_grad(): pred=model(Xva,lv).cpu().numpy()
    return _context_forecast_risk(pred,yva,gva,semantics,ymu,ysd),model,A,B,bias



def _train_quaternion_kinematics_candidate(
    xva: np.ndarray, yva: np.ndarray, gva: np.ndarray, contract: ModelingContract,
    mu: np.ndarray, sd: np.ndarray, ymu: np.ndarray, ysd: np.ndarray, semantics: str
):
    cfg=dict((contract.constraints or {}).get('temporal_physics') or {})
    states=tuple(map(str,cfg.get('state_columns') or ())); drives=tuple(map(str,cfg.get('drive_columns') or ()))
    state_idx=[contract.input_columns.index(c) for c in states]; drive_idx=[contract.input_columns.index(c) for c in drives]
    model=_QuaternionKinematics(state_idx,drive_idx,float(cfg['dt']),str(cfg.get('composition','LEFT')),float(cfg.get('rate_sign',1.0)),mu,sd,ymu,ysd).double().eval()
    Xva=torch.as_tensor(xva,dtype=torch.float64); lv=torch.full((len(Xva),),Xva.shape[1],dtype=torch.long)
    with torch.no_grad(): pred=model(Xva,lv).cpu().numpy()
    return _context_forecast_risk(pred,yva,gva,semantics,ymu,ysd),model

def _training_cfg(spec:Mapping[str,Any])->dict[str,Any]:
    c=dict((spec.get('modeling_contract') or {}).get('constraints') or {})
    t=dict(c.get('temporal_training') or {})
    return t


def run_temporal_design(spec_path:str|Path,data_path:str|Path,out_dir:str|Path)->TemporalDesignSummary:
    from .goal_design import _read_csv
    spec=json.loads(Path(spec_path).read_text(encoding='utf-8')); data=_read_csv(Path(data_path)); contract=ModelingContract.from_mapping(spec.get('modeling_contract') or {})
    q=qualify_temporal_design(spec,data); out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    (out/'temporal_qualification.json').write_text(json.dumps(q.to_dict(),indent=2,ensure_ascii=False),encoding='utf-8')
    if q.status!='READY':
        s=TemporalDesignSummary(q.status,None,str(out),{},None,q.to_dict(),'UNRESOLVED')
        (out/'temporal_design_result.json').write_text(json.dumps(s.to_dict(),indent=2,ensure_ascii=False),encoding='utf-8'); return s
    seqs,mu,sd=_context_split_sequences(spec,contract,data); cfg=_training_cfg(spec); seeds=tuple(map(int,cfg.get('seeds') or (101,202,303)))
    if len(seeds)<2: raise ValueError('temporal design requires at least two optimizer seeds/restarts')
    risks={}; final_models={}; metadata={}; one_step_risks=None
    task='SEQUENCE_TO_CLASS' if contract.objective==ModelingObjective.CATEGORICAL_PREDICTION else ('SEQUENCE_TO_VECTOR' if contract.objective==ModelingObjective.STATIC_REGRESSION else 'ONE_STEP_FORECAST')
    if task=='SEQUENCE_TO_CLASS':
        labels=np.asarray(data[contract.label_column]).astype(str); train_labels=[]; val_labels=[]; train_seq=[]; val_seq=[]
        all_train=set()
        for c,idx,x in seqs['train']:
            u=np.unique(labels[idx]);
            if len(u)!=1: raise ValueError(f'label must be constant within context {c!r}')
            all_train.add(str(u[0])); train_labels.append(str(u[0])); train_seq.append(x)
        classes=sorted(all_train); cmap={c:i for i,c in enumerate(classes)}
        for c,idx,x in seqs['val']:
            u=np.unique(labels[idx]);
            if len(u)!=1: raise ValueError(f'label must be constant within context {c!r}')
            if str(u[0]) not in cmap: raise ValueError(f'validation class {u[0]!r} is unseen in training')
            val_labels.append(str(u[0])); val_seq.append(x)
        ytr=np.asarray([cmap[x] for x in train_labels]); yva=np.asarray([cmap[x] for x in val_labels])
        n_features=len(contract.input_columns); n_classes=len(classes)
        for cand in q.candidates:
            vals=[]; models=[]; extra={}
            candidate_wcfg=dict((cand.config or {}).get('waveform_representation') or (contract.constraints or {}).get('waveform_representation') or {})
            if cand.kind is TemporalModelKind.WAVEFORM_FEATURE_LINEAR:
                r,m=_train_waveform_feature_linear(train_seq,ytr,val_seq,yva,n_features,n_classes,contract,candidate_wcfg)
                vals=[r for _ in seeds]; models=[m for _ in seeds]
                extra={'waveform_representation':candidate_wcfg}
            elif cand.kind is TemporalModelKind.WAVEFORM_ORDER_LINEAR:
                r,m=_train_waveform_order_linear(train_seq,ytr,val_seq,yva,n_features,n_classes,contract,mu,sd,candidate_wcfg)
                vals=[r for _ in seeds]; models=[m for _ in seeds]
                extra={'waveform_representation':candidate_wcfg}
            else:
                for seed in seeds:
                    r,m=_train_class_candidate(cand.kind,train_seq,ytr,val_seq,yva,n_features,n_classes,seed,cfg); vals.append(r); models.append(m)
            risks[cand.id]=vals; final_models[cand.id]=models[int(np.argmin(vals))]
            metadata[cand.id]={'kind':cand.kind.value,'task':task,'classes':classes,**extra}
    elif task=='SEQUENCE_TO_VECTOR':
        ycols=contract.target_columns
        train_seq=[]; val_seq=[]; train_y=[]; val_y=[]
        yraw=np.column_stack([np.asarray(data[c],dtype=float) for c in ycols])
        for c,idx,x in seqs['train']:
            vals=yraw[idx]
            if not np.allclose(vals,vals[0],rtol=0.0,atol=1e-12): raise ValueError(f'static regression targets must be constant within context {c!r}')
            train_seq.append(x); train_y.append(vals[0])
        for c,idx,x in seqs['val']:
            vals=yraw[idx]
            if not np.allclose(vals,vals[0],rtol=0.0,atol=1e-12): raise ValueError(f'static regression targets must be constant within context {c!r}')
            val_seq.append(x); val_y.append(vals[0])
        train_y=np.asarray(train_y,dtype=float); val_y=np.asarray(val_y,dtype=float)
        ymu=train_y.mean(0); ysd=train_y.std(0); ysd=np.where(ysd>1e-12,ysd,1.0)
        ytr=(train_y-ymu)/ysd; yva=(val_y-ymu)/ysd; n_features=len(contract.input_columns); n_out=len(ycols)
        for cand in q.candidates:
            vals=[]; models=[]; extra={}
            if cand.kind is TemporalModelKind.WAVEFORM_FEATURE_LINEAR:
                r,m=_train_waveform_feature_regression(train_seq,ytr,val_seq,yva,n_features,n_out,contract)
                vals=[r for _ in seeds]; models=[m for _ in seeds]; extra={'waveform_representation':dict((contract.constraints or {}).get('waveform_representation') or {})}
            else:
                for seed in seeds:
                    r,m=_train_sequence_regression_candidate(cand.kind,train_seq,ytr,val_seq,yva,n_features,n_out,seed,cfg); vals.append(r); models.append(m)
            risks[cand.id]=vals; final_models[cand.id]=models[int(np.argmin(vals))]
            metadata[cand.id]={'kind':cand.kind.value,'task':task,'target_columns':list(ycols),'target_mean':ymu.tolist(),'target_std':ysd.tolist(),**extra}
    else:
        ycols=contract.target_columns or contract.input_columns
        yraw=np.column_stack([np.asarray(data[c],dtype=float) for c in ycols])
        tr_idx=np.concatenate([idx for _,idx,_ in seqs['train']]); ymu=yraw[tr_idx].mean(0); ysd=yraw[tr_idx].std(0); ysd=np.where(ysd>1e-12,ysd,1.0); yz=(yraw-ymu)/ysd
        min_len=q.min_sequence_length; semantics=_target_semantics(contract); lookback=int(cfg.get('lookback',max(2,min(12,min_len//4 if min_len>=8 else min_len-1))))
        if lookback>=min_len: lookback=max(1,min_len-1)
        xtr,ytr,gtr=_forecast_windows(seqs['train'],len(contract.input_columns),yz,lookback); xva,yva,gva=_forecast_windows(seqs['val'],len(contract.input_columns),yz,lookback)
        forecast_eval=dict(q.forecast_evaluation or _forecast_evaluation_policy(contract,min_len))
        rollout_mode=forecast_eval.get('mode')=='RECURSIVE_ROLLOUT'
        horizon_steps=int(forecast_eval.get('horizon_steps',1))
        one_step_risks={}
        for cand in q.candidates:
            base_vals=[]; models=[]
            extra={}
            if cand.kind is TemporalModelKind.K1_INTEGRATOR:
                r,m,gains=_train_k1_integrator_candidate(xtr,ytr,xva,yva,gva,contract,mu,sd,ymu,ysd,semantics)
                base_vals=[r for _ in seeds]; models=[m for _ in seeds]; extra={'learned_gains':gains.tolist(),'temporal_physics':dict((contract.constraints or {}).get('temporal_physics') or {})}
            elif cand.kind is TemporalModelKind.BOUNDED_INTEGRATOR:
                r,m,gains=_train_bounded_integrator_candidate(xtr,ytr,xva,yva,gva,contract,mu,sd,ymu,ysd,semantics)
                base_vals=[r for _ in seeds]; models=[m for _ in seeds]; extra={'learned_gains':gains.tolist(),'temporal_physics':dict((contract.constraints or {}).get('temporal_physics') or {})}
            elif cand.kind is TemporalModelKind.AFFINE_STATE_SPACE:
                r,m,A,B,bias=_train_affine_state_space_candidate(xtr,ytr,xva,yva,gva,contract,mu,sd,ymu,ysd,semantics)
                base_vals=[r for _ in seeds]; models=[m for _ in seeds]; extra={'state_matrix_A':A.tolist(),'drive_matrix_B':B.tolist(),'bias':bias.tolist(),'temporal_physics':dict((contract.constraints or {}).get('temporal_physics') or {})}
            elif cand.kind is TemporalModelKind.QUATERNION_KINEMATICS:
                r,m=_train_quaternion_kinematics_candidate(xva,yva,gva,contract,mu,sd,ymu,ysd,semantics)
                base_vals=[r for _ in seeds]; models=[m for _ in seeds]; extra={'temporal_physics':dict((contract.constraints or {}).get('temporal_physics') or {})}
            else:
                for seed in seeds:
                    r,m=_train_forecast_candidate(cand.kind,xtr,ytr,xva,yva,gva,seed,cfg,semantics,ymu,ysd); base_vals.append(r); models.append(m)
            one_step_risks[cand.id]=list(base_vals)
            if rollout_mode:
                vals=[_recursive_closed_loop_endpoint_risk(m,seqs['val'],yz,lookback,horizon_steps,mu,sd,ymu,ysd) for m in models]
            else:
                vals=list(base_vals)
            risks[cand.id]=vals; final_models[cand.id]=models[int(np.argmin(vals))]
            metadata[cand.id]={
                'kind':cand.kind.value,'task':task,'lookback':lookback,'target_columns':list(ycols),
                'target_mean':ymu.tolist(),'target_std':ysd.tolist(),'target_semantics':semantics,
                'forecast_evaluation':forecast_eval,**extra
            }
    policy_cfg=dict((spec.get('selection') or {})); policy=ArchitectureSelectionPolicy(practical_ratio=float(policy_cfg.get('practical_ratio',0.80)),restart_stability=float(policy_cfg.get('restart_stability',0.75)))
    if len(risks)==1 and contract.mechanism_usage==MechanismUsage.REQUIRED:
        only=next(iter(risks)); med=float(np.median(np.asarray(risks[only],dtype=float)))
        selection_status='SELECTED'; selection_selected=only
        selection_dict={
            'status':'SELECTED','selected':only,'runner_up':None,'median_risk_ratio':0.0,
            'pairwise_practical_win_fraction':1.0,'candidate_median_risks':[[only,med]],
            'selection_basis':'REQUIRED_MECHANISM_CONTRACT',
            'recommendation':'Freeze the sole mechanism-preserving candidate allowed by the REQUIRED modeling contract; this is contract selection, not evidence that it outperforms excluded model families.'
        }
    else:
        selection=select_architecture(risks,policy=policy)
        selection_status=selection.status; selection_selected=selection.selected; selection_dict=selection.as_dict()
        if selection.status=='AMBIGUOUS':
            selection_dict['recommendation']='Retain the finite temporal model portfolio or collect broader independent trajectories; do not force a top-1 temporal architecture.'
        else:
            selection_dict['recommendation']='Freeze the selected temporal candidate for held-out modeling evaluation; diagnostic meaning remains outside M2N.'
    if task=='ONE_STEP_FORECAST' and selection_dict is not None:
        fe=dict(q.forecast_evaluation or {})
        selection_dict['forecast_evaluation']=fe
        if fe.get('mode')=='RECURSIVE_ROLLOUT':
            selection_dict['selection_basis']=f"RECURSIVE_ROLLOUT_H{int(fe.get('horizon_steps',1))}"
            if selection_status=='AMBIGUOUS':
                selection_dict['recommendation']='Retain the finite temporal model portfolio or collect broader independent trajectories at the declared rollout horizon; do not force a one-step winner onto a longer-horizon deployment contract.'
            else:
                selection_dict['recommendation']='Freeze the candidate selected under the declared recursive rollout horizon; one-step diagnostics are retained separately and diagnostic/prognostic meaning remains outside M2N.'
    deploy=[selection_selected] if selection_status=='SELECTED' else list(risks)
    model_dir=out/'models'; model_dir.mkdir(exist_ok=True)
    for cid in deploy:
        payload={'toolkit_format':'M2N_TEMPORAL_MODEL_V1','candidate_id':cid,'model_metadata':metadata[cid],'modeling_contract':contract.to_dict(),'input_columns':list(contract.input_columns),'input_mean':mu.tolist(),'input_std':sd.tolist(),'state_dict':final_models[cid].state_dict()}
        torch.save(payload,model_dir/f'{cid}.pt')
    result=TemporalDesignSummary(selection_status,selection_selected,str(out),risks,selection_dict,q.to_dict(),task,one_step_risks if task=='ONE_STEP_FORECAST' else None,dict(q.forecast_evaluation or {}) if task=='ONE_STEP_FORECAST' else None)
    (out/'temporal_design_result.json').write_text(json.dumps(result.to_dict(),indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'temporal_candidate_plan.json').write_text(json.dumps([x.to_dict() for x in q.candidates],indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'reproduce.txt').write_text(f'm2n temporal-design --spec "{spec_path}" --data "{data_path}" --out "{out}"\n',encoding='utf-8')
    return result


def _infer_model_from_payload(payload:Mapping[str,Any], max_len:int)->nn.Module:
    meta=payload['model_metadata']; kind=TemporalModelKind(meta['kind']); sd=payload['state_dict']; n_in=len(payload['input_columns']); task=meta['task']
    context_level=task in ('SEQUENCE_TO_CLASS','SEQUENCE_TO_VECTOR')
    classification=task=='SEQUENCE_TO_CLASS'
    if classification: n_out=len(meta['classes'])
    else: n_out=len(meta['target_columns'])
    if kind is TemporalModelKind.SUMMARY_MLP:
        hidden=int(sd['net.0.weight'].shape[0]); model=_SummaryMLP(n_in,n_out,hidden).double()
    elif kind is TemporalModelKind.WAVEFORM_FEATURE_LINEAR:
        wcfg=dict(meta.get('waveform_representation') or {}); start,end=map(int,wcfg['sample_window'])
        model=_WaveformFeatureLinear(n_in,n_out,start,end,sd['feature_mean'],sd['feature_std']).double()
    elif kind is TemporalModelKind.WAVEFORM_ORDER_LINEAR:
        wcfg=dict(meta.get('waveform_representation') or {}); ocfg=dict(wcfg.get('order_domain') or {}); start,end=map(int,wcfg['sample_window'])
        signal_indices=[payload['input_columns'].index(str(c)) for c in ocfg['signal_columns']]; rotation_index=payload['input_columns'].index(str(ocfg['rotation_frequency_column']))
        model=_WaveformOrderLinear(n_in,n_out,start,end,signal_indices,rotation_index,payload['input_mean'],payload['input_std'],float(ocfg['sample_rate_hz']),ocfg['carrier_band_hz'],ocfg['characteristic_orders'],int(ocfg.get('harmonics',1)),ocfg.get('sidebands',[0.0]),sd['feature_mean'],sd['feature_std']).double()
    elif kind is TemporalModelKind.LINEAR_AR:
        lookback=int(meta['lookback']); model=_LinearAR(n_in,n_out,lookback).double()
    elif kind is TemporalModelKind.K1_INTEGRATOR:
        pcfg=dict(meta['temporal_physics']); states=tuple(map(str,pcfg['state_columns'])); drives=tuple(map(str,pcfg['drive_columns']))
        state_idx=[payload['input_columns'].index(c) for c in states]; drive_idx=[payload['input_columns'].index(c) for c in drives]
        model=_K1Integrator(state_idx,drive_idx,meta['learned_gains'],float(pcfg['dt']),payload['input_mean'],payload['input_std'],meta['target_mean'],meta['target_std']).double()
    elif kind is TemporalModelKind.BOUNDED_INTEGRATOR:
        pcfg=dict(meta['temporal_physics']); states=tuple(map(str,pcfg['state_columns'])); drives=tuple(map(str,pcfg['drive_columns']))
        state_idx=[payload['input_columns'].index(c) for c in states]; drive_idx=[payload['input_columns'].index(c) for c in drives]
        lower_cols=tuple(map(str,pcfg.get('lower_bound_columns') or ())); upper_cols=tuple(map(str,pcfg.get('upper_bound_columns') or ()))
        lower_idx=[payload['input_columns'].index(c) for c in lower_cols] if lower_cols else []; upper_idx=[payload['input_columns'].index(c) for c in upper_cols] if upper_cols else []
        n=len(states); lower_fixed=pcfg.get('lower_bounds') if pcfg.get('lower_bounds') is not None else [-float('inf')]*n; upper_fixed=pcfg.get('upper_bounds') if pcfg.get('upper_bounds') is not None else [float('inf')]*n
        model=_BoundedIntegrator(state_idx,drive_idx,meta['learned_gains'],float(pcfg['dt']),lower_fixed,upper_fixed,lower_idx,upper_idx,payload['input_mean'],payload['input_std'],meta['target_mean'],meta['target_std']).double()
    elif kind is TemporalModelKind.AFFINE_STATE_SPACE:
        pcfg=dict(meta['temporal_physics']); states=tuple(map(str,pcfg['state_columns'])); drives=tuple(map(str,pcfg.get('drive_columns') or ()))
        state_idx=[payload['input_columns'].index(c) for c in states]; drive_idx=[payload['input_columns'].index(c) for c in drives]
        model=_AffineStateSpace(state_idx,drive_idx,meta['state_matrix_A'],meta['drive_matrix_B'],meta['bias'],payload['input_mean'],payload['input_std'],meta['target_mean'],meta['target_std']).double()
    elif kind is TemporalModelKind.QUATERNION_KINEMATICS:
        pcfg=dict(meta['temporal_physics']); states=tuple(map(str,pcfg['state_columns'])); drives=tuple(map(str,pcfg['drive_columns']))
        state_idx=[payload['input_columns'].index(c) for c in states]; drive_idx=[payload['input_columns'].index(c) for c in drives]
        model=_QuaternionKinematics(state_idx,drive_idx,float(pcfg['dt']),str(pcfg.get('composition','LEFT')),float(pcfg.get('rate_sign',1.0)),payload['input_mean'],payload['input_std'],meta['target_mean'],meta['target_std']).double()
    elif kind is TemporalModelKind.TCN:
        hidden=int(sd['conv.0.weight'].shape[0]); model=_TCNHead(n_in,n_out,hidden,classification=context_level).double()
    elif kind is TemporalModelKind.GRU:
        hidden=int(sd['gru.weight_hh_l0'].shape[1]); model=_GRUHead(n_in,n_out,hidden).double()
    elif kind is TemporalModelKind.TRANSFORMER:
        hidden=int(sd['proj.weight'].shape[0]); model=_TransformerHead(n_in,n_out,hidden,max_len,classification=context_level).double()
    else: raise ValueError(kind)
    model.load_state_dict(sd); model.eval(); return model


def predict_temporal_design(design_dir:str|Path,data_path:str|Path,out_path:str|Path)->Path:
    import csv
    from .goal_design import _read_csv
    ddir=Path(design_dir); data=_read_csv(Path(data_path)); result=json.loads((ddir/'temporal_design_result.json').read_text(encoding='utf-8'))
    model_files=sorted((ddir/'models').glob('*.pt'))
    if not model_files: raise ValueError('temporal design contains no deployable models')
    payloads=[torch.load(p,map_location='cpu',weights_only=True) for p in model_files]
    contract=ModelingContract.from_mapping(payloads[0]['modeling_contract'])
    # Build a minimal spec from the frozen qualification result for context/time semantics.
    q=result['qualification']; context_col=q['context_column']; time_col=q['time_column']
    if context_col not in data or time_col not in data: raise ValueError('prediction data is missing frozen temporal context/time columns')
    for c in contract.input_columns:
        if c not in data: raise ValueError(f'prediction data is missing input column {c!r}')
    contexts=_ordered_contexts(data,context_col,time_col)
    raw=np.column_stack([np.asarray(data[c],dtype=float) for c in contract.input_columns]); mu=np.asarray(payloads[0]['input_mean'],float); sd=np.asarray(payloads[0]['input_std'],float); X=(raw-mu)/sd
    task=payloads[0]['model_metadata']['task']; out=Path(out_path); out.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    if task=='SEQUENCE_TO_CLASS':
        seq=[X[idx] for _,idx in contexts]; xt,lens=_pad_sequences(seq,len(contract.input_columns)); max_len=xt.shape[1]
        classes=list(payloads[0]['model_metadata']['classes']); probs=[]
        for pl in payloads:
            model=_infer_model_from_payload(pl,max_len)
            with torch.no_grad(): probs.append(torch.softmax(model(xt,lens),dim=1).cpu().numpy())
        mean=np.mean(np.stack(probs),axis=0)
        for i,(cid,idx) in enumerate(contexts):
            duration=float(np.asarray(data[time_col],float)[idx[-1]]-np.asarray(data[time_col],float)[idx[0]]); support=_context_support_details_from_frozen(q,data,idx,len(idx),duration); r={context_col:cid,'time_start':float(np.asarray(data[time_col],float)[idx[0]]),'time_end':float(np.asarray(data[time_col],float)[idx[-1]]),'model_count':len(payloads),'m2n_context_length':int(len(idx)),'m2n_context_duration':duration,'m2n_context_support':support['overall'],'m2n_context_shape_support':support['shape'],'m2n_context_operating_support':support['operating'],'m2n_context_operating_distance':support['operating_distance']}
            for cls_i,cls in enumerate(classes): r[f'portfolio_score__{cls}']=float(mean[i,cls_i])
            for pi,pl in enumerate(payloads):
                cidm=pl['candidate_id']
                for cls_i,cls in enumerate(classes): r[f'{cidm}__score__{cls}']=float(probs[pi][i,cls_i])
            r['portfolio_argmax']=classes[int(np.argmax(mean[i]))]
            rows.append(r)
    elif task=='SEQUENCE_TO_VECTOR':
        seq=[X[idx] for _,idx in contexts]; xt,lens=_pad_sequences(seq,len(contract.input_columns)); max_len=xt.shape[1]
        ycols=list(payloads[0]['model_metadata']['target_columns']); ymu=np.asarray(payloads[0]['model_metadata']['target_mean'],float); ysd=np.asarray(payloads[0]['model_metadata']['target_std'],float)
        preds=[]
        for pl in payloads:
            model=_infer_model_from_payload(pl,max_len)
            with torch.no_grad(): preds.append(model(xt,lens).cpu().numpy()*ysd+ymu)
        mean=np.mean(np.stack(preds),axis=0)
        for i,(cid,idx) in enumerate(contexts):
            duration=float(np.asarray(data[time_col],float)[idx[-1]]-np.asarray(data[time_col],float)[idx[0]]); support=_context_support_details_from_frozen(q,data,idx,len(idx),duration)
            r={context_col:cid,'time_start':float(np.asarray(data[time_col],float)[idx[0]]),'time_end':float(np.asarray(data[time_col],float)[idx[-1]]),'model_count':len(payloads),'m2n_context_length':int(len(idx)),'m2n_context_duration':duration,'m2n_context_support':support['overall'],'m2n_context_shape_support':support['shape'],'m2n_context_operating_support':support['operating'],'m2n_context_operating_distance':support['operating_distance']}
            for k,col in enumerate(ycols):
                r[f'portfolio_prediction__{col}']=float(mean[i,k])
                if col in data:
                    vals=np.asarray(data[col],dtype=float)[idx]
                    if np.allclose(vals,vals[0],rtol=0.0,atol=1e-12):
                        actual=float(vals[0]); r[f'actual__{col}']=actual; r[f'portfolio_residual__{col}']=actual-float(mean[i,k])
            for pi,pl in enumerate(payloads):
                for k,col in enumerate(ycols):
                    r[f'{pl["candidate_id"]}__prediction__{col}']=float(preds[pi][i,k])
            rows.append(r)
    else:
        # Forecast each context with the frozen lookback used by each model.  Current v1 bundle requires one common lookback.
        lbs={int(pl['model_metadata']['lookback']) for pl in payloads}
        if len(lbs)!=1: raise ValueError('temporal forecast portfolio requires one common lookback in v1 bundle')
        lookback=next(iter(lbs)); ycols=list(payloads[0]['model_metadata']['target_columns']); ymu=np.asarray(payloads[0]['model_metadata']['target_mean'],float); ysd=np.asarray(payloads[0]['model_metadata']['target_std'],float)
        forecast_eval=dict(payloads[0]['model_metadata'].get('forecast_evaluation') or {'mode':'ONE_STEP','horizon_steps':1})
        forecast_mode=str(forecast_eval.get('mode','ONE_STEP')).upper(); forecast_horizon=int(forecast_eval.get('horizon_steps',1))
        if forecast_mode=='RECURSIVE_ROLLOUT' and tuple(contract.input_columns)!=tuple(contract.target_columns):
            raise ValueError('frozen recursive-rollout bundle is not closed: input_columns must equal target_columns')
        for cid,idx in contexts:
            if len(idx)<=lookback: continue
            windows=np.stack([X[idx[j-lookback:j]] for j in range(lookback,len(idx))]); tx=torch.as_tensor(windows,dtype=torch.float64); lens=torch.full((len(tx),),lookback,dtype=torch.long)
            preds=[]; models=[]
            semantics=str(payloads[0]['model_metadata'].get('target_semantics','EUCLIDEAN')).upper()
            for pl in payloads:
                model=_infer_model_from_payload(pl,lookback); models.append(model)
                with torch.no_grad(): p=model(tx,lens).cpu().numpy()*ysd+ymu
                if semantics=='UNIT_QUATERNION':
                    n=np.linalg.norm(p,axis=1,keepdims=True)
                    if np.any(n<=1e-12): raise ValueError('quaternion predictor produced a near-zero output')
                    p=p/n
                preds.append(p)
            if semantics=='UNIT_QUATERNION':
                aligned=[]; ref=preds[0]
                for p in preds:
                    signs=np.where(np.sum(p*ref,axis=1,keepdims=True)<0.0,-1.0,1.0)
                    aligned.append(p*signs)
                mean=np.mean(np.stack(aligned),axis=0)
                mean=mean/np.linalg.norm(mean,axis=1,keepdims=True).clip(min=1e-12)
            else:
                mean=np.mean(np.stack(preds),axis=0)

            # v2.16 deployment-native horizon output.  One-step columns are kept
            # unchanged; recursive endpoint columns are additive and populated
            # only at rows that are true H-step rollout endpoints.
            rollout_maps=[]; rollout_portfolio={}
            if forecast_mode=='RECURSIVE_ROLLOUT':
                for pl,model in zip(payloads,models):
                    pmu=np.asarray(pl['input_mean'],float); psd=np.asarray(pl['input_std'],float)
                    pymu=np.asarray(pl['model_metadata']['target_mean'],float); pysd=np.asarray(pl['model_metadata']['target_std'],float)
                    xctx=(raw[idx]-pmu)/psd
                    pred_z,pos=_recursive_closed_loop_endpoint_predictions(model,xctx,lookback,forecast_horizon,pmu,psd,pymu,pysd)
                    pred_raw=pred_z*pysd+pymu
                    rollout_maps.append({int(pp):np.asarray(v,float) for pp,v in zip(pos,pred_raw)})
                common_positions=sorted(set.intersection(*(set(m) for m in rollout_maps))) if rollout_maps else []
                for pp in common_positions:
                    rollout_portfolio[int(pp)]=np.mean(np.stack([m[int(pp)] for m in rollout_maps]),axis=0)

            for j,row_idx in enumerate(idx[lookback:]):
                rel_pos=lookback+j
                duration=float(np.asarray(data[time_col],float)[idx[-1]]-np.asarray(data[time_col],float)[idx[0]]); support=_context_support_details_from_frozen(q,data,idx,len(idx),duration); r={context_col:cid,time_col:float(np.asarray(data[time_col],float)[row_idx]),'model_count':len(payloads),'m2n_context_length':int(len(idx)),'m2n_context_duration':duration,'m2n_context_support':support['overall'],'m2n_context_shape_support':support['shape'],'m2n_context_operating_support':support['operating'],'m2n_context_operating_distance':support['operating_distance'],'m2n_forecast_mode':forecast_mode,'m2n_forecast_horizon_steps':forecast_horizon}
                for k,col in enumerate(ycols):
                    r[f'portfolio_prediction__{col}']=float(mean[j,k])
                    if col in data:
                        actual=float(np.asarray(data[col],dtype=float)[row_idx])
                        r[f'actual__{col}']=actual
                        r[f'portfolio_residual__{col}']=actual-float(mean[j,k])
                    if forecast_mode=='RECURSIVE_ROLLOUT':
                        key=f'portfolio_rollout_h{forecast_horizon}_prediction__{col}'
                        rkey=f'portfolio_rollout_h{forecast_horizon}_residual__{col}'
                        if rel_pos in rollout_portfolio:
                            rv=float(rollout_portfolio[rel_pos][k]); r[key]=rv
                            r[rkey]=(float(np.asarray(data[col],dtype=float)[row_idx])-rv) if col in data else float('nan')
                        else:
                            r[key]=float('nan'); r[rkey]=float('nan')
                for pi,pl in enumerate(payloads):
                    for k,col in enumerate(ycols):
                        predv=float(preds[pi][j,k]); r[f'{pl["candidate_id"]}__prediction__{col}']=predv
                        if col in data:
                            r[f'{pl["candidate_id"]}__residual__{col}']=float(np.asarray(data[col],dtype=float)[row_idx])-predv
                        if forecast_mode=='RECURSIVE_ROLLOUT':
                            key=f'{pl["candidate_id"]}__rollout_h{forecast_horizon}__prediction__{col}'
                            rkey=f'{pl["candidate_id"]}__rollout_h{forecast_horizon}__residual__{col}'
                            if rel_pos in rollout_maps[pi]:
                                rv=float(rollout_maps[pi][rel_pos][k]); r[key]=rv
                                r[rkey]=(float(np.asarray(data[col],dtype=float)[row_idx])-rv) if col in data else float('nan')
                            else:
                                r[key]=float('nan'); r[rkey]=float('nan')
                if semantics=='UNIT_QUATERNION' and all(col in data for col in ycols):
                    actual=np.asarray([float(np.asarray(data[col],dtype=float)[row_idx]) for col in ycols],dtype=float)
                    an=np.linalg.norm(actual)
                    if an>1e-12:
                        actual=actual/an
                        dot=float(np.clip(abs(np.dot(mean[j],actual)),0.0,1.0))
                        r['portfolio_angular_error_deg']=float(math.degrees(2.0*math.acos(dot)))
                        for pi,pl in enumerate(payloads):
                            dot=float(np.clip(abs(np.dot(preds[pi][j],actual)),0.0,1.0))
                            r[f'{pl["candidate_id"]}__angular_error_deg']=float(math.degrees(2.0*math.acos(dot)))
                rows.append(r)
    if not rows: raise ValueError('temporal prediction produced no rows')
    cols=list(rows[0])
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(rows)
    return out
