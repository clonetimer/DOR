from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import csv
import hashlib
import json

import numpy as np
import torch

from .goal_design import (
    SUPPORTED_TEMPLATES,
    SUPPORTED_SCHEMA_VERSIONS,
    _candidate_from_payload,
    _inference_required_columns,
    _evidence_coordinates,
    _predict,
    _read_csv,
    _split_indices,
    _target_output_names,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _as_jsonable(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


@dataclass(frozen=True)
class PractitionerCheckResult:
    status: str  # PASS | WARN | FAIL
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    actions: tuple[str, ...]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _required_columns(spec: Mapping[str, Any]) -> tuple[str, ...]:
    sup = spec.get('supervision', {})
    cols: list[str] = []
    template = spec.get('template')

    if template == 'second_order_mechanics':
        for key in ('position_column', 'velocity_column', 'target_acceleration_column'):
            if sup.get(key):
                cols.append(str(sup[key]))
        der = sup.get('derive_acceleration') or {}
        if der:
            if der.get('time_column'):
                cols.append(str(der['time_column']))
            if der.get('velocity_column'):
                cols.append(str(der['velocity_column']))
        cols.extend(map(str, (sup.get('input_columns') or {}).values()))
    elif template in ('separable_peak_response', 'generic_mechanism_graph'):
        cols.extend(map(str, (sup.get('input_columns') or {}).values()))
        cols.extend(map(str, (sup.get('output_columns') or {}).values()))
    dcfg = spec.get('data', {})
    if dcfg.get('split_column'):
        cols.append(str(dcfg['split_column']))
    if dcfg.get('time_column'):
        cols.append(str(dcfg['time_column']))
    qcfg = spec.get('qualification', {})
    if isinstance(qcfg, Mapping) and qcfg.get('independent_context_column'):
        cols.append(str(qcfg['independent_context_column']))
    mc = spec.get('modeling_contract', {})
    if isinstance(mc, Mapping):
        cols.extend(map(str, mc.get('input_columns') or []))
        cols.extend(map(str, mc.get('target_columns') or []))
        if mc.get('label_column'):
            cols.append(str(mc['label_column']))
        if mc.get('time_column'):
            cols.append(str(mc['time_column']))
    return tuple(dict.fromkeys(cols))


def validate_device_card(spec_path: str | Path) -> PractitionerCheckResult:
    p = Path(spec_path)
    errors: list[str] = []
    warnings: list[str] = []
    actions: list[str] = []
    diag: dict[str, Any] = {'spec_path': str(p)}
    if not p.exists():
        return PractitionerCheckResult('FAIL', (f'device card not found: {p}',), (), ('check --spec path',), diag)
    try:
        spec = json.loads(p.read_text(encoding='utf-8'))
    except Exception as e:
        return PractitionerCheckResult('FAIL', (f'invalid JSON: {e}',), (), ('fix JSON syntax before design',), diag)
    if not isinstance(spec, dict):
        errors.append('device card root must be a JSON object')
        return PractitionerCheckResult('FAIL', tuple(errors), (), ('use an object at the JSON root',), diag)

    diag['schema_version'] = spec.get('schema_version')
    diag['template'] = spec.get('template')
    if str(spec.get('schema_version')) not in SUPPORTED_SCHEMA_VERSIONS:
        errors.append(f'schema_version must be one of {SUPPORTED_SCHEMA_VERSIONS}')
    template = spec.get('template')
    if template not in SUPPORTED_TEMPLATES:
        errors.append(f'unsupported template {template!r}; supported={sorted(SUPPORTED_TEMPLATES)}')
    for key in ('data', 'mechanism', 'knowledge', 'supervision'):
        if key not in spec:
            errors.append(f'missing top-level section: {key}')

    if spec.get('modeling_contract') is not None:
        try:
            from .model_family import ModelingContract
            mc = ModelingContract.from_mapping(spec['modeling_contract'])
            diag['modeling_contract'] = mc.to_dict()
            actions.append('run: m2n model-plan --spec <device_card.json> [--data <data.csv>]')
            if mc.input_representation.value == 'MULTIVARIATE_SEQUENCE':
                actions.append('run: m2n temporal-design --spec <device_card.json> --data <sequence.csv> --out <output_dir>')
        except Exception as e:
            errors.append(f'invalid modeling_contract: {e}')

    qcfg = spec.get('qualification', {})
    if not isinstance(qcfg, dict):
        errors.append('qualification must be a JSON object when provided')
        qcfg = {}
    context_col = qcfg.get('independent_context_column')
    evidence_scope=str(qcfg.get('evidence_scope','PRODUCTION')).upper()
    if evidence_scope not in ('PRODUCTION','BLINDED_FOR_VALIDATION'):
        errors.append("qualification.evidence_scope must be 'PRODUCTION' or 'BLINDED_FOR_VALIDATION'")
    if evidence_scope=='BLINDED_FOR_VALIDATION' and not context_col:
        errors.append('BLINDED_FOR_VALIDATION requires qualification.independent_context_column')
    if context_col:
        diag['independent_context_qualification'] = {
            'column': str(context_col),
            'evidence_scope': evidence_scope,
            'min_train_independent_contexts': qcfg.get('min_train_independent_contexts'),
            'min_validation_independent_contexts': qcfg.get('min_validation_independent_contexts'),
        }
        missing_limits = [
            k for k in ('min_train_independent_contexts', 'min_validation_independent_contexts')
            if qcfg.get(k) is None
        ]
        if len(missing_limits)==1:
            errors.append('min_train_independent_contexts and min_validation_independent_contexts must be supplied together')
        elif len(missing_limits)==2 and evidence_scope=='PRODUCTION':
            warnings.append(
                'production independent-context qualification has no evidence-justified context minimums; '
                'architecture qualification will fail closed to AMBIGUOUS'
            )
        elif len(missing_limits)==2 and evidence_scope=='BLINDED_FOR_VALIDATION':
            warnings.append(
                'BLINDED_FOR_VALIDATION permits controlled architecture falsification without claiming production sample adequacy'
            )
        for key in ('min_train_independent_contexts', 'min_validation_independent_contexts'):
            if qcfg.get(key) is None:
                continue
            value=qcfg.get(key)
            if isinstance(value,bool) or not isinstance(value,int) or value < 1:
                errors.append(f'{key} must be a positive integer')
        if not isinstance(spec.get('data'), dict) or not spec.get('data', {}).get('split_column'):
            warnings.append(
                'independent contexts are declared without an explicit split_column; row-wise random splitting may place one context in both train and validation'
            )

    if template == 'generic_mechanism_graph' and isinstance(spec.get('mechanism'), dict):
        mech = spec['mechanism']
        inputs = mech.get('inputs') or []
        nodes = mech.get('nodes') or []
        outputs = mech.get('outputs') or []
        if not inputs:
            errors.append('generic mechanism graph requires mechanism.inputs')
        if not nodes:
            errors.append('generic mechanism graph requires mechanism.nodes')
        if not outputs:
            errors.append('generic mechanism graph requires mechanism.outputs')
        ids = [str(x.get('name')) for x in inputs if isinstance(x, dict)] + [str(x.get('id')) for x in nodes if isinstance(x, dict)]
        dup = sorted({x for x in ids if ids.count(x) > 1})
        if dup:
            errors.append(f'duplicate graph input/node ids: {dup}')
        known = set(ids)
        for n in nodes:
            if not isinstance(n, dict):
                errors.append('every mechanism.nodes entry must be an object')
                continue
            missing_up = [str(x) for x in n.get('inputs', []) if str(x) not in known]
            if missing_up:
                errors.append(f"node {n.get('id')!r} references unknown upstream ids: {missing_up}")
        slot_ids = [str(x.get('id')) for x in (spec.get('knowledge', {}).get('slots') or []) if isinstance(x, dict)]
        bindings = spec.get('bindings') or {}
        unknown_bindings = sorted(set(map(str, bindings.keys())) - set(slot_ids))
        if unknown_bindings:
            errors.append(f'bindings reference undeclared knowledge slots: {unknown_bindings}')
        for sid in slot_ids:
            slot = next(x for x in spec.get('knowledge', {}).get('slots', []) if str(x.get('id')) == sid)
            if slot.get('level') == 'K2':
                fam = tuple(map(str, slot.get('finite_candidates') or []))
                if len(fam) < 2 and not slot.get('allow_k3_escape', False):
                    warnings.append(f'K2 slot {sid!r} has fewer than two finite candidates')
                b = bindings.get(sid, {})
                missing = [x for x in fam if x not in b]
                if missing:
                    errors.append(f'K2 slot {sid!r} lacks binding patches for candidates: {missing}')

    if template == 'second_order_mechanics':
        sup = spec.get('supervision', {})
        if not sup.get('target_acceleration_column') and not sup.get('derive_acceleration'):
            errors.append('second_order_mechanics needs target_acceleration_column or derive_acceleration')
        k = spec.get('knowledge', {}).get('resistance', {})
        if k and k.get('level') == 'K2' and len(k.get('finite_candidates') or []) < 2:
            warnings.append('resistance is K2 but finite_candidates has fewer than two entries')

    if template == 'separable_peak_response':
        m = spec.get('mechanism', {})
        axes = m.get('axes') or []
        fam = m.get('families') or []
        if not axes:
            errors.append('separable_peak_response requires mechanism.axes')
        if len(fam) < 2:
            warnings.append('separable_peak_response has fewer than two candidate families')

    if not errors:
        req = _required_columns(spec)
        diag['declared_required_columns'] = list(req)
        if not spec.get('goal', {}).get('name'):
            warnings.append('goal.name is missing; adding a concise engineering goal improves report readability')
        actions.append('run: m2n inspect-data --spec <device_card.json> --data <normal.csv>')
        if warnings:
            actions.append('review warnings before running architecture competition')
    else:
        actions.append('fix device-card errors before running m2n design')
    status = 'FAIL' if errors else ('WARN' if warnings else 'PASS')
    return PractitionerCheckResult(status, tuple(errors), tuple(warnings), tuple(actions), diag)


def _infer_time_columns(spec: Mapping[str, Any] | None, data: Mapping[str, np.ndarray]) -> tuple[str, ...]:
    cols: list[str] = []
    if spec:
        dcfg = spec.get('data', {})
        if dcfg.get('time_column'):
            cols.append(str(dcfg['time_column']))
        sup = spec.get('supervision', {})
        der = sup.get('derive_acceleration') or {}
        if der.get('time_column'):
            cols.append(str(der['time_column']))
        for req in spec.get('backends', {}).get('requests', []) or []:
            if isinstance(req, dict) and req.get('time_column'):
                cols.append(str(req['time_column']))
    for c in ('time_s', 't_s', 'time', 'timestamp'):
        if c in data:
            cols.append(c)
    return tuple(dict.fromkeys(cols))


def inspect_tabular_data(data_path: str | Path, spec_path: str | Path | None = None) -> PractitionerCheckResult:
    p = Path(data_path)
    errors: list[str] = []
    warnings: list[str] = []
    actions: list[str] = []
    diag: dict[str, Any] = {'data_path': str(p)}
    if not p.exists():
        return PractitionerCheckResult('FAIL', (f'data file not found: {p}',), (), ('check --data path',), diag)
    if p.suffix.lower() != '.csv':
        return PractitionerCheckResult('FAIL', ('M2N practitioner preflight currently supports CSV input',), (), ('convert input to a headered CSV used by m2n design',), diag)
    try:
        data = _read_csv(p)
    except Exception as e:
        return PractitionerCheckResult('FAIL', (f'cannot read CSV: {e}',), (), ('repair CSV header/encoding/row shape',), diag)
    n = len(next(iter(data.values()))) if data else 0
    diag['rows'] = n
    diag['columns'] = list(data.keys())
    if n < 3:
        errors.append(f'too few data rows: {n}')
    spec = None
    if spec_path:
        vr = validate_device_card(spec_path)
        if vr.status == 'FAIL':
            errors.extend([f'device card: {x}' for x in vr.errors])
        else:
            spec = json.loads(Path(spec_path).read_text(encoding='utf-8'))
            missing = [c for c in _required_columns(spec) if c not in data]
            if missing:
                errors.append(f'missing columns required by device card: {missing}')
                actions.append('fix supervision/input mappings or add the missing sensor/data columns')

    modeling_numeric_columns: set[str] = set()
    if spec and isinstance(spec.get('modeling_contract'), Mapping):
        mc = spec['modeling_contract']
        modeling_numeric_columns.update(map(str, mc.get('input_columns') or []))
        modeling_numeric_columns.update(map(str, mc.get('target_columns') or []))
        if mc.get('time_column'):
            modeling_numeric_columns.add(str(mc['time_column']))

    col_diag: dict[str, Any] = {}
    for c, a0 in data.items():
        a = np.asarray(a0)
        cd: dict[str, Any] = {'dtype_kind': a.dtype.kind}
        if a.dtype.kind in 'fiu':
            af = np.asarray(a, float)
            finite = np.isfinite(af)
            cd['nonfinite_count'] = int(np.sum(~finite))
            if np.any(finite):
                cd['min'] = float(np.nanmin(af[finite])); cd['max'] = float(np.nanmax(af[finite])); cd['std'] = float(np.nanstd(af[finite]))
                if cd['std'] <= 1e-12:
                    warnings.append(f'numeric column {c!r} is effectively constant')
            if not np.all(finite):
                errors.append(f'numeric column {c!r} contains {int(np.sum(~finite))} non-finite values')
        else:
            vals, counts = np.unique(a.astype(str), return_counts=True)
            cd['unique_count'] = int(len(vals)); cd['top_values'] = {str(v): int(k) for v, k in zip(vals[:10], counts[:10])}
            if c in modeling_numeric_columns:
                try:
                    af = np.asarray(a, dtype=float)
                    finite = np.isfinite(af)
                    cd['modeling_numeric'] = True
                    cd['nonfinite_count'] = int(np.sum(~finite))
                    if np.any(finite):
                        cd['min'] = float(np.nanmin(af[finite])); cd['max'] = float(np.nanmax(af[finite])); cd['std'] = float(np.nanstd(af[finite]))
                    if not np.all(finite):
                        errors.append(
                            f'modeling-contract numeric column {c!r} contains {int(np.sum(~finite))} non-finite values; '
                            'declare and apply an evidence-justified preprocessing/missing-data strategy before model design'
                        )
                except Exception:
                    cd['modeling_numeric'] = True
                    blank_count = int(sum(not str(x).strip() for x in np.asarray(a, dtype=object)))
                    cd['blank_or_nonnumeric_count'] = blank_count if blank_count else None
                    errors.append(
                        f'modeling-contract numeric column {c!r} contains blank or non-numeric values; '
                        'declare and apply an evidence-justified preprocessing/missing-data strategy before model design'
                    )
        col_diag[c] = cd
    diag['column_diagnostics'] = col_diag

    for tc in _infer_time_columns(spec, data):
        if tc not in data or np.asarray(data[tc]).dtype.kind not in 'fiu':
            continue
        t = np.asarray(data[tc], float)
        context_col = None if not spec else spec.get('qualification', {}).get('independent_context_column')
        if context_col and str(context_col) in data:
            context=np.asarray(data[str(context_col)])
            per_context=[]; all_pos=[]; bad_total=0
            for cid in np.unique(context):
                ii=np.flatnonzero(context==cid); tt=t[ii]; dt=np.diff(tt)
                bad=int(np.sum(dt<=0)); bad_total+=bad
                pos=dt[dt>0]
                if len(pos): all_pos.extend(pos.tolist())
                per_context.append({'context':_as_jsonable(cid),'rows':int(len(ii)),'duplicate_or_reverse_steps':bad})
            td={'context_aware':True,'context_column':str(context_col),'contexts_checked':int(len(per_context)),
                'strictly_increasing_within_context':bool(bad_total==0),'duplicate_or_reverse_steps_within_context':int(bad_total),
                'per_context':per_context[:20]}
            if all_pos:
                pos=np.asarray(all_pos,float); td['median_dt']=float(np.median(pos)); td['dt_cv']=float(np.std(pos)/max(np.mean(pos),1e-15))
                if td['dt_cv']>0.05:
                    warnings.append(f'time column {tc!r} is irregular within independent contexts (dt CV={td["dt_cv"]:.3g}); verify the estimator supports this')
            if bad_total:
                errors.append(f'time column {tc!r} is not strictly increasing within independent contexts')
                actions.append('sort/deduplicate time within each independent context before deriving rates or windowing evidence')
        else:
            dt = np.diff(t)
            td = {'context_aware':False,'strictly_increasing': bool(np.all(dt > 0)), 'duplicate_or_reverse_steps': int(np.sum(dt <= 0))}
            if len(dt) and np.all(np.isfinite(dt)):
                pos = dt[dt > 0]
                if len(pos):
                    med = float(np.median(pos)); td['median_dt'] = med
                    td['dt_cv'] = float(np.std(pos) / max(np.mean(pos), 1e-15))
                    if td['dt_cv'] > 0.05:
                        warnings.append(f'time column {tc!r} is irregularly sampled (dt CV={td["dt_cv"]:.3g}); verify the estimator supports this')
            if td['duplicate_or_reverse_steps']:
                errors.append(f'time column {tc!r} is not strictly increasing')
                actions.append('sort/deduplicate time and preserve the corresponding sensor rows before deriving rates')
        diag.setdefault('time_diagnostics', {})[tc] = td

    if spec and not errors:
        try:
            tri, vai = _split_indices(data, spec.get('data', {}))
            diag['split'] = {'train_rows': int(len(tri)), 'validation_rows': int(len(vai))}
            qcfg = spec.get('qualification', {})
            if len(tri) < int(qcfg.get('min_train_rows', 50)):
                warnings.append(f'train split has only {len(tri)} rows; declared minimum is {int(qcfg.get("min_train_rows", 50))}')
            if len(vai) < int(qcfg.get('min_validation_rows', 20)):
                warnings.append(f'validation split has only {len(vai)} rows; declared minimum is {int(qcfg.get("min_validation_rows", 20))}')
            context_col=qcfg.get('independent_context_column')
            if context_col:
                context_col=str(context_col)
                if context_col not in data:
                    errors.append(f'missing declared independent context column: {context_col!r}')
                else:
                    context=np.asarray(data[context_col])
                    trc=np.unique(context[tri]); vac=np.unique(context[vai]); overlap=np.intersect1d(trc,vac)
                    diag['split'].update({
                        'independent_context_column':context_col,
                        'train_independent_contexts':int(len(trc)),
                        'validation_independent_contexts':int(len(vac)),
                        'independent_context_overlap':int(len(overlap)),
                    })
                    if len(overlap):
                        errors.append(
                            f'independent context column {context_col!r} has {len(overlap)} contexts present in both train and validation'
                        )
                        actions.append('split by whole independent contexts before architecture qualification')
                    scope=str(qcfg.get('evidence_scope','PRODUCTION')).upper()
                    diag['split']['independent_context_evidence_scope']=scope
                    mt=qcfg.get('min_train_independent_contexts'); mv=qcfg.get('min_validation_independent_contexts')
                    if (mt is None) != (mv is None):
                        errors.append('independent-context minimums must be supplied together or both omitted')
                    elif mt is None and scope=='PRODUCTION':
                        warnings.append(
                            'production independent-context qualification has no evidence-justified context minimums; '
                            'm2n design will fail closed to AMBIGUOUS'
                        )
                        actions.append('declare evidence-justified train/validation independent-context minimums before production architecture competition')
                    elif mt is None and scope=='BLINDED_FOR_VALIDATION':
                        warnings.append('independent-context evidence is BLINDED_FOR_VALIDATION only; production sample adequacy remains unqualified')
                        if len(trc)<2 or len(vac)<2:
                            warnings.append('BLINDED_FOR_VALIDATION has fewer than two train or validation independent contexts')
                    else:
                        mt=int(mt); mv=int(mv)
                        if len(trc)<mt or len(vac)<mv:
                            warnings.append(
                                f'independent context support is below the declared minimum: '
                                f'train={len(trc)} (<{mt}), validation={len(vac)} (<{mv})'
                            )
                            actions.append('collect more independent normal contexts or revise the task; do not satisfy this gate by duplicating within-context rows')
                    mc=spec.get('modeling_contract') or {}
                    constraints=(mc.get('constraints') or {}) if isinstance(mc, Mapping) else {}
                    if constraints.get('temporal_context_support'):
                        from .model_family import ModelingContract
                        from .temporal_design import _context_support_analysis
                        contract=ModelingContract.from_mapping(mc)
                        cs,_=_context_support_analysis(spec,contract,data)
                        diag['split']['temporal_context_support']=cs
                        if cs.get('enabled') and int(cs.get('validation_out_of_support_contexts',0))>0:
                            warnings.append(
                                f"{int(cs['validation_out_of_support_contexts'])} validation independent contexts are outside the empirical train temporal-context envelope; "
                                'temporal-design can reserve them for extrapolation audit instead of architecture ranking'
                            )
                            actions.append('review temporal context support and preserve out-of-support contexts as extrapolation evidence rather than silently mixing them into architecture selection')
            if spec.get('template') == 'second_order_mechanics':
                vcol = spec.get('supervision', {}).get('velocity_column')
                if vcol in data:
                    v = np.asarray(data[vcol], float)[tri]
                    pos = int(np.sum(v > 1e-9)); neg = int(np.sum(v < -1e-9))
                    diag['velocity_excitation'] = {'positive_rows': pos, 'negative_rows': neg, 'near_zero_rows': int(len(v) - pos - neg)}
                    candidates = set(map(str, spec.get('knowledge', {}).get('resistance', {}).get('finite_candidates', [])))
                    if candidates.intersection({'viscous_coulomb', 'stribeck'}) and (pos == 0 or neg == 0):
                        warnings.append('friction candidates include direction-sensitive laws but training velocity does not cover both signs')
                        actions.append('collect normal commissioning data with both positive and negative velocity before friction-family adjudication')
        except Exception as e:
            errors.append(f'cannot resolve declared train/validation split: {e}')

    if any('modeling-contract numeric column' in e for e in errors):
        actions.append(
            'repair/qualify missing or non-numeric modeling inputs before design; do not silently zero-fill invalid telemetry packets'
        )
    if not errors and not actions:
        is_temporal=bool(spec and isinstance(spec.get('modeling_contract'),dict) and str(spec['modeling_contract'].get('input_representation','')).upper()=='MULTIVARIATE_SEQUENCE')
        actions.append('data preflight is acceptable; run m2n temporal-design' if is_temporal else 'data preflight is acceptable; run m2n design')
    elif warnings and not errors:
        actions.append('review warnings; warnings are not automatically a PASS for physical identifiability')
    status = 'FAIL' if errors else ('WARN' if warnings else 'PASS')
    return PractitionerCheckResult(status, tuple(dict.fromkeys(errors)), tuple(dict.fromkeys(warnings)), tuple(dict.fromkeys(actions)), diag)


def suggested_next_actions(design_dir: str | Path) -> tuple[str, ...]:
    root = Path(design_dir)
    result_path = root / 'design_result.json'
    qual_path = root / 'qualification.json'
    if not result_path.exists():
        return ('design_result.json is missing; rerun m2n design',)
    result = json.loads(result_path.read_text(encoding='utf-8'))
    qual = json.loads(qual_path.read_text(encoding='utf-8')) if qual_path.exists() else {}
    status = str(result.get('status', 'UNKNOWN'))
    reasons = list(result.get('reasons') or []) + list(qual.get('qualification_reasons') or [])
    out: list[str] = []
    if status in ('SELECTED', 'FIXED', 'K3_SELECTED'):
        out += [
            'run held-out prediction with m2n predict-design before interpreting PHM evidence',
            'use m2n evidence only after checking normal support and target definition',
        ]
    elif status == 'AMBIGUOUS':
        out.append('do not force a single architecture; use m2n predict-design to deploy the retained finite set when models are present')
    elif status == 'UNSUPPORTED':
        out.append('do not infer physical structure from the current observation contract; add/move sensors or change the declared task')
    elif status == 'MODEL_SET_BACKEND_RECOMMENDED':
        out.append('the exact finite cross-product is large; review model_set_routing.json before enabling a model-set inference backend')
    else:
        out.append('inspect design_result.json and qualification.json before retrying')

    text = ' '.join(map(str, reasons)).lower()
    if 'missing required observed columns' in text or 'missing column' in text:
        out.append('repair column mappings or acquire the missing measured variable; do not substitute simulator truth')
    if 'insufficient rows' in text:
        out.append('collect more independent normal commissioning samples rather than lowering the row gate only to obtain a decision')
    if 'rank deficient' in text or 'ill-conditioned' in text:
        out.append('change commissioning excitation so candidate mechanisms are independently excited')
    if 'support exceeds' in text or 'commissioning box' in text:
        out.append('expand normal commissioning coverage or evaluate the out-of-support regime separately')
    if 'target' in text and 'variation' in text:
        out.append('collect a normal trajectory where the supervised physical response actually varies')
    return tuple(dict.fromkeys(out))


def write_next_actions(design_dir: str | Path) -> Path:
    root = Path(design_dir)
    actions = suggested_next_actions(root)
    p = root / 'NEXT_ACTIONS.md'
    lines = ['# M2N 下一步建议', '', '这些建议由当前设计状态和资格原因生成；它们不是自动放宽门槛的指令。', '']
    lines.extend(f'{i}. {a}' for i, a in enumerate(actions, 1))
    p.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return p


def seal_design_bundle(design_dir: str | Path) -> Path:
    from . import __version__
    root = Path(design_dir)
    if not root.exists():
        raise ValueError(f'design directory not found: {root}')
    manifest_path = root / 'bundle_manifest.json'
    files: dict[str, Any] = {}
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p == manifest_path:
            continue
        rel = p.relative_to(root).as_posix()
        files[rel] = {'sha256': _sha256(p), 'size_bytes': p.stat().st_size}
    status = None
    rp = root / 'design_result.json'
    if rp.exists():
        try:
            status = json.loads(rp.read_text(encoding='utf-8')).get('status')
        except Exception:
            status = None
    payload = {
        'contract': 'M2N-DESIGN-BUNDLE/1.0',
        'toolkit_version': __version__,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'design_status': status,
        'file_count': len(files),
        'model_count': len(list((root / 'models').glob('*.pt'))) if (root / 'models').exists() else 0,
        'files': files,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    return manifest_path


def verify_design_bundle(design_dir: str | Path) -> PractitionerCheckResult:
    root = Path(design_dir)
    p = root / 'bundle_manifest.json'
    if not p.exists():
        return PractitionerCheckResult('FAIL', ('bundle_manifest.json is missing',), (), ('run m2n seal-bundle --design-dir <output>',), {'design_dir': str(root)})
    try:
        m = json.loads(p.read_text(encoding='utf-8'))
    except Exception as e:
        return PractitionerCheckResult('FAIL', (f'invalid bundle_manifest.json: {e}',), (), ('regenerate the bundle manifest',), {'design_dir': str(root)})
    errors: list[str] = []
    for rel, info in (m.get('files') or {}).items():
        fp = root / rel
        if not fp.exists():
            errors.append(f'missing bundle file: {rel}')
            continue
        actual = _sha256(fp)
        if actual != info.get('sha256'):
            errors.append(f'hash mismatch: {rel}')
    extras = []
    recorded = set((m.get('files') or {}).keys())
    for fp in root.rglob('*'):
        if fp.is_file() and fp.name != 'bundle_manifest.json':
            rel = fp.relative_to(root).as_posix()
            if rel not in recorded:
                extras.append(rel)
    warnings = tuple(f'unsealed extra file: {x}' for x in extras)
    actions = ('do not modify a sealed design bundle in place; create a new design output when inputs or models change',) if errors or warnings else ('bundle integrity is valid',)
    return PractitionerCheckResult('FAIL' if errors else ('WARN' if warnings else 'PASS'), tuple(errors), warnings, actions, {'design_dir': str(root), 'recorded_files': len(recorded), 'extra_files': extras})


def predict_design_bundle(design_dir: str | Path, data_path: str | Path, output_path: str | Path) -> Path:
    root = Path(design_dir)
    model_paths = sorted((root / 'models').glob('*.pt'))
    if not model_paths:
        raise ValueError('design contains no deployable model artifacts')
    data = _read_csv(Path(data_path))
    n = len(next(iter(data.values())))
    idx = np.arange(n)
    predictions: list[np.ndarray] = []
    candidate_ids: list[str] = []
    names: tuple[str, ...] | None = None
    identity_spec: Mapping[str, Any] | None = None
    for mp in model_paths:
        payload = torch.load(mp, map_location='cpu', weights_only=True)
        spec, model = _candidate_from_payload(payload)
        if identity_spec is None:
            identity_spec = spec
        missing = [c for c in _inference_required_columns(spec) if c not in data]
        if missing:
            raise ValueError(f'missing inference columns: {missing}')
        with torch.no_grad():
            pred = np.asarray(_predict(spec, model, data, idx).detach().cpu().numpy(), float)
        if pred.ndim == 1:
            pred = pred[:, None]
        predictions.append(pred)
        candidate_ids.append(str(payload['candidate_id']))
        current_names = _target_output_names(spec)
        if names is None:
            names = current_names
        elif tuple(names) != tuple(current_names):
            raise ValueError('portfolio artifacts have incompatible target outputs')
    stack = np.stack(predictions, axis=0)
    mean = np.mean(stack, axis=0); std = np.std(stack, axis=0); minimum = np.min(stack, axis=0); maximum = np.max(stack, axis=0)
    names = tuple(names or ())
    headers: list[str] = []
    arrays: list[np.ndarray] = []
    if identity_spec is not None:
        t,context,time_col=_evidence_coordinates(identity_spec,data)
        if context is not None:
            headers.append('m2n_independent_context'); arrays.append(np.asarray(context))
        if time_col is not None:
            headers.append('time_s'); arrays.append(np.asarray(t,float))
    for ci, cid in enumerate(candidate_ids):
        safe = ''.join(ch if ch.isalnum() else '_' for ch in cid)[:80]
        for j, name in enumerate(names):
            headers.append(f'candidate__{safe}__{name}')
            arrays.append(stack[ci, :, j])
    for label, arr in [('portfolio_mean', mean), ('portfolio_std', std), ('portfolio_min', minimum), ('portfolio_max', maximum)]:
        for j, name in enumerate(names):
            headers.append(f'{label}__{name}')
            arrays.append(arr[:, j])
    out = Path(output_path); out.parent.mkdir(parents=True, exist_ok=True)
    mat = np.column_stack(arrays)
    with out.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f); w.writerow(headers); w.writerows(mat.tolist())
    return out
