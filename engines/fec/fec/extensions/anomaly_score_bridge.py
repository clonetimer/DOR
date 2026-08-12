from __future__ import annotations

"""Context-conditioned calibration bridge for external scalar anomaly scores.

This module is Extension/Control-Plane only. It consumes a scalar anomaly
ranking score produced by an upstream qualified detector (for example M2N),
fits independent trusted-normal empirical references per externally declared
context, and emits p-values plus NORMAL/ABNORMAL/CONTEXT_UNSUPPORTED states.

It does not turn a scalar score into a fault class or root-cause probability.
Validity still requires exchangeability of trusted-normal examples *within*
each declared context. Sparse/unseen contexts fail closed; no implicit pooling
or nearest-group fallback is performed.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence
import csv
import hashlib
import json
import numpy as np

from fec.core.fec_core import empirical_upper_p


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    with p.open('r', encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'empty anomaly score CSV: {p}')
    if 'anomaly_score' not in rows[0]:
        raise ValueError('anomaly score pack requires anomaly_score column')
    return rows


def _parse_context_columns(value: Sequence[str] | str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        cols = tuple(x.strip() for x in value.split(',') if x.strip())
    else:
        cols = tuple(str(x).strip() for x in value if str(x).strip())
    if len(set(cols)) != len(cols):
        raise ValueError('duplicate context columns')
    return cols


def _key(row: dict[str, str], context_columns: Sequence[str]) -> tuple[str, ...]:
    if not context_columns:
        return ('__GLOBAL__',)
    missing = [c for c in context_columns if c not in row]
    if missing:
        raise ValueError(f'missing context columns: {missing}')
    return tuple(str(row[c]) for c in context_columns)


@dataclass(frozen=True)
class AnomalyContextReference:
    contract: str
    context_columns: tuple[str, ...]
    alpha: float
    min_group_size: int
    groups: tuple[dict[str, Any], ...]
    calibration_score_sha256: str
    source_score_contract_sha256: str | None
    provenance: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d['context_columns'] = list(self.context_columns)
        d['groups'] = [dict(x) for x in self.groups]
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> 'AnomalyContextReference':
        if d.get('contract') != 'FEC-ANOMALY-CONTEXT-REFERENCE/1.0':
            raise ValueError('unsupported anomaly context reference contract')
        return AnomalyContextReference(
            contract=d['contract'],
            context_columns=tuple(map(str, d.get('context_columns') or ())),
            alpha=float(d['alpha']),
            min_group_size=int(d['min_group_size']),
            groups=tuple(dict(x) for x in d['groups']),
            calibration_score_sha256=str(d['calibration_score_sha256']),
            source_score_contract_sha256=d.get('source_score_contract_sha256'),
            provenance=str(d.get('provenance', 'unspecified')),
        )


def fit_anomaly_score_reference(
    calibration_csv: str | Path,
    output_reference: str | Path,
    *,
    context_columns: Sequence[str] | str | None = None,
    alpha: float = 0.10,
    min_group_size: int = 20,
    score_contract: str | Path | None = None,
    provenance: str = 'external-selected-detector',
) -> Path:
    if not 0 < alpha < 1:
        raise ValueError('alpha must be in (0,1)')
    if min_group_size < 2:
        raise ValueError('min_group_size must be >=2')
    rows = _read_csv(calibration_csv)
    ctx = _parse_context_columns(context_columns)
    scores = np.asarray([float(r['anomaly_score']) for r in rows], dtype=float)
    if np.any(~np.isfinite(scores)):
        raise ValueError('anomaly scores must be finite')
    if 'true_label' in rows[0]:
        labels = np.asarray([str(r['true_label']).strip() for r in rows], dtype=object)
        bad = [x for x in labels if x not in {'0', '0.0', 'NORMAL', 'normal', 'False', 'false'}]
        if bad:
            raise ValueError('anomaly calibration reference must be trusted-normal only; non-normal true_label found')

    grouped: dict[tuple[str, ...], list[float]] = {}
    for row, score in zip(rows, scores):
        grouped.setdefault(_key(row, ctx), []).append(float(score))
    groups = []
    for key in sorted(grouped):
        arr = np.sort(np.asarray(grouped[key], dtype=float))
        if len(arr) < min_group_size:
            # Count is intentionally persisted as unsupported so operators can
            # see why a context was not admitted.
            groups.append({'key': list(key), 'n_reference': int(len(arr)), 'supported': False, 'scores': []})
            continue
        groups.append({
            'key': list(key), 'n_reference': int(len(arr)), 'supported': True,
            'scores': [float(x) for x in arr],
            'min_attainable_p': float(1.0 / (len(arr) + 1.0)),
        })
    if not any(bool(g['supported']) for g in groups):
        raise ValueError('no context group has enough trusted-normal calibration samples; FEC refuses empty reference')
    sc_hash = None if score_contract is None else _sha256(Path(score_contract))
    ref = AnomalyContextReference(
        'FEC-ANOMALY-CONTEXT-REFERENCE/1.0', ctx, float(alpha), int(min_group_size), tuple(groups),
        _sha256(Path(calibration_csv)), sc_hash, str(provenance),
    )
    out = Path(output_reference); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ref.to_dict(), indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return out


def infer_anomaly_decisions(
    score_csv: str | Path,
    reference_json: str | Path,
    output_csv: str | Path,
) -> Path:
    rows = _read_csv(score_csv)
    ref = AnomalyContextReference.from_dict(json.loads(Path(reference_json).read_text(encoding='utf-8')))
    refs = {tuple(map(str, g['key'])): g for g in ref.groups}
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        key = _key(row, ref.context_columns)
        g = refs.get(key)
        base = {k: v for k, v in row.items() if k != 'true_label'}
        base['calibration_context'] = json.dumps(list(key), ensure_ascii=False)
        if g is None or not bool(g.get('supported')):
            n = 0 if g is None else int(g.get('n_reference', 0))
            base.update({
                'p_value': '', 'calibration_supported': False,
                'calibration_n_reference': n,
                'calibration_reason': f'context {key!r} lacks supported trusted-normal calibration',
                'decision': 'CONTEXT_UNSUPPORTED',
            })
        else:
            p = float(empirical_upper_p(np.asarray(g['scores'], dtype=float), np.asarray([float(row['anomaly_score'])]))[0])
            base.update({
                'p_value': p, 'calibration_supported': True,
                'calibration_n_reference': int(g['n_reference']),
                'calibration_reason': 'exact context-conditioned trusted-normal reference supported',
                'decision': 'ABNORMAL' if p <= ref.alpha else 'NORMAL',
            })
        if 'true_label' in row:
            base['true_label'] = row['true_label']  # evaluation-only passthrough
        out_rows.append(base)
    out = Path(output_csv); out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(out_rows[0].keys())
    with out.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out_rows)
    return out
