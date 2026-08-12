from __future__ import annotations
from dataclasses import dataclass
from typing import Hashable, Mapping
import math
import numpy as np


@dataclass(frozen=True)
class GroupCalibrationAudit:
    key: Hashable
    n: int
    alpha: float
    empirical_fpr: float
    absolute_gap: float
    binomial_se: float
    pass_tolerance: bool


@dataclass(frozen=True)
class CalibrationAuditSummary:
    groups: tuple[GroupCalibrationAudit, ...]
    weighted_fpr: float
    max_gap: float
    fpr_std: float
    all_groups_adequate: bool


def audit_group_pvalues(
    grouped_pvalues: Mapping[Hashable, np.ndarray],
    *,
    alpha: float = 0.05,
    min_group_size: int = 20,
    tolerance: float | None = None,
) -> CalibrationAuditSummary:
    """Audit empirical group-conditional calibration.

    This does not create a theoretical guarantee. It makes the operational
    assumptions visible: each declared stratum must have enough held-out
    trusted-normal samples and its empirical FPR should be inspected by group,
    rather than only after pooling all contexts/presence masks.
    """
    if not (0 < alpha < 1):
        raise ValueError('alpha must be in (0,1)')
    if min_group_size < 2:
        raise ValueError('min_group_size must be >=2')
    audits = []
    total_n = 0
    total_fp = 0
    for key in sorted(grouped_pvalues, key=lambda x: repr(x)):
        p = np.asarray(grouped_pvalues[key], dtype=float)
        p = p[np.isfinite(p)]
        n = len(p)
        if n == 0:
            fpr = float('nan'); gap = float('inf'); se = float('inf')
        else:
            fp = int(np.sum(p <= alpha)); fpr = fp / n
            total_n += n; total_fp += fp
            gap = abs(fpr - alpha)
            se = math.sqrt(max(alpha * (1 - alpha) / n, 0.0))
        tol = tolerance if tolerance is not None else max(0.02, 2.5 * se)
        audits.append(GroupCalibrationAudit(key, n, alpha, fpr, gap, se, bool(n >= min_group_size and gap <= tol)))
    finite = np.array([g.empirical_fpr for g in audits if math.isfinite(g.empirical_fpr)])
    return CalibrationAuditSummary(
        tuple(audits),
        float(total_fp / total_n) if total_n else float('nan'),
        float(max((g.absolute_gap for g in audits), default=float('nan'))),
        float(np.std(finite)) if len(finite) else float('nan'),
        bool(all(g.n >= min_group_size for g in audits)),
    )
