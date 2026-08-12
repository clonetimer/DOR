from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CleaningPlan:
    candidate_indices: np.ndarray
    candidate_scores: np.ndarray
    budget: int


@dataclass(frozen=True)
class CleaningResult:
    cleaned_scores: np.ndarray
    removed_indices: np.ndarray
    reviewed_indices: np.ndarray
    reviewed_outliers: int
    reviewed_inliers: int


class ActiveReferenceCleaner:
    """Limited-budget label-and-trim cleaning for a contaminated reference set.

    FEC adaptation of the active data-cleaning idea in Bashari, Sesia & Romano
    (ICML 2025): review the most suspicious calibration examples and remove only
    samples explicitly confirmed to be outliers. Suspicious but confirmed-normal
    points remain in the reference, avoiding naive unlabelled top-score trimming.

    Labels are intentionally external; this class never infers 'outlier' from the
    same score that selects candidates.
    """
    def __init__(self, *, budget: int):
        if int(budget) <= 0:
            raise ValueError('budget must be positive')
        self.budget = int(budget)

    def propose(self, reference_scores) -> CleaningPlan:
        s = np.asarray(reference_scores, float)
        if s.ndim != 1 or len(s) < 2 or np.any(~np.isfinite(s)):
            raise ValueError('reference_scores must be finite 1D with >=2 rows')
        b = min(self.budget, len(s))
        idx = np.argsort(s)[-b:][::-1]
        return CleaningPlan(idx.copy(), s[idx].copy(), b)

    def apply(self, reference_scores, plan: CleaningPlan, *, confirmed_outlier) -> CleaningResult:
        s = np.asarray(reference_scores, float)
        labels = np.asarray(confirmed_outlier, bool)
        if labels.shape != (len(plan.candidate_indices),):
            raise ValueError('confirmed_outlier must provide one external label per reviewed candidate')
        remove = plan.candidate_indices[labels]
        keep = np.ones(len(s), bool); keep[remove] = False
        return CleaningResult(
            s[keep].copy(), remove.copy(), plan.candidate_indices.copy(),
            int(labels.sum()), int((~labels).sum()),
        )


def naive_unlabelled_trim(reference_scores, *, trim_fraction: float):
    """Unsafe matched baseline: remove the largest scores without labels."""
    s = np.asarray(reference_scores, float)
    if not (0 <= trim_fraction < 1):
        raise ValueError('trim_fraction must be in [0,1)')
    n = int(np.floor(len(s) * trim_fraction))
    if n == 0:
        return s.copy()
    keep = np.ones(len(s), bool); keep[np.argsort(s)[-n:]] = False
    return s[keep]
