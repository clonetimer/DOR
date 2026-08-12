from __future__ import annotations

"""Experimental class-evidence fusion/calibration under missing modalities.

This is a deliberately optional path for *genuine class evidence*.  It does not
reinterpret scalar FEC anomaly Evidence E as a Dirichlet opinion.

The implementation assimilates two useful mechanisms from missing-modality
conformal fusion work:
  1. an absent class-evidence source is treated as vacuous/neutral evidence;
  2. conformal calibration is keyed by the externally observed modality mask.

It intentionally does not implement a trainable multimodal backbone or claim
instance-conditional validity.  Finite-sample validity still requires
exchangeability within each calibrated mask stratum.
"""

from dataclasses import dataclass
from math import ceil
from typing import Iterable, Sequence

import numpy as np

from .class_evidence import ClassEvidenceBatch, validate_shared_frame
from .opinion_fusion import DirichletOpinion, dempster_shafer_pair


class MaskCalibrationUnavailable(KeyError):
    pass


def _mask_tuple(row: Sequence[int | bool]) -> tuple[int, ...]:
    out = tuple(int(bool(x)) for x in row)
    if not out or not any(out):
        raise ValueError("modality mask must be non-empty and contain at least one present modality")
    return out


def _check_masks(masks, n_samples: int, n_modalities: int) -> tuple[tuple[int, ...], ...]:
    a = np.asarray(masks)
    if a.ndim != 2 or a.shape != (n_samples, n_modalities):
        raise ValueError(f"masks must have shape {(n_samples, n_modalities)}")
    return tuple(_mask_tuple(row) for row in a)


@dataclass(frozen=True)
class MaskedFusionBatch:
    probabilities: np.ndarray
    combined_vacuity: np.ndarray
    per_modality_vacuity: np.ndarray
    masks: tuple[tuple[int, ...], ...]
    class_frame: tuple[str, ...]

    def __post_init__(self) -> None:
        p = np.asarray(self.probabilities, dtype=float)
        u = np.asarray(self.combined_vacuity, dtype=float)
        pmu = np.asarray(self.per_modality_vacuity, dtype=float)
        if p.ndim != 2 or len(p) != len(u) or len(p) != len(self.masks):
            raise ValueError("invalid fused probability/vacuity dimensions")
        if pmu.ndim != 2 or pmu.shape[0] != len(p):
            raise ValueError("per_modality_vacuity dimensions do not match samples")
        if p.shape[1] != len(self.class_frame):
            raise ValueError("probabilities do not match class_frame")
        if not np.all(np.isfinite(p)) or np.any(p < 0):
            raise ValueError("probabilities must be finite and non-negative")
        if not np.allclose(p.sum(axis=1), 1.0, atol=1e-10):
            raise ValueError("probability rows must sum to one")
        if not np.all(np.isfinite(u)) or np.any((u < 0) | (u > 1)):
            raise ValueError("combined vacuity must lie in [0,1]")
        if not np.all(np.isfinite(pmu)) or np.any((pmu < 0) | (pmu > 1)):
            raise ValueError("per-modality vacuity must lie in [0,1]")
        object.__setattr__(self, "probabilities", p.copy())
        object.__setattr__(self, "combined_vacuity", u.copy())
        object.__setattr__(self, "per_modality_vacuity", pmu.copy())


def fuse_class_evidence_by_mask(
    batches: Sequence[ClassEvidenceBatch],
    masks,
) -> MaskedFusionBatch:
    """Fuse modality class-evidence while making absent modalities vacuous.

    If a mask marks a modality absent, that row's stored evidence is ignored
    entirely.  This is important when arrays are preallocated or stale values
    remain in an unavailable input stream.
    """
    frame = validate_shared_frame(batches)
    n = len(batches[0].evidence)
    M = len(batches)
    mask_rows = _check_masks(masks, n, M)
    K = len(frame)
    probabilities = np.empty((n, K), dtype=float)
    combined_u = np.empty(n, dtype=float)
    per_u = np.ones((n, M), dtype=float)

    for i, mask in enumerate(mask_rows):
        fused: DirichletOpinion | None = None
        for m, present in enumerate(mask):
            if not present:
                # Vacuous opinion: evidence == 0, uncertainty == 1.  We do not
                # actually combine it because it is the identity element for
                # Dempster-style opinion combination.
                per_u[i, m] = 1.0
                continue
            opinion = DirichletOpinion(batches[m].evidence[i], source=batches[m].source)
            per_u[i, m] = opinion.uncertainty
            fused = opinion if fused is None else dempster_shafer_pair(fused, opinion)
        if fused is None:  # protected by _mask_tuple, retained as invariant guard
            raise ValueError("all modalities are missing")
        probabilities[i] = fused.probability
        combined_u[i] = fused.uncertainty

    return MaskedFusionBatch(probabilities, combined_u, per_u, mask_rows, frame)


@dataclass(frozen=True)
class MaskCalibrationInfo:
    mask: tuple[int, ...]
    n_reference: int
    threshold: float
    alpha: float


class MaskConditionalClassConformal:
    """Mondrian split-conformal calibration keyed by modality-presence mask.

    The score is r(x,y,s)=1-p_y + lambda_vacuity*u, matching the useful
    evidential-conformal form while keeping the calibrator independent of any
    particular neural backbone.
    """

    maturity = "experimental"

    def __init__(
        self,
        *,
        alpha: float = 0.10,
        lambda_vacuity: float = 0.0,
        min_group_size: int = 20,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0,1)")
        if lambda_vacuity < 0:
            raise ValueError("lambda_vacuity must be >= 0")
        if min_group_size < 2:
            raise ValueError("min_group_size must be >= 2")
        self.alpha = float(alpha)
        self.lambda_vacuity = float(lambda_vacuity)
        self.min_group_size = int(min_group_size)
        self._thresholds: dict[tuple[int, ...], float] = {}
        self._counts: dict[tuple[int, ...], int] = {}
        self._n_classes: int | None = None
        self._n_modalities: int | None = None

    @staticmethod
    def _validate_probabilities(probabilities) -> np.ndarray:
        p = np.asarray(probabilities, dtype=float)
        if p.ndim != 2 or p.shape[1] < 2:
            raise ValueError("probabilities must be n_samples x n_classes with at least two classes")
        if not np.all(np.isfinite(p)) or np.any(p < 0):
            raise ValueError("probabilities must be finite and non-negative")
        if not np.allclose(p.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("probability rows must sum to one")
        return p

    def _vacuity(self, vacuity, n: int) -> np.ndarray:
        if vacuity is None:
            return np.zeros(n, dtype=float)
        u = np.asarray(vacuity, dtype=float)
        if u.ndim != 1 or len(u) != n or not np.all(np.isfinite(u)) or np.any((u < 0) | (u > 1)):
            raise ValueError("vacuity must be a finite vector in [0,1]")
        return u

    def _true_scores(self, probabilities: np.ndarray, labels, vacuity: np.ndarray) -> np.ndarray:
        y = np.asarray(labels, dtype=int)
        if y.ndim != 1 or len(y) != len(probabilities):
            raise ValueError("labels must be a 1D integer vector matching probabilities")
        if np.any((y < 0) | (y >= probabilities.shape[1])):
            raise ValueError("labels contain class indices outside the probability frame")
        return 1.0 - probabilities[np.arange(len(y)), y] + self.lambda_vacuity * vacuity

    def fit(self, probabilities, labels, masks, *, vacuity=None) -> "MaskConditionalClassConformal":
        p = self._validate_probabilities(probabilities)
        mask_arr = np.asarray(masks)
        if mask_arr.ndim != 2 or mask_arr.shape[0] != len(p):
            raise ValueError("masks must be n_samples x n_modalities")
        mask_rows = _check_masks(mask_arr, len(p), mask_arr.shape[1])
        u = self._vacuity(vacuity, len(p))
        scores = self._true_scores(p, labels, u)
        self._thresholds.clear()
        self._counts.clear()
        self._n_classes = p.shape[1]
        self._n_modalities = mask_arr.shape[1]
        for mask in sorted(set(mask_rows)):
            idx = np.asarray([m == mask for m in mask_rows], dtype=bool)
            group = np.sort(scores[idx])
            n = len(group)
            self._counts[mask] = n
            if n < self.min_group_size:
                continue
            # Split-conformal finite-sample quantile: ceil((n+1)(1-alpha))-th
            # order statistic, clipped at the largest observed score.
            rank = min(n, ceil((n + 1) * (1.0 - self.alpha)))
            self._thresholds[mask] = float(group[rank - 1])
        return self

    def has_mask(self, mask: Sequence[int | bool]) -> bool:
        return _mask_tuple(mask) in self._thresholds

    def info(self, mask: Sequence[int | bool]) -> MaskCalibrationInfo:
        key = _mask_tuple(mask)
        if key not in self._thresholds:
            n = self._counts.get(key, 0)
            raise MaskCalibrationUnavailable(
                f"mask {key!r} has no supported calibration group (n={n}, min={self.min_group_size})"
            )
        return MaskCalibrationInfo(key, self._counts[key], self._thresholds[key], self.alpha)

    def prediction_sets(self, probabilities, masks, *, vacuity=None) -> list[tuple[int, ...]]:
        p = self._validate_probabilities(probabilities)
        if self._n_classes is None or self._n_modalities is None:
            raise RuntimeError("calibrator is not fitted")
        if p.shape[1] != self._n_classes:
            raise ValueError("class frame size differs from calibration")
        mask_rows = _check_masks(masks, len(p), self._n_modalities)
        u = self._vacuity(vacuity, len(p))
        out: list[tuple[int, ...]] = []
        for i, mask in enumerate(mask_rows):
            if mask not in self._thresholds:
                n = self._counts.get(mask, 0)
                raise MaskCalibrationUnavailable(
                    f"mask {mask!r} has no supported calibration group (n={n}, min={self.min_group_size}); "
                    "FEC will not silently borrow a more permissive mask"
                )
            scores = 1.0 - p[i] + self.lambda_vacuity * u[i]
            selected = tuple(int(k) for k in np.flatnonzero(scores <= self._thresholds[mask]))
            out.append(selected)
        return out

    def coverage(self, probabilities, labels, masks, *, vacuity=None) -> float:
        sets = self.prediction_sets(probabilities, masks, vacuity=vacuity)
        y = np.asarray(labels, dtype=int)
        if y.ndim != 1 or len(y) != len(sets):
            raise ValueError("labels must match samples")
        return float(np.mean([int(label) in pred for label, pred in zip(y, sets)]))

    @property
    def calibrated_masks(self) -> tuple[tuple[int, ...], ...]:
        return tuple(sorted(self._thresholds))
