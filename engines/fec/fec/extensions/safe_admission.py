from __future__ import annotations
from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping
import numpy as np

from fec.core.fec_core import (
    RootCalibrationReference,
    empirical_upper_p,
    calibration_admission,
)


@dataclass(frozen=True)
class AdmissionDecision:
    allowed: bool
    dependency_ok: bool
    reserve_p: float
    reason: str
    reserve_key: Hashable | None = None


class FrozenReserveAdmissionGate:
    """Conservative single-reserve normal-reference write gate.

    This is a FEC adaptation inspired by the frozen-reserve principle in
    recent self-poisoning analyses of adaptive OOD detection. It is not a
    reproduction of that paper's certificate.
    """

    def __init__(self, reserve_scores: np.ndarray, *, alpha: float = 0.05):
        r = np.asarray(reserve_scores, dtype=float)
        if r.ndim != 1 or len(r) < 2:
            raise ValueError('reserve_scores must be a 1D array with >=2 values')
        if not (0 < alpha < 1):
            raise ValueError('alpha must be in (0,1)')
        min_p = 1.0 / (len(r) + 1.0)
        if alpha < min_p:
            raise ValueError(
                f'alpha={alpha} is below empirical resolution {min_p:.6g}; '
                'increase frozen reserve size or use a larger alpha'
            )
        self._reserve = r.copy()
        self.alpha = float(alpha)

    @property
    def reserve_size(self) -> int:
        return len(self._reserve)

    def check(self, candidate_score: float, *, trusted_claims: Iterable[str], target_reference: RootCalibrationReference) -> AdmissionDecision:
        dep_ok = calibration_admission(trusted_claims, target_reference)
        p = float(empirical_upper_p(self._reserve, np.asarray([candidate_score]))[0])
        if not dep_ok:
            return AdmissionDecision(False, False, p, 'dependency/provenance contract not satisfied')
        if p <= self.alpha:
            return AdmissionDecision(False, True, p, 'candidate inconsistent with frozen trusted reserve')
        return AdmissionDecision(True, True, p, 'dependency contract and frozen reserve both support admission')


class FrozenReserveRegistry:
    """Context/presence keyed immutable reserve registry.

    Selecting the reserve uses an *external key*. No fallback to the reserve
    that makes a candidate look most normal is permitted.
    """
    def __init__(self, reserves: Mapping[Hashable, np.ndarray], *, alpha: float = 0.05):
        if not reserves:
            raise ValueError('at least one reserve is required')
        self._gates = {k: FrozenReserveAdmissionGate(v, alpha=alpha) for k, v in reserves.items()}
        self.alpha = float(alpha)

    def keys(self):
        return tuple(self._gates.keys())

    def check(self, key: Hashable, candidate_score: float, *, trusted_claims: Iterable[str], target_reference: RootCalibrationReference) -> AdmissionDecision:
        if key not in self._gates:
            return AdmissionDecision(False, False, float('nan'), 'no frozen reserve for externally declared key', key)
        d = self._gates[key].check(candidate_score, trusted_claims=trusted_claims, target_reference=target_reference)
        return AdmissionDecision(d.allowed, d.dependency_ok, d.reserve_p, d.reason, key)
