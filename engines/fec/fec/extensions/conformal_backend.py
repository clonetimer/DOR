from __future__ import annotations
from dataclasses import dataclass, field
from typing import Hashable, Mapping, Sequence
import math
import numpy as np

from fec.core.fec_core import ContactSpec, empirical_upper_p, topology_signature
from .conditional_calibration import MondrianRootRegistry, CalibrationGroupUnavailable
from .weighted_conformal import weighted_upper_p


@dataclass(frozen=True)
class ConformalResult:
    p_values: np.ndarray
    supported: np.ndarray
    backend: str
    diagnostics: Mapping[str, object] = field(default_factory=dict)


class EmpiricalRootBackend:
    """Plain empirical upper-tail root calibration with a topology guard."""
    name = 'empirical'

    def __init__(self):
        self._scores = None
        self._sig = None

    def fit(self, reference_scores, specs: Sequence[ContactSpec], *, A=None, G=None):
        r = np.asarray(reference_scores, float)
        if r.ndim != 1 or len(r) < 2:
            raise ValueError('reference_scores must be 1D with at least two values')
        self._scores = r.copy()
        self._sig = topology_signature(specs, A, G)
        return self

    def p_values(self, scores, specs: Sequence[ContactSpec], *, A=None, G=None) -> ConformalResult:
        if self._scores is None:
            raise RuntimeError('backend is not fitted')
        if topology_signature(specs, A, G) != self._sig:
            raise ValueError('root calibration topology mismatch')
        p = empirical_upper_p(self._scores, np.asarray(scores, float))
        return ConformalResult(p, np.ones(len(np.atleast_1d(p)), bool), self.name, {'n_reference': len(self._scores)})


class MondrianRootBackend:
    """Adapter around the existing exact-key Mondrian registry."""
    name = 'mondrian'

    def __init__(self, *, min_group_size: int = 20):
        self.registry = MondrianRootRegistry(min_group_size=min_group_size)

    def fit_group(self, key: Hashable, reference_scores, specs: Sequence[ContactSpec], *, A=None, G=None, **kwargs):
        self.registry.fit_group(key, np.asarray(reference_scores, float), specs, A=A, G=G, **kwargs)
        return self

    def p_values(self, key: Hashable, scores, specs: Sequence[ContactSpec], *, A=None, G=None) -> ConformalResult:
        try:
            p = self.registry.p_value(key, np.asarray(scores, float), specs, A=A, G=G)
        except CalibrationGroupUnavailable:
            n = len(np.atleast_1d(scores))
            return ConformalResult(np.full(n, np.nan), np.zeros(n, bool), self.name, {'group': repr(key), 'reason': 'group_unavailable'})
        return ConformalResult(p, np.ones(len(np.atleast_1d(p)), bool), self.name, {'group': repr(key), 'n_reference': self.registry.info(key).n_reference})


def _standardize_fit(X: np.ndarray):
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2 or len(X) < 2:
        raise ValueError('covariates must be a 2D array with at least two rows')
    mu = X.mean(0)
    sd = X.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return X, mu, sd


def _gaussian_kernel_weights(X: np.ndarray, center: np.ndarray, bandwidth: float) -> np.ndarray:
    d2 = np.sum(((X - center[None, :]) / bandwidth) ** 2, axis=1)
    # The common normalizing constant cancels in the p-value and is omitted.
    return np.exp(-0.5 * d2)


class LocalizedRootBackend:
    """Localized conformal p-values for a continuous, externally declared context.

    Implements the single-test localized conformal p-value in Wu et al. (ICLR
    2025, Eq. 5) for a Gaussian localization kernel. Larger root scores mean
    more anomalous. For each test covariate x, a randomized local center
    x_tilde ~ N(x, h^2 I) is sampled; calibration scores are then compared
    using H(X_i, x_tilde), with an independent U[0,1] test-mass randomizer.

    This class intentionally does *not* implement the paper's multi-test BH/FDR
    pruning algorithm. It is a p-value backend for FEC PolicyEvidenceState.
    The Adapter must provide a defensible external context covariate and a
    trusted-normal reference set under the same conditional mechanism.
    """
    name = 'localized_lcp'

    def __init__(self, *, bandwidth: float | None = None, min_effective_n: float = 20.0, random_state: int = 0):
        self.bandwidth = bandwidth
        self.min_effective_n = float(min_effective_n)
        self.random_state = int(random_state)
        self._scores = self._X = self._mu = self._sd = self._sig = None
        self._h = None

    def fit(self, reference_scores, reference_covariates, specs: Sequence[ContactSpec], *, A=None, G=None):
        r = np.asarray(reference_scores, float)
        X, mu, sd = _standardize_fit(reference_covariates)
        if r.ndim != 1 or len(r) != len(X) or len(r) < 2:
            raise ValueError('reference scores/covariates must have matching rows >=2')
        if np.any(~np.isfinite(r)) or np.any(~np.isfinite(X)):
            raise ValueError('reference data must be finite')
        d = X.shape[1]
        # Paper simulation default h = (n/2)^(-1/(d+2)); after standardization.
        h = float(self.bandwidth if self.bandwidth is not None else (len(X) / 2.0) ** (-1.0 / (d + 2.0)))
        if not np.isfinite(h) or h <= 0:
            raise ValueError('bandwidth must be positive')
        self._scores = r.copy(); self._X = (X - mu) / sd; self._mu = mu; self._sd = sd
        self._sig = topology_signature(specs, A, G); self._h = h
        return self

    def p_values(self, scores, covariates, specs: Sequence[ContactSpec], *, A=None, G=None, random_state: int | None = None) -> ConformalResult:
        if self._scores is None:
            raise RuntimeError('backend is not fitted')
        if topology_signature(specs, A, G) != self._sig:
            raise ValueError('root calibration topology mismatch')
        s = np.atleast_1d(np.asarray(scores, float))
        Xq = np.asarray(covariates, float)
        if Xq.ndim == 1:
            Xq = Xq[:, None]
        if len(Xq) != len(s) or Xq.shape[1] != self._X.shape[1]:
            raise ValueError('test covariates must match scores and reference covariate dimension')
        Zq = (Xq - self._mu) / self._sd
        rng = np.random.default_rng(self.random_state if random_state is None else random_state)
        p = np.empty(len(s), float); ess = np.empty(len(s), float); supported = np.ones(len(s), bool)
        for j, (v, x) in enumerate(zip(s, Zq)):
            x_tilde = x + rng.normal(0.0, self._h, size=len(x))
            w = _gaussian_kernel_weights(self._X, x_tilde, self._h)
            w_test = float(_gaussian_kernel_weights(x[None, :], x_tilde, self._h)[0])
            sw = float(np.sum(w))
            ess[j] = (sw * sw) / max(float(np.sum(w * w)), 1e-15)
            xi = float(rng.random())
            numerator = float(np.sum(w[self._scores >= v])) + xi * w_test
            denominator = sw + w_test
            p[j] = numerator / max(denominator, 1e-15)
            supported[j] = bool(ess[j] >= self.min_effective_n)
        return ConformalResult(
            p,
            supported,
            self.name,
            {'bandwidth': self._h, 'effective_n': ess, 'min_effective_n': self.min_effective_n, 'randomized': True},
        )


class WeightedRootBackend:
    """Importance-weighted root p-values under a declared covariate-shift contract.

    The numerical p-value is the conservative weighted upper-tail form already
    used in v1.3. It remains an *assumption-gated* backend: readiness requires a
    validated source->target density ratio and a stable conditional mechanism.
    """
    name = 'weighted_covariate_shift'

    def __init__(self, *, min_effective_n: float = 20.0):
        self.min_effective_n = float(min_effective_n)
        self._scores = self._weights = self._sig = None

    def fit(self, reference_scores, reference_weights, specs: Sequence[ContactSpec], *, A=None, G=None):
        r = np.asarray(reference_scores, float); w = np.asarray(reference_weights, float)
        if r.ndim != 1 or w.ndim != 1 or len(r) != len(w) or len(r) < 2:
            raise ValueError('reference scores/weights must be equal-length 1D arrays')
        if np.any(w < 0) or not np.all(np.isfinite(w)) or np.sum(w) <= 0:
            raise ValueError('weights must be finite, nonnegative and have positive mass')
        self._scores = r.copy(); self._weights = w.copy(); self._sig = topology_signature(specs, A, G)
        return self

    def p_values(self, scores, specs: Sequence[ContactSpec], *, test_weight=1.0, A=None, G=None) -> ConformalResult:
        if self._scores is None:
            raise RuntimeError('backend is not fitted')
        if topology_signature(specs, A, G) != self._sig:
            raise ValueError('root calibration topology mismatch')
        p = weighted_upper_p(self._scores, self._weights, scores, test_weight=test_weight)
        ess = float(np.sum(self._weights) ** 2 / max(np.sum(self._weights ** 2), 1e-15))
        supported = np.full(len(np.atleast_1d(p)), ess >= self.min_effective_n, bool)
        return ConformalResult(p, supported, self.name, {'effective_n': ess, 'min_effective_n': self.min_effective_n})
