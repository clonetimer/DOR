from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Iterable, Mapping, Sequence, Any
import copy
import numpy as np

from fec.core.fec_core import ContactSpec, topology_signature
from .conformal_backend import ConformalResult
from .drift_guard import CalibrationHealth, audit_p_values, audit_p_value_series


@dataclass(frozen=True)
class CalibrationReferenceMeta:
    """Metadata/permission contract for one production calibration reference.

    The numerical calibration backend is deliberately separate from this
    metadata. `dependency_set` is the FEC D_cal contract: all named health
    claims must be externally trusted before a candidate is even allowed to
    enter shadow validation.
    """

    reference_id: str
    key: Hashable
    provenance: str
    dependency_set: frozenset[str] = frozenset()
    scope: str = "root"
    notes: str = ""
    reference_data_id: str = ""


@dataclass
class ManagedReference:
    meta: CalibrationReferenceMeta
    backend: Any
    topology_sig: tuple
    lifecycle_state: str = "shadow"  # shadow | active | retired | rejected
    health_status: str = "unknown"
    health: CalibrationHealth | None = None
    validation_supported_fraction: float | None = None
    generation: int = 0


@dataclass(frozen=True)
class ProductionCalibrationResult:
    p_values: np.ndarray
    supported: np.ndarray
    usable_for_policy: np.ndarray
    reference_id: str | None
    reference_key: Hashable | None
    backend: str | None
    state: str
    reason: str
    diagnostics: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateValidation:
    reference_id: str
    key: Hashable
    accepted_for_promotion: bool
    power_limited: bool
    supported_fraction: float
    health: CalibrationHealth | None
    reason: str


class ProductionCalibrationManager:
    """Validity-first calibration lifecycle for FEC under non-exchangeability.

    This manager does *not* promise that one conformal formula remains valid
    under arbitrary distribution shift. Instead it enforces the production
    contract discovered in FEC validation:

      1. select references only by an externally declared key;
      2. require exact FEC G/A/topology compatibility inside each backend;
      3. never use backend p-values when local/weighted support is inadequate;
      4. block anti-conservative references discovered on trusted-normal
         sentinels;
      5. allow normal-reference changes only through D_cal/trusted claims;
      6. validate refreshes in shadow mode on a disjoint trusted sentinel;
      7. promote candidates atomically; never learn normal from arbitrary
         online samples merely because their anomaly score is small.

    `backend` objects need only implement a `p_values(...) -> ConformalResult`
    method. Their fit procedure stays backend-specific and therefore outside
    the Frozen FEC Core.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        min_sentinel_samples: int = 20,
        min_supported_fraction: float = 0.8,
        sentinel_block_size: int = 1,
        sentinel_bootstrap_reps: int = 2000,
        random_state: int = 0,
        require_disjoint_sentinel: bool = True,
    ):
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0,1)")
        if min_sentinel_samples < 2:
            raise ValueError("min_sentinel_samples must be >=2")
        if not (0 < min_supported_fraction <= 1):
            raise ValueError("min_supported_fraction must be in (0,1]")
        self.alpha = float(alpha)
        self.min_sentinel_samples = int(min_sentinel_samples)
        self.min_supported_fraction = float(min_supported_fraction)
        self.sentinel_block_size = int(sentinel_block_size)
        self.sentinel_bootstrap_reps = int(sentinel_bootstrap_reps)
        self.random_state = int(random_state)
        self.require_disjoint_sentinel = bool(require_disjoint_sentinel)
        if self.sentinel_block_size < 1:
            raise ValueError("sentinel_block_size must be >=1")
        self._active: dict[Hashable, ManagedReference] = {}
        self._shadow: dict[Hashable, ManagedReference] = {}
        self._retired: dict[Hashable, list[ManagedReference]] = {}
        self._history: list[dict[str, object]] = []
        self._generation = 0

    @staticmethod
    def _check_trusted_claims(meta: CalibrationReferenceMeta, trusted_claims: Iterable[str]) -> bool:
        claims = frozenset(str(x) for x in trusted_claims)
        return meta.dependency_set.issubset(claims)

    def install_active(
        self,
        meta: CalibrationReferenceMeta,
        backend: Any,
        specs: Sequence[ContactSpec],
        *,
        A=None,
        G=None,
        health_status: str = "unverified",
    ) -> None:
        """Install an already commissioned reference as active.

        This is intended for initial commissioning or loading a previously
        validated reference from persistent storage. Runtime refreshes should
        use `stage_candidate` -> `validate_candidate` -> `promote`.
        """
        self._generation += 1
        ref = ManagedReference(
            meta=meta,
            backend=backend,
            topology_sig=topology_signature(specs, A, G),
            lifecycle_state="active",
            health_status=str(health_status),
            generation=self._generation,
        )
        old = self._active.get(meta.key)
        if old is not None:
            old.lifecycle_state = "retired"
            self._retired.setdefault(meta.key, []).append(old)
        self._active[meta.key] = ref
        self._history.append({"event": "install_active", "key": repr(meta.key), "reference_id": meta.reference_id})

    def stage_candidate(
        self,
        meta: CalibrationReferenceMeta,
        backend: Any,
        specs: Sequence[ContactSpec],
        *,
        trusted_claims: Iterable[str],
        A=None,
        G=None,
    ) -> bool:
        if not self._check_trusted_claims(meta, trusted_claims):
            self._history.append({
                "event": "stage_rejected_dependency",
                "key": repr(meta.key),
                "reference_id": meta.reference_id,
                "missing": sorted(meta.dependency_set.difference(frozenset(str(x) for x in trusted_claims))),
            })
            return False
        self._generation += 1
        self._shadow[meta.key] = ManagedReference(
            meta=meta,
            backend=backend,
            topology_sig=topology_signature(specs, A, G),
            lifecycle_state="shadow",
            generation=self._generation,
        )
        self._history.append({"event": "stage_candidate", "key": repr(meta.key), "reference_id": meta.reference_id})
        return True

    @staticmethod
    def _call_backend(ref: ManagedReference, scores, specs, *, A=None, G=None, backend_kwargs=None) -> ConformalResult:
        kwargs = dict(backend_kwargs or {})
        # Backends deliberately retain their own method signatures. Common FEC
        # topology inputs are appended here; group/covariate/test-weight inputs
        # are supplied through backend_kwargs.
        return ref.backend.p_values(scores, specs=specs, A=A, G=G, **kwargs)

    def evaluate(
        self,
        key: Hashable,
        scores,
        specs: Sequence[ContactSpec],
        *,
        A=None,
        G=None,
        backend_kwargs: Mapping[str, object] | None = None,
    ) -> ProductionCalibrationResult:
        s = np.atleast_1d(np.asarray(scores, float))
        ref = self._active.get(key)
        if ref is None:
            return ProductionCalibrationResult(
                np.full(len(s), np.nan), np.zeros(len(s), bool), np.zeros(len(s), bool),
                None, key, None, "reference_unsupported",
                "no active calibration reference for externally declared key",
                {"key": repr(key)},
            )
        if ref.health_status in {"unknown", "unverified"}:
            return ProductionCalibrationResult(
                np.full(len(s), np.nan), np.zeros(len(s), bool), np.zeros(len(s), bool),
                ref.meta.reference_id, key, getattr(ref.backend, "name", type(ref.backend).__name__),
                "calibration_unverified",
                "active reference has not passed a trusted-normal sentinel audit", {},
            )
        if ref.health_status in {"anti_conservative", "blocked", "rejected"}:
            return ProductionCalibrationResult(
                np.full(len(s), np.nan), np.zeros(len(s), bool), np.zeros(len(s), bool),
                ref.meta.reference_id, key, getattr(ref.backend, "name", type(ref.backend).__name__),
                "calibration_blocked",
                f"active reference health is {ref.health_status}",
                {"health": None if ref.health is None else ref.health.__dict__},
            )
        if topology_signature(specs, A, G) != ref.topology_sig:
            return ProductionCalibrationResult(
                np.full(len(s), np.nan), np.zeros(len(s), bool), np.zeros(len(s), bool),
                ref.meta.reference_id, key, getattr(ref.backend, "name", type(ref.backend).__name__),
                "topology_unsupported", "active reference topology does not match current G/A topology", {},
            )
        try:
            cres = self._call_backend(ref, s, specs, A=A, G=G, backend_kwargs=backend_kwargs)
        except Exception as exc:
            return ProductionCalibrationResult(
                np.full(len(s), np.nan), np.zeros(len(s), bool), np.zeros(len(s), bool),
                ref.meta.reference_id, key, getattr(ref.backend, "name", type(ref.backend).__name__),
                "backend_error", str(exc), {},
            )
        supported = np.asarray(cres.supported, bool)
        usable = supported & np.isfinite(cres.p_values)
        state = "valid" if ref.health_status != "conservative" else "power_limited"
        if not np.all(supported):
            state = "partial_support" if np.any(supported) else "reference_unsupported"
        return ProductionCalibrationResult(
            np.asarray(cres.p_values, float), supported, usable,
            ref.meta.reference_id, key, cres.backend, state,
            "p-values are usable only where supported=true",
            dict(cres.diagnostics) | {"reference_health": ref.health_status},
        )

    def audit_active(
        self,
        key: Hashable,
        sentinel_scores,
        specs: Sequence[ContactSpec],
        *,
        A=None,
        G=None,
        backend_kwargs: Mapping[str, object] | None = None,
        min_samples: int | None = None,
    ) -> CalibrationHealth | None:
        ref = self._active.get(key)
        if ref is None:
            return None
        if topology_signature(specs, A, G) != ref.topology_sig:
            ref.health_status = "blocked"
            self._history.append({"event":"audit_active_topology_mismatch","key":repr(key),"reference_id":ref.meta.reference_id})
            return None
        try:
            cres = self._call_backend(ref, sentinel_scores, specs, A=A, G=G, backend_kwargs=backend_kwargs)
        except Exception:
            ref.health_status = "blocked"
            self._history.append({"event":"audit_active_backend_error","key":repr(key),"reference_id":ref.meta.reference_id})
            return None
        supported = np.asarray(cres.supported, bool)
        p_all = np.asarray(cres.p_values, float)
        p = p_all[supported & np.isfinite(p_all)]
        health = audit_p_value_series(
            p,
            alpha=self.alpha,
            min_samples=self.min_sentinel_samples if min_samples is None else int(min_samples),
            block_size=self.sentinel_block_size,
            n_bootstrap=self.sentinel_bootstrap_reps,
            random_state=self.random_state,
        )
        frac = float(np.mean(supported)) if len(supported) else 0.0
        # Inadequate backend support is a reference-support failure, not a
        # calibration-health success merely because the few remaining p-values
        # happen to look calibrated.
        if frac < self.min_supported_fraction:
            ref.health_status = "blocked"
        else:
            ref.health_status = health.status
        ref.health = health
        ref.validation_supported_fraction = frac
        self._history.append({
            "event": "audit_active", "key": repr(key), "reference_id": ref.meta.reference_id,
            "health": ref.health_status, "supported_fraction": frac,
            "empirical_fpr": health.empirical_fpr,
        })
        return health

    def validate_candidate(
        self,
        key: Hashable,
        sentinel_scores,
        specs: Sequence[ContactSpec],
        *,
        A=None,
        G=None,
        backend_kwargs: Mapping[str, object] | None = None,
        min_samples: int | None = None,
        allow_conservative: bool = True,
        sentinel_id: str | None = None,
    ) -> CandidateValidation:
        ref = self._shadow.get(key)
        if ref is None:
            raise KeyError(f"no staged candidate for {key!r}")
        if self.require_disjoint_sentinel:
            if not sentinel_id:
                ref.lifecycle_state = "rejected"; ref.health_status = "rejected"
                return CandidateValidation(ref.meta.reference_id, key, False, False, 0.0, None, "independent sentinel_id is required")
            if ref.meta.reference_data_id and str(sentinel_id) == ref.meta.reference_data_id:
                ref.lifecycle_state = "rejected"; ref.health_status = "rejected"
                return CandidateValidation(ref.meta.reference_id, key, False, False, 0.0, None, "shadow sentinel must be disjoint from candidate reference data")
        if topology_signature(specs, A, G) != ref.topology_sig:
            ref.lifecycle_state = "rejected"; ref.health_status = "rejected"
            return CandidateValidation(ref.meta.reference_id, key, False, False, 0.0, None, "candidate topology mismatch")
        try:
            cres = self._call_backend(ref, sentinel_scores, specs, A=A, G=G, backend_kwargs=backend_kwargs)
        except Exception as exc:
            ref.lifecycle_state = "rejected"; ref.health_status = "rejected"
            return CandidateValidation(ref.meta.reference_id, key, False, False, 0.0, None, f"backend error: {exc}")
        supported = np.asarray(cres.supported, bool)
        frac = float(np.mean(supported)) if len(supported) else 0.0
        p = np.asarray(cres.p_values, float)[supported & np.isfinite(cres.p_values)]
        health = audit_p_value_series(
            p,
            alpha=self.alpha,
            min_samples=self.min_sentinel_samples if min_samples is None else int(min_samples),
            block_size=self.sentinel_block_size,
            n_bootstrap=self.sentinel_bootstrap_reps,
            random_state=self.random_state,
        )
        accepted = frac >= self.min_supported_fraction and health.status in {"healthy", "conservative"}
        if health.status == "conservative" and not allow_conservative:
            accepted = False
        power_limited = health.status == "conservative"
        ref.health = health; ref.validation_supported_fraction = frac
        ref.health_status = health.status if frac >= self.min_supported_fraction else "blocked"
        ref.lifecycle_state = "shadow" if accepted else "rejected"
        reason = (
            "candidate passed shadow sentinel"
            if accepted and not power_limited else
            "candidate passed validity audit but is power-limited"
            if accepted else
            "candidate failed support/health audit"
        )
        self._history.append({
            "event": "validate_candidate", "key": repr(key), "reference_id": ref.meta.reference_id,
            "accepted": accepted, "health": ref.health_status, "supported_fraction": frac,
            "empirical_fpr": health.empirical_fpr,
        })
        return CandidateValidation(ref.meta.reference_id, key, accepted, power_limited, frac, health, reason)

    def promote(self, key: Hashable) -> str:
        cand = self._shadow.get(key)
        if cand is None:
            raise KeyError(f"no staged candidate for {key!r}")
        if cand.lifecycle_state == "rejected" or cand.health_status not in {"healthy", "conservative"}:
            raise RuntimeError("candidate has not passed shadow validation")
        old = self._active.get(key)
        if old is not None:
            old.lifecycle_state = "retired"
            self._retired.setdefault(key, []).append(old)
        cand.lifecycle_state = "active"
        self._active[key] = cand
        del self._shadow[key]
        self._history.append({"event": "promote", "key": repr(key), "reference_id": cand.meta.reference_id})
        return cand.meta.reference_id

    def active_info(self, key: Hashable) -> Mapping[str, object] | None:
        ref = self._active.get(key)
        if ref is None:
            return None
        return {
            "reference_id": ref.meta.reference_id,
            "key": ref.meta.key,
            "provenance": ref.meta.provenance,
            "dependency_set": sorted(ref.meta.dependency_set),
            "scope": ref.meta.scope,
            "reference_data_id": ref.meta.reference_data_id,
            "backend": getattr(ref.backend, "name", type(ref.backend).__name__),
            "health_status": ref.health_status,
            "lifecycle_state": ref.lifecycle_state,
            "generation": ref.generation,
            "validation_supported_fraction": ref.validation_supported_fraction,
        }

    def rollback(self, key: Hashable) -> str:
        stack = self._retired.get(key, [])
        if not stack:
            raise RuntimeError(f"no retired reference available for {key!r}")
        current = self._active.get(key)
        if current is not None:
            current.lifecycle_state = "retired"
        prev = stack.pop()
        prev.lifecycle_state = "active"
        self._active[key] = prev
        self._history.append({"event":"rollback","key":repr(key),"reference_id":prev.meta.reference_id})
        return prev.meta.reference_id

    def manifest(self) -> Mapping[str, object]:
        return {
            "active": {repr(k): dict(self.active_info(k) or {}) for k in self._active},
            "shadow": {repr(k): {"reference_id":v.meta.reference_id,"provenance":v.meta.provenance,"reference_data_id":v.meta.reference_data_id,"health_status":v.health_status} for k,v in self._shadow.items()},
            "retired": {repr(k): [v.meta.reference_id for v in vs] for k,vs in self._retired.items()},
            "history": list(self.history),
        }

    @property
    def history(self) -> tuple[Mapping[str, object], ...]:
        return tuple(copy.deepcopy(self._history))
