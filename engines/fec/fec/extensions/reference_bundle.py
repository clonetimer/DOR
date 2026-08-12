from __future__ import annotations

"""Atomic Leaf + Root calibration reference lifecycle for FEC.

The Frozen Core deliberately keeps leaf Evidence and root calibration separate.
That separation is useful scientifically, but a production refresh must not
replace the root calibration while accidentally retaining an incompatible leaf
normal reference. This Extension wraps the v1.5 ProductionCalibrationManager
and versions both references as one bundle.

This module does not choose when a sample is trusted normal. U/D_cal and an
externally declared context key remain prerequisites. It also does not promise
to repair arbitrary non-exchangeability; a candidate is usable only after an
independent raw-data sentinel passes the existing v1.5 calibration audit.
"""

from dataclasses import dataclass
from typing import Any, Callable, Hashable, Iterable, Mapping, Protocol, Sequence
import copy

import numpy as np

from fec.core.fec_core import ContactSpec, EmpiricalReference, coordinate, evidence, fit_reference
from .conformal_backend import EmpiricalRootBackend
from .evidence_provenance import ReferenceCompatibilityContract
from .production_calibration import (
    CalibrationReferenceMeta,
    CandidateValidation,
    ProductionCalibrationManager,
    ProductionCalibrationResult,
)


class LeafEvidenceProvider(Protocol):
    """Adapter/Extension contract for a leaf Evidence generator."""

    name: str

    def fit(self, X: np.ndarray) -> Any: ...

    def evidence(self, X: np.ndarray, reference: Any) -> np.ndarray: ...


class CoreEmpiricalLeafProvider:
    """Default provider using the unchanged Frozen Core two-sided Evidence."""

    name = "core_empirical_two_sided"

    def fit(self, X: np.ndarray) -> EmpiricalReference:
        return fit_reference(np.asarray(X, dtype=float))

    def evidence(self, X: np.ndarray, reference: EmpiricalReference) -> np.ndarray:
        return evidence(np.asarray(X, dtype=float), reference)


@dataclass(frozen=True)
class ReferenceBundleMeta:
    bundle_id: str
    key: Hashable
    provenance: str
    dependency_set: frozenset[str]
    leaf_data_ids: frozenset[str]
    root_data_ids: frozenset[str]
    scope: str = "leaf+root"
    notes: str = ""
    compatibility: ReferenceCompatibilityContract | None = None

    def __post_init__(self) -> None:
        if not self.bundle_id:
            raise ValueError("bundle_id is required")
        if not self.leaf_data_ids:
            raise ValueError("leaf_data_ids must be non-empty")
        if not self.root_data_ids:
            raise ValueError("root_data_ids must be non-empty")
        if self.compatibility is not None and not isinstance(self.compatibility, ReferenceCompatibilityContract):
            raise TypeError("compatibility must be a ReferenceCompatibilityContract or None")
        overlap = self.leaf_data_ids & self.root_data_ids
        if overlap:
            raise ValueError(f"leaf/root reference data must be disjoint: {sorted(overlap)}")

    @property
    def reference_data_ids(self) -> frozenset[str]:
        return self.leaf_data_ids | self.root_data_ids


@dataclass
class ManagedReferenceBundle:
    meta: ReferenceBundleMeta
    leaf_reference: Any
    specs: tuple[ContactSpec, ...]
    A: np.ndarray
    G: np.ndarray
    operators: tuple[str, ...] | None
    expected_contact_count: int
    lifecycle_state: str = "shadow"
    health_status: str = "unknown"
    generation: int = 0
    validated_for_promotion: bool = False
    power_limited: bool = False


class ReferenceBundleManager:
    """Version leaf reference + root calibration reference atomically.

    Candidate construction is end-to-end from raw trusted-normal Contact rows:

        X_leaf -> leaf reference
        X_root -> leaf Evidence -> FEC coordinate -> root backend

    Candidate validation is also end-to-end from *raw* independent sentinel
    rows. Promotion and rollback update the leaf and root references together.
    """

    def __init__(
        self,
        *,
        leaf_provider: LeafEvidenceProvider | None = None,
        backend_factory: Callable[[], Any] | None = None,
        alpha: float = 0.05,
        min_sentinel_samples: int = 20,
        min_supported_fraction: float = 0.8,
        sentinel_block_size: int = 1,
        sentinel_bootstrap_reps: int = 2000,
        random_state: int = 0,
        require_disjoint_sentinel: bool = True,
    ):
        self.leaf_provider = leaf_provider or CoreEmpiricalLeafProvider()
        self.backend_factory = backend_factory or EmpiricalRootBackend
        self._root = ProductionCalibrationManager(
            alpha=alpha,
            min_sentinel_samples=min_sentinel_samples,
            min_supported_fraction=min_supported_fraction,
            sentinel_block_size=sentinel_block_size,
            sentinel_bootstrap_reps=sentinel_bootstrap_reps,
            random_state=random_state,
            require_disjoint_sentinel=require_disjoint_sentinel,
        )
        self._active: dict[Hashable, ManagedReferenceBundle] = {}
        self._shadow: dict[Hashable, ManagedReferenceBundle] = {}
        self._retired: dict[Hashable, list[ManagedReferenceBundle]] = {}
        self._history: list[dict[str, object]] = []
        self._generation = 0

    @staticmethod
    def _as_vector(x, n: int, *, kind: str) -> np.ndarray:
        if x is None:
            return np.ones(n, dtype=float)
        a = np.asarray(x, dtype=float)
        if a.ndim != 1 or len(a) != n:
            raise ValueError(f"{kind} must be a 1D vector of length {n}")
        return a.copy()

    @staticmethod
    def _ids(ids: Iterable[str]) -> frozenset[str]:
        out = frozenset(str(x) for x in ids)
        if not out or any(not x for x in out):
            raise ValueError("data ids must be non-empty strings")
        return out

    @staticmethod
    def _check_matrix(X, n_contacts: int, *, name: str) -> np.ndarray:
        a = np.asarray(X, dtype=float)
        if a.ndim != 2 or a.shape[1] != n_contacts or len(a) < 2:
            raise ValueError(f"{name} must be 2D with {n_contacts} columns and at least two rows")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{name} must be finite")
        return a

    def _root_scores(self, bundle: ManagedReferenceBundle, X) -> np.ndarray:
        X = self._check_matrix(X, len(bundle.specs), name="X")
        E = np.asarray(self.leaf_provider.evidence(X, bundle.leaf_reference), dtype=float)
        if E.shape != X.shape or not np.all(np.isfinite(E)):
            raise ValueError("leaf provider must return finite Evidence with the same shape as X")
        state = coordinate(
            E,
            bundle.specs,
            A=bundle.A,
            G=bundle.G,
            operators=bundle.operators,
            expected_contact_count=bundle.expected_contact_count,
        )
        return np.asarray(state.root_E, dtype=float)

    @staticmethod
    def _compatibility_reasons(
        bundle: ManagedReferenceBundle,
        runtime_compatibility: ReferenceCompatibilityContract | None,
    ) -> tuple[str, ...]:
        expected = bundle.meta.compatibility
        if expected is None:
            return ()
        if runtime_compatibility is None:
            return ("runtime reference-compatibility contract was not provided",)
        return expected.mismatch_reasons(runtime_compatibility)

    @staticmethod
    def _unsupported_compatibility_result(
        bundle: ManagedReferenceBundle,
        n: int,
        reasons: Sequence[str],
    ) -> ProductionCalibrationResult:
        return ProductionCalibrationResult(
            np.full(n, np.nan),
            np.zeros(n, dtype=bool),
            np.zeros(n, dtype=bool),
            f"{bundle.meta.bundle_id}:root",
            bundle.meta.key,
            None,
            "reference_unsupported",
            "active Reference Bundle is incompatible with current modality/provider contract",
            {"compatibility_mismatch": list(reasons)},
        )

    def stage_candidate(
        self,
        meta: ReferenceBundleMeta,
        X_leaf,
        X_root,
        specs: Sequence[ContactSpec],
        *,
        trusted_claims: Iterable[str],
        A=None,
        G=None,
        operators: Sequence[str] | None = None,
        expected_contact_count: int | None = None,
        backend_fit_kwargs: Mapping[str, object] | None = None,
    ) -> bool:
        specs_t = tuple(specs)
        if not specs_t:
            raise ValueError("at least one ContactSpec is required")
        Xl = self._check_matrix(X_leaf, len(specs_t), name="X_leaf")
        Xr = self._check_matrix(X_root, len(specs_t), name="X_root")
        avec = self._as_vector(A, len(specs_t), kind="A")
        gvec = self._as_vector(G, len(specs_t), kind="G")
        expected = int(len(specs_t) if expected_contact_count is None else expected_contact_count)
        if expected <= 0:
            raise ValueError("expected_contact_count must be positive")
        ops = None if operators is None else tuple(str(x) for x in operators)
        claims = frozenset(str(x) for x in trusted_claims)
        if not meta.dependency_set.issubset(claims):
            self._history.append({
                "event": "stage_rejected_dependency",
                "key": repr(meta.key),
                "bundle_id": meta.bundle_id,
                "missing": sorted(meta.dependency_set - claims),
            })
            return False

        leaf_ref = self.leaf_provider.fit(Xl)
        bundle = ManagedReferenceBundle(
            meta=meta,
            leaf_reference=leaf_ref,
            specs=specs_t,
            A=avec,
            G=gvec,
            operators=ops,
            expected_contact_count=expected,
        )
        root_scores = self._root_scores(bundle, Xr)
        backend = self.backend_factory()
        fit_kwargs = dict(backend_fit_kwargs or {})
        backend.fit(root_scores, specs=specs_t, A=avec, G=gvec, **fit_kwargs)
        root_meta = CalibrationReferenceMeta(
            reference_id=f"{meta.bundle_id}:root",
            key=meta.key,
            provenance=meta.provenance,
            dependency_set=meta.dependency_set,
            scope=meta.scope,
            notes=f"atomic reference bundle; leaf_provider={self.leaf_provider.name}; {meta.notes}".strip(),
            reference_data_id="|".join(sorted(meta.reference_data_ids)),
        )
        staged = self._root.stage_candidate(
            root_meta,
            backend,
            specs_t,
            trusted_claims=claims,
            A=avec,
            G=gvec,
        )
        if not staged:
            return False
        self._generation += 1
        bundle.generation = self._generation
        self._shadow[meta.key] = bundle
        self._history.append({
            "event": "stage_bundle",
            "key": repr(meta.key),
            "bundle_id": meta.bundle_id,
            "leaf_data_ids": sorted(meta.leaf_data_ids),
            "root_data_ids": sorted(meta.root_data_ids),
        })
        return True

    def validate_candidate(
        self,
        key: Hashable,
        X_sentinel,
        *,
        sentinel_data_ids: Iterable[str],
        backend_kwargs: Mapping[str, object] | None = None,
        min_samples: int | None = None,
        allow_conservative: bool = True,
        protect_active_power: bool = True,
        sentinel_compatibility: ReferenceCompatibilityContract | None = None,
    ) -> CandidateValidation:
        bundle = self._shadow.get(key)
        if bundle is None:
            raise KeyError(f"no staged reference bundle for {key!r}")
        mismatch = self._compatibility_reasons(bundle, sentinel_compatibility)
        if mismatch:
            bundle.lifecycle_state = "rejected"
            bundle.validated_for_promotion = False
            self._history.append({
                "event": "validate_rejected_compatibility",
                "key": repr(key),
                "bundle_id": bundle.meta.bundle_id,
                "reasons": list(mismatch),
            })
            return CandidateValidation(
                f"{bundle.meta.bundle_id}:root", key, False, False, 0.0, None,
                "sentinel modality/provider contract is incompatible with staged Reference Bundle",
            )
        sentinel_ids = self._ids(sentinel_data_ids)
        overlap = sentinel_ids & bundle.meta.reference_data_ids
        if overlap:
            self._history.append({
                "event": "validate_rejected_data_overlap",
                "key": repr(key),
                "bundle_id": bundle.meta.bundle_id,
                "overlap": sorted(overlap),
            })
            raise ValueError(f"sentinel data overlaps bundle reference data: {sorted(overlap)}")
        scores = self._root_scores(bundle, X_sentinel)
        result = self._root.validate_candidate(
            key,
            scores,
            bundle.specs,
            A=bundle.A,
            G=bundle.G,
            backend_kwargs=backend_kwargs,
            min_samples=min_samples,
            allow_conservative=allow_conservative,
            sentinel_id="|".join(sorted(sentinel_ids)),
        )
        bundle.health_status = "unknown" if result.health is None else result.health.status
        bundle.power_limited = bool(result.power_limited)
        promotion_allowed = bool(result.accepted_for_promotion)
        active = self._active.get(key)
        protected_power_regression = bool(
            promotion_allowed
            and protect_active_power
            and result.power_limited
            and active is not None
            and active.health_status == "healthy"
        )
        if protected_power_regression:
            promotion_allowed = False
            reason = "candidate is validity-safe but power-limited; automatic replacement of a healthy active bundle is blocked"
            wrapped = CandidateValidation(
                result.reference_id, result.key, False, True, result.supported_fraction, result.health, reason
            )
            bundle.lifecycle_state = "power_review"
        else:
            wrapped = result
            bundle.lifecycle_state = "validated" if promotion_allowed else "rejected"
        bundle.validated_for_promotion = promotion_allowed
        self._history.append({
            "event": "validate_bundle",
            "key": repr(key),
            "bundle_id": bundle.meta.bundle_id,
            "accepted": promotion_allowed,
            "backend_accepted": result.accepted_for_promotion,
            "health": bundle.health_status,
            "power_limited": bundle.power_limited,
            "protected_power_regression": protected_power_regression,
            "sentinel_data_ids": sorted(sentinel_ids),
        })
        return wrapped

    def promote(self, key: Hashable) -> str:
        bundle = self._shadow.get(key)
        if bundle is None:
            raise KeyError(f"no staged reference bundle for {key!r}")
        if not bundle.validated_for_promotion:
            raise RuntimeError("reference bundle has not passed bundle-level promotion policy")
        root_reference_id = self._root.promote(key)
        old = self._active.get(key)
        if old is not None:
            old.lifecycle_state = "retired"
            self._retired.setdefault(key, []).append(old)
        bundle.lifecycle_state = "active"
        self._active[key] = bundle
        del self._shadow[key]
        self._history.append({
            "event": "promote_bundle",
            "key": repr(key),
            "bundle_id": bundle.meta.bundle_id,
            "root_reference_id": root_reference_id,
        })
        return bundle.meta.bundle_id

    def evaluate(
        self,
        key: Hashable,
        X,
        *,
        backend_kwargs: Mapping[str, object] | None = None,
        runtime_compatibility: ReferenceCompatibilityContract | None = None,
    ) -> ProductionCalibrationResult:
        Xarr = np.asarray(X, dtype=float)
        bundle = self._active.get(key)
        if bundle is None:
            # Delegate unsupported-state formatting to the existing v1.5 manager.
            n = len(Xarr) if Xarr.ndim >= 1 else 1
            return self._root.evaluate(key, np.zeros(n), (), backend_kwargs=backend_kwargs)
        n = len(Xarr) if Xarr.ndim >= 1 else 1
        mismatch = self._compatibility_reasons(bundle, runtime_compatibility)
        if mismatch:
            self._history.append({
                "event": "evaluate_reference_unsupported_compatibility",
                "key": repr(key),
                "bundle_id": bundle.meta.bundle_id,
                "reasons": list(mismatch),
            })
            return self._unsupported_compatibility_result(bundle, n, mismatch)
        scores = self._root_scores(bundle, Xarr)
        return self._root.evaluate(
            key,
            scores,
            bundle.specs,
            A=bundle.A,
            G=bundle.G,
            backend_kwargs=backend_kwargs,
        )

    def audit_active(
        self,
        key: Hashable,
        X_sentinel,
        *,
        sentinel_data_ids: Iterable[str],
        backend_kwargs: Mapping[str, object] | None = None,
        min_samples: int | None = None,
        sentinel_compatibility: ReferenceCompatibilityContract | None = None,
    ):
        bundle = self._active.get(key)
        if bundle is None:
            return None
        mismatch = self._compatibility_reasons(bundle, sentinel_compatibility)
        if mismatch:
            self._history.append({
                "event": "audit_rejected_compatibility",
                "key": repr(key),
                "bundle_id": bundle.meta.bundle_id,
                "reasons": list(mismatch),
            })
            raise ValueError("trusted sentinel modality/provider contract is incompatible with active Reference Bundle")
        sentinel_ids = self._ids(sentinel_data_ids)
        overlap = sentinel_ids & bundle.meta.reference_data_ids
        if overlap:
            raise ValueError(f"sentinel data overlaps active bundle reference data: {sorted(overlap)}")
        scores = self._root_scores(bundle, X_sentinel)
        health = self._root.audit_active(
            key,
            scores,
            bundle.specs,
            A=bundle.A,
            G=bundle.G,
            backend_kwargs=backend_kwargs,
            min_samples=min_samples,
        )
        info = self._root.active_info(key)
        if info is not None:
            bundle.health_status = str(info["health_status"])
        self._history.append({
            "event": "audit_active_bundle",
            "key": repr(key),
            "bundle_id": bundle.meta.bundle_id,
            "health": bundle.health_status,
            "sentinel_data_ids": sorted(sentinel_ids),
        })
        return health

    def rollback(self, key: Hashable) -> str:
        stack = self._retired.get(key, [])
        if not stack:
            raise RuntimeError(f"no retired reference bundle available for {key!r}")
        self._root.rollback(key)
        current = self._active.get(key)
        if current is not None:
            current.lifecycle_state = "retired"
        prev = stack.pop()
        prev.lifecycle_state = "active"
        root_info = self._root.active_info(key)
        prev.health_status = "unknown" if root_info is None else str(root_info["health_status"])
        self._active[key] = prev
        self._history.append({"event": "rollback_bundle", "key": repr(key), "bundle_id": prev.meta.bundle_id})
        return prev.meta.bundle_id

    def active_info(self, key: Hashable) -> Mapping[str, object] | None:
        b = self._active.get(key)
        if b is None:
            return None
        root = self._root.active_info(key)
        return {
            "bundle_id": b.meta.bundle_id,
            "key": b.meta.key,
            "provenance": b.meta.provenance,
            "compatibility": None if b.meta.compatibility is None else b.meta.compatibility.to_dict(),
            "dependency_set": sorted(b.meta.dependency_set),
            "leaf_data_ids": sorted(b.meta.leaf_data_ids),
            "root_data_ids": sorted(b.meta.root_data_ids),
            "leaf_provider": self.leaf_provider.name,
            "lifecycle_state": b.lifecycle_state,
            "health_status": b.health_status,
            "generation": b.generation,
            "root_reference": root,
        }

    def manifest(self) -> Mapping[str, object]:
        return {
            "active": {repr(k): dict(self.active_info(k) or {}) for k in self._active},
            "shadow": {
                repr(k): {
                    "bundle_id": v.meta.bundle_id,
                    "leaf_data_ids": sorted(v.meta.leaf_data_ids),
                    "root_data_ids": sorted(v.meta.root_data_ids),
                    "compatibility": None if v.meta.compatibility is None else v.meta.compatibility.to_dict(),
                    "health_status": v.health_status,
                    "validated_for_promotion": v.validated_for_promotion,
                    "power_limited": v.power_limited,
                }
                for k, v in self._shadow.items()
            },
            "retired": {repr(k): [v.meta.bundle_id for v in vs] for k, vs in self._retired.items()},
            "history": copy.deepcopy(self._history),
            "root_manager": self._root.manifest(),
        }

# Public audit-log view for persistence/operations tooling.
def _reference_bundle_history(self):
    return tuple(copy.deepcopy(self._history))

ReferenceBundleManager.history = property(_reference_bundle_history)
