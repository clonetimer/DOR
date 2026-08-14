from __future__ import annotations

"""Cross-asset admission and reference-uncertainty robustness gates for FEC v2.3-rc4.

A shared schema/provider stack is not evidence that a normal calibration
Reference is exchangeable across physical assets.  This module lets a candidate
asset challenge an already-active Reference using an *independent externally
trusted normal sentinel* before that asset can be added to a fleet scope.

The gate deliberately bypasses only the asset-membership check during the audit.
Modality, provider-fingerprint, source-domain, topology, support and data-role
contracts remain enforced.  It does not infer that an arbitrary online batch is
healthy and it does not mutate the active Reference automatically.
"""

from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping

import numpy as np

from .drift_guard import CalibrationHealth, audit_p_value_series
from .evidence_provenance import AssetReferenceScope, ReferenceCompatibilityContract
from .reference_bundle import ReferenceBundleManager


@dataclass(frozen=True)
class FleetAssetAdmissionDecision:
    asset_id: str
    fleet_group_id: str
    accepted: bool
    state: str
    reason: str
    supported_fraction: float
    health: CalibrationHealth | None
    sentinel_data_ids: frozenset[str]
    bundle_id: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "asset_id": self.asset_id,
            "fleet_group_id": self.fleet_group_id,
            "accepted": self.accepted,
            "state": self.state,
            "reason": self.reason,
            "supported_fraction": self.supported_fraction,
            "health": None if self.health is None else self.health.__dict__,
            "sentinel_data_ids": sorted(self.sentinel_data_ids),
            "bundle_id": self.bundle_id,
        }



def audit_asset_for_fleet_reference(
    manager: ReferenceBundleManager,
    key: Hashable,
    X_trusted_normal,
    *,
    asset_id: str,
    fleet_group_id: str,
    sentinel_data_ids: Iterable[str],
    runtime_compatibility: ReferenceCompatibilityContract,
    backend_kwargs: Mapping[str, object] | None = None,
    min_samples: int | None = None,
    allow_conservative: bool = False,
) -> FleetAssetAdmissionDecision:
    """Audit whether one physical asset may reuse an existing Reference.

    `X_trusted_normal` must be independently justified healthy/normal.  The
    asset id is not inferred from model output and fault labels are not needed.
    The active bundle is not modified by this function.
    """
    aid = str(asset_id)
    gid = str(fleet_group_id)
    if not aid or not gid:
        raise ValueError("asset_id and fleet_group_id are required")
    if runtime_compatibility.runtime_asset_id != aid:
        return FleetAssetAdmissionDecision(
            aid, gid, False, "reference_unsupported",
            "runtime compatibility asset_id does not match candidate asset", 0.0,
            None, frozenset(str(x) for x in sentinel_data_ids), None,
        )
    bundle = manager._active.get(key)
    ids = frozenset(str(x) for x in sentinel_data_ids)
    if bundle is None:
        return FleetAssetAdmissionDecision(
            aid, gid, False, "reference_unsupported", "no active Reference Bundle", 0.0,
            None, ids, None,
        )
    if not ids:
        raise ValueError("sentinel_data_ids must be non-empty")
    overlap = ids & bundle.meta.reference_data_ids
    if overlap:
        raise ValueError(f"fleet sentinel data overlaps active reference data: {sorted(overlap)}")

    # Bypass only the commissioned asset membership.  Every other hard
    # compatibility dimension still has to match exactly.
    expected = bundle.meta.compatibility
    if expected is not None:
        expected_common = ReferenceCompatibilityContract(
            modality_signature=expected.modality_signature,
            provider_fingerprints=expected.provider_fingerprints,
            source_domain=expected.source_domain,
            operating_context=expected.operating_context,
        )
        mismatch = expected_common.mismatch_reasons(runtime_compatibility)
        if mismatch:
            return FleetAssetAdmissionDecision(
                aid, gid, False, "reference_unsupported",
                "candidate asset does not match non-asset Reference contract: " + "; ".join(mismatch),
                0.0, None, ids, bundle.meta.bundle_id,
            )

    X = manager._check_matrix(X_trusted_normal, len(bundle.specs), name="X_trusted_normal")
    scores = manager._root_scores(bundle, X)
    result = manager._root.evaluate(
        key,
        scores,
        bundle.specs,
        A=bundle.A,
        G=bundle.G,
        backend_kwargs=backend_kwargs,
    )
    supported = np.asarray(result.supported, dtype=bool)
    p_values = np.asarray(result.p_values, dtype=float)
    usable = supported & np.isfinite(p_values)
    fraction = float(np.mean(supported)) if len(supported) else 0.0
    root = manager._root
    health = audit_p_value_series(
        p_values[usable],
        alpha=root.alpha,
        min_samples=root.min_sentinel_samples if min_samples is None else int(min_samples),
        block_size=root.sentinel_block_size,
        n_bootstrap=root.sentinel_bootstrap_reps,
        random_state=root.random_state,
    )
    if fraction < root.min_supported_fraction:
        return FleetAssetAdmissionDecision(
            aid, gid, False, "reference_unsupported",
            "candidate asset has inadequate backend support", fraction, health, ids, bundle.meta.bundle_id,
        )
    if health.status == "healthy":
        return FleetAssetAdmissionDecision(
            aid, gid, True, "admitted", "independent trusted-normal sentinel is calibration-compatible",
            fraction, health, ids, bundle.meta.bundle_id,
        )
    if health.status == "conservative" and allow_conservative:
        return FleetAssetAdmissionDecision(
            aid, gid, True, "admitted_power_limited",
            "candidate is calibration-safe but conservative/power-limited; explicit override was supplied",
            fraction, health, ids, bundle.meta.bundle_id,
        )
    if health.status == "conservative":
        reason = "candidate asset is conservative/power-limited; fleet admission is blocked by default"
    elif health.status == "anti_conservative":
        reason = "candidate asset is anti-conservative under the active Reference"
    else:
        reason = "candidate asset has insufficient trusted-normal sentinel evidence"
    return FleetAssetAdmissionDecision(
        aid, gid, False, "blocked", reason, fraction, health, ids, bundle.meta.bundle_id,
    )


def fleet_scope_from_admissions(
    fleet_group_id: str,
    admissions: Iterable[FleetAssetAdmissionDecision],
    *,
    base_scope: AssetReferenceScope | None = None,
) -> AssetReferenceScope:
    """Create a fleet scope only from successful independent admission records.

    `base_scope` carries forward the asset(s) whose Reference is being widened.
    For the common asset-local -> fleet transition, pass the current local scope
    so the commissioning asset remains a member after admitting new assets.
    """
    gid = str(fleet_group_id)
    decisions = tuple(admissions)
    if not decisions and base_scope is None:
        raise ValueError("at least one base asset or fleet admission decision is required")
    wrong_group = sorted({d.fleet_group_id for d in decisions if d.fleet_group_id != gid})
    if wrong_group:
        raise ValueError(f"admission decisions belong to other fleet groups: {wrong_group}")
    rejected = sorted(d.asset_id for d in decisions if not d.accepted)
    if rejected:
        raise ValueError(f"cannot create fleet scope from rejected/unverified assets: {rejected}")
    assets = set()
    if base_scope is not None:
        if base_scope.kind.value == "fleet_group" and base_scope.fleet_group_id != gid:
            raise ValueError(
                f"base fleet scope belongs to {base_scope.fleet_group_id!r}, not {gid!r}"
            )
        assets.update(base_scope.admitted_asset_ids)
    assets.update(d.asset_id for d in decisions)
    return AssetReferenceScope.fleet_group(gid, assets)


@dataclass(frozen=True)
class FleetReferenceRobustnessDecision:
    """Aggregate one asset's fleet-admission results across Reference variants.

    The variants are expected to be *separately validated Reference Bundles*.
    This object is a robustness/promotion decision, not a claim that the
    variants or their sentinel observations are statistically independent.
    """

    asset_id: str
    fleet_group_id: str
    accepted: bool
    state: str
    reason: str
    planned_variant_count: int
    valid_variant_count: int
    valid_variant_fraction: float
    accepted_variant_count: int
    accepted_variant_fraction: float
    anti_conservative_variant_count: int
    conservative_variant_count: int
    insufficient_variant_count: int
    reference_unsupported_variant_count: int
    bundle_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "asset_id": self.asset_id,
            "fleet_group_id": self.fleet_group_id,
            "accepted": self.accepted,
            "state": self.state,
            "reason": self.reason,
            "planned_variant_count": self.planned_variant_count,
            "valid_variant_count": self.valid_variant_count,
            "valid_variant_fraction": self.valid_variant_fraction,
            "accepted_variant_count": self.accepted_variant_count,
            "accepted_variant_fraction": self.accepted_variant_fraction,
            "anti_conservative_variant_count": self.anti_conservative_variant_count,
            "conservative_variant_count": self.conservative_variant_count,
            "insufficient_variant_count": self.insufficient_variant_count,
            "reference_unsupported_variant_count": self.reference_unsupported_variant_count,
            "bundle_ids": list(self.bundle_ids),
        }


def robust_fleet_admission_from_variants(
    admissions: Iterable[FleetAssetAdmissionDecision],
    *,
    planned_variant_count: int,
    min_valid_variants: int = 2,
    min_valid_fraction: float = 1.0,
    min_accept_fraction: float = 1.0,
    block_on_any_anti_conservative: bool = True,
) -> FleetReferenceRobustnessDecision:
    """Require cross-Reference robustness before widening an asset scope.

    A single finite-sample Reference construction can make a fleet admission
    appear compatible even when another separately validated construction of
    the *same normal Reference population* is anti-conservative.  This gate
    aggregates already-computed :class:`FleetAssetAdmissionDecision` records
    and, by default, requires unanimity across sufficiently many valid
    Reference variants while hard-blocking any anti-conservative variant.

    `planned_variant_count` includes variants that were predeclared but failed
    their own independent Reference validation and therefore could not produce
    an asset admission decision.  Such failures reduce reference-variant
    coverage rather than being silently discarded.

    This is a robustness gate, not a substitute for more independent normal
    data.  Reusing the same observations across folds/bootstraps does not
    increase the effective sample size.
    """
    ds = tuple(admissions)
    if not ds:
        raise ValueError("at least one valid Reference-variant admission decision is required")
    if min_valid_variants <= 0:
        raise ValueError("min_valid_variants must be positive")
    for name, value in (("min_valid_fraction", min_valid_fraction), ("min_accept_fraction", min_accept_fraction)):
        if not (0.0 < float(value) <= 1.0):
            raise ValueError(f"{name} must be in (0, 1]")

    asset_ids = {d.asset_id for d in ds}
    group_ids = {d.fleet_group_id for d in ds}
    if len(asset_ids) != 1:
        raise ValueError(f"all variant decisions must belong to one asset: {sorted(asset_ids)}")
    if len(group_ids) != 1:
        raise ValueError(f"all variant decisions must belong to one fleet group: {sorted(group_ids)}")
    aid = next(iter(asset_ids))
    gid = next(iter(group_ids))

    bundle_ids = tuple(str(d.bundle_id) for d in ds if d.bundle_id is not None)
    if len(bundle_ids) != len(ds):
        raise ValueError("every variant decision must identify its validated Reference bundle_id")
    if len(set(bundle_ids)) != len(bundle_ids):
        raise ValueError("Reference-variant bundle_ids must be unique; duplicate votes are not allowed")

    valid_n = len(ds)
    planned_n = int(planned_variant_count)
    if planned_n < valid_n or planned_n <= 0:
        raise ValueError("planned_variant_count must be >= the number of valid variant decisions")
    valid_fraction = valid_n / planned_n

    accepted_n = sum(bool(d.accepted) for d in ds)
    accept_fraction = accepted_n / valid_n
    anti_n = sum(d.health is not None and d.health.status == "anti_conservative" for d in ds)
    conservative_n = sum(d.health is not None and d.health.status == "conservative" for d in ds)
    insufficient_n = sum(d.health is not None and d.health.status.startswith("insufficient") for d in ds)
    unsupported_n = sum(d.state == "reference_unsupported" for d in ds)

    common = dict(
        asset_id=aid,
        fleet_group_id=gid,
        planned_variant_count=planned_n,
        valid_variant_count=valid_n,
        valid_variant_fraction=float(valid_fraction),
        accepted_variant_count=accepted_n,
        accepted_variant_fraction=float(accept_fraction),
        anti_conservative_variant_count=anti_n,
        conservative_variant_count=conservative_n,
        insufficient_variant_count=insufficient_n,
        reference_unsupported_variant_count=unsupported_n,
        bundle_ids=tuple(sorted(bundle_ids)),
    )

    if valid_n < int(min_valid_variants) or valid_fraction < float(min_valid_fraction):
        return FleetReferenceRobustnessDecision(
            accepted=False,
            state="reference_uncertainty_insufficient",
            reason=(
                "too few predeclared Reference variants passed their own validation "
                f"(valid={valid_n}/{planned_n}, required_count>={int(min_valid_variants)}, "
                f"required_fraction>={float(min_valid_fraction):.3f})"
            ),
            **common,
        )
    if block_on_any_anti_conservative and anti_n:
        return FleetReferenceRobustnessDecision(
            accepted=False,
            state="blocked_reference_sensitive",
            reason=(
                f"candidate is anti-conservative under {anti_n}/{valid_n} valid Reference variants; "
                "fleet scope widening is blocked"
            ),
            **common,
        )
    if accept_fraction + 1e-15 < float(min_accept_fraction):
        return FleetReferenceRobustnessDecision(
            accepted=False,
            state="blocked_reference_sensitive",
            reason=(
                f"candidate passed only {accepted_n}/{valid_n} valid Reference variants "
                f"(fraction={accept_fraction:.3f}, required>={float(min_accept_fraction):.3f})"
            ),
            **common,
        )
    return FleetReferenceRobustnessDecision(
        accepted=True,
        state="robustly_admitted",
        reason=(
            f"candidate passed Reference-uncertainty robustness gate across {accepted_n}/{valid_n} "
            "valid Reference variants"
        ),
        **common,
    )


def fleet_scope_from_robust_admissions(
    fleet_group_id: str,
    admissions: Iterable[FleetReferenceRobustnessDecision],
    *,
    base_scope: AssetReferenceScope | None = None,
) -> AssetReferenceScope:
    """Create a fleet scope only from successful multi-Reference robustness records.

    This is the preferred v2.3-rc4 direct-transfer production path.  The older
    :func:`fleet_scope_from_admissions` remains available for backwards
    compatibility and single-Reference experiments, but does not protect
    against finite Reference-construction sensitivity.
    """
    gid = str(fleet_group_id)
    decisions = tuple(admissions)
    if not decisions and base_scope is None:
        raise ValueError("at least one base asset or robust fleet admission decision is required")
    wrong_group = sorted({d.fleet_group_id for d in decisions if d.fleet_group_id != gid})
    if wrong_group:
        raise ValueError(f"robust admission decisions belong to other fleet groups: {wrong_group}")
    rejected = sorted(d.asset_id for d in decisions if not d.accepted)
    if rejected:
        raise ValueError(f"cannot create fleet scope from non-robust/rejected assets: {rejected}")
    assets: set[str] = set()
    if base_scope is not None:
        if base_scope.kind.value == "fleet_group" and base_scope.fleet_group_id != gid:
            raise ValueError(f"base fleet scope belongs to {base_scope.fleet_group_id!r}, not {gid!r}")
        assets.update(base_scope.admitted_asset_ids)
    assets.update(d.asset_id for d in decisions)
    return AssetReferenceScope.fleet_group(gid, assets)


@dataclass(frozen=True)
class FleetReferenceAssetValidation:
    """One member asset's independent sentinel audit for a staged fleet Reference."""

    asset_id: str
    accepted: bool
    state: str
    reason: str
    supported_fraction: float
    health: CalibrationHealth | None
    sentinel_data_ids: frozenset[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "asset_id": self.asset_id,
            "accepted": self.accepted,
            "state": self.state,
            "reason": self.reason,
            "supported_fraction": self.supported_fraction,
            "health": None if self.health is None else self.health.__dict__,
            "sentinel_data_ids": sorted(self.sentinel_data_ids),
        }


@dataclass(frozen=True)
class FleetReferenceCommissioningDecision:
    """Atomic validation result for one staged multi-asset fleet Reference.

    ``accepted`` means every declared fleet member passed this *one* Reference
    construction.  When a multi-construction robustness plan is declared, an
    accepted variant remains ``promotion_pending`` until the aggregate
    robustness decision is explicitly attested to the staged bundle.  This
    keeps the atomic member audit and the cross-construction promotion policy
    separate without allowing the latter to be silently bypassed.
    """

    fleet_group_id: str
    bundle_id: str
    accepted: bool
    state: str
    reason: str
    required_asset_ids: tuple[str, ...]
    asset_validations: tuple[FleetReferenceAssetValidation, ...]
    power_limited: bool
    promotion_guard_id: str | None = None
    planned_variant_count: int | None = None
    promotion_pending: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "fleet_group_id": self.fleet_group_id,
            "bundle_id": self.bundle_id,
            "accepted": self.accepted,
            "state": self.state,
            "reason": self.reason,
            "required_asset_ids": list(self.required_asset_ids),
            "asset_validations": [x.to_dict() for x in self.asset_validations],
            "power_limited": self.power_limited,
            "promotion_guard_id": self.promotion_guard_id,
            "planned_variant_count": self.planned_variant_count,
            "promotion_pending": self.promotion_pending,
        }


def validate_fleet_reference_candidate(
    manager: ReferenceBundleManager,
    key: Hashable,
    sentinels_by_asset: Mapping[str, object],
    *,
    sentinel_data_ids_by_asset: Mapping[str, Iterable[str]],
    runtime_compatibility_by_asset: Mapping[str, ReferenceCompatibilityContract],
    backend_kwargs: Mapping[str, object] | None = None,
    min_samples: int | None = None,
    allow_conservative: bool = False,
    reference_construction_plan_id: str | None = None,
    planned_variant_count: int | None = None,
) -> FleetReferenceCommissioningDecision:
    """Atomically validate a staged fleet Reference against every member asset.

    For a single predeclared Reference construction, existing v2.3 behavior is
    unchanged: an all-member PASS makes the candidate promotion-eligible.

    For a predeclared *multi-construction* plan, pass both
    ``reference_construction_plan_id`` and ``planned_variant_count > 1`` for
    every variant.  An all-member PASS then marks the variant as
    ``validated_robustness_pending`` but deliberately leaves
    ``manager.promote()`` blocked.  The caller must aggregate all predeclared
    variants with :func:`robust_fleet_commissioning_from_variants` and bind a
    passing aggregate decision back to the chosen staged bundle using
    :func:`attest_fleet_reference_robustness`.

    This closes the production gap where a caller could previously validate one
    favorable construction, ignore a failed robustness audit across the full
    predeclared plan, and still call ``manager.promote()`` successfully.
    """
    bundle = manager._shadow.get(key)
    if bundle is None:
        raise KeyError(f"no staged reference bundle for {key!r}")
    compat = bundle.meta.compatibility
    scope = None if compat is None else compat.asset_scope
    if scope is None or scope.kind.value != "fleet_group" or not scope.fleet_group_id:
        raise ValueError("staged Reference Bundle must declare a fleet_group asset scope")

    plan_supplied = reference_construction_plan_id is not None or planned_variant_count is not None
    if plan_supplied and (reference_construction_plan_id is None or planned_variant_count is None):
        raise ValueError(
            "reference_construction_plan_id and planned_variant_count must be supplied together"
        )
    plan_id: str | None = None
    plan_n: int | None = None
    multi_plan = False
    if plan_supplied:
        plan_id = str(reference_construction_plan_id)
        if not plan_id:
            raise ValueError("reference_construction_plan_id must be non-empty")
        plan_n = int(planned_variant_count)
        if plan_n <= 0:
            raise ValueError("planned_variant_count must be positive")
        multi_plan = plan_n > 1

    # A declared multi-construction plan is an immutable promotion policy for
    # this staged candidate.  Revalidating without the plan (or under a
    # different plan) must not silently clear/replace the guard.  Restage a
    # new candidate if the construction plan itself changes.
    if bundle.promotion_guard_id is not None:
        if not multi_plan:
            raise ValueError(
                "staged Reference Bundle already declares a multi-construction promotion guard; "
                "the same reference_construction_plan_id and planned_variant_count are required"
            )
        if (
            plan_id != bundle.promotion_guard_id
            or plan_n != bundle.promotion_guard_planned_variants
        ):
            raise ValueError(
                "staged Reference Bundle promotion guard cannot be changed; restage the candidate for a new plan"
            )
    elif multi_plan:
        bundle.promotion_guard_id = plan_id
        bundle.promotion_guard_planned_variants = plan_n
        bundle.promotion_guard_satisfied = False

    # Single-construction candidates preserve legacy v2.3 behavior.  A plan of
    # one does not create a robustness guard.
    if bundle.promotion_guard_id is None:
        bundle.promotion_guard_planned_variants = None
        bundle.promotion_guard_satisfied = False

    # Promotion for a fleet-scope bundle is legal only after this asset-stratified
    # path succeeds. Reset the shadow attestation on every fresh fleet audit and
    # set it only when *all* declared members pass.
    bundle.fleet_commissioning_satisfied = False

    gid = str(scope.fleet_group_id)
    required = frozenset(str(x) for x in scope.admitted_asset_ids)
    provided = frozenset(str(x) for x in sentinels_by_asset)
    ids_keys = frozenset(str(x) for x in sentinel_data_ids_by_asset)
    runtime_keys = frozenset(str(x) for x in runtime_compatibility_by_asset)
    if provided != required or ids_keys != required or runtime_keys != required:
        missing = sorted(required - (provided & ids_keys & runtime_keys))
        extras = sorted((provided | ids_keys | runtime_keys) - required)
        reason = f"fleet candidate requires one sentinel contract for every admitted asset; missing={missing}, extra={extras}"
        bundle.lifecycle_state = "rejected"
        bundle.validated_for_promotion = False
        bundle.health_status = "blocked"
        root_ref = manager._root._shadow.get(key)
        if root_ref is not None:
            root_ref.lifecycle_state = "rejected"
            root_ref.health_status = "blocked"
        manager._history.append({
            "event": "validate_fleet_bundle_rejected_membership",
            "key": repr(key), "bundle_id": bundle.meta.bundle_id,
            "required_assets": sorted(required), "provided_assets": sorted(provided),
            "reason": reason,
            "promotion_guard_id": plan_id if multi_plan else None,
            "planned_variant_count": plan_n if multi_plan else None,
        })
        return FleetReferenceCommissioningDecision(
            gid, bundle.meta.bundle_id, False, "reference_unsupported", reason,
            tuple(sorted(required)), (), False,
            promotion_guard_id=plan_id if multi_plan else None,
            planned_variant_count=plan_n if multi_plan else None,
            promotion_pending=False,
        )

    # Validate role provenance globally before looking at p-values.
    all_sentinel_ids: set[str] = set()
    normalized_ids: dict[str, frozenset[str]] = {}
    for aid in sorted(required):
        ids = frozenset(str(x) for x in sentinel_data_ids_by_asset[aid])
        if not ids:
            raise ValueError(f"sentinel_data_ids for asset {aid!r} must be non-empty")
        overlap_ref = ids & bundle.meta.reference_data_ids
        if overlap_ref:
            raise ValueError(
                f"fleet sentinel data for asset {aid!r} overlaps bundle reference data: {sorted(overlap_ref)}"
            )
        overlap_other = ids & all_sentinel_ids
        if overlap_other:
            raise ValueError(f"fleet sentinel ids overlap across assets: {sorted(overlap_other)}")
        all_sentinel_ids.update(ids)
        normalized_ids[aid] = ids

    root_ref = manager._root._shadow.get(key)
    if root_ref is None:
        raise RuntimeError("staged Reference Bundle is missing its staged root Reference")

    audits: list[FleetReferenceAssetValidation] = []
    any_power_limited = False
    min_supported_fraction = 1.0
    for aid in sorted(required):
        runtime = runtime_compatibility_by_asset[aid]
        if runtime.runtime_asset_id != aid:
            audits.append(FleetReferenceAssetValidation(
                aid, False, "reference_unsupported",
                "runtime compatibility asset_id does not match sentinel asset",
                0.0, None, normalized_ids[aid],
            ))
            continue
        mismatch = manager._compatibility_reasons(bundle, runtime)
        if mismatch:
            audits.append(FleetReferenceAssetValidation(
                aid, False, "reference_unsupported",
                "sentinel compatibility mismatch: " + "; ".join(mismatch),
                0.0, None, normalized_ids[aid],
            ))
            continue
        X = manager._check_matrix(sentinels_by_asset[aid], len(bundle.specs), name=f"X_sentinel[{aid}]")
        scores = manager._root_scores(bundle, X)
        try:
            cres = manager._root._call_backend(
                root_ref, scores, bundle.specs,
                A=bundle.A, G=bundle.G, backend_kwargs=backend_kwargs,
            )
        except Exception as exc:
            audits.append(FleetReferenceAssetValidation(
                aid, False, "backend_error", f"backend error: {exc}", 0.0,
                None, normalized_ids[aid],
            ))
            continue
        supported = np.asarray(cres.supported, dtype=bool)
        p_values = np.asarray(cres.p_values, dtype=float)
        usable = supported & np.isfinite(p_values)
        frac = float(np.mean(supported)) if len(supported) else 0.0
        min_supported_fraction = min(min_supported_fraction, frac)
        health = audit_p_value_series(
            p_values[usable],
            alpha=manager._root.alpha,
            min_samples=manager._root.min_sentinel_samples if min_samples is None else int(min_samples),
            block_size=manager._root.sentinel_block_size,
            n_bootstrap=manager._root.sentinel_bootstrap_reps,
            random_state=manager._root.random_state,
        )
        accepted = frac >= manager._root.min_supported_fraction and health.status == "healthy"
        state = health.status
        reason = "independent trusted-normal sentinel is calibration-compatible"
        if health.status == "conservative":
            any_power_limited = True
            accepted = bool(allow_conservative and frac >= manager._root.min_supported_fraction)
            reason = (
                "candidate is conservative/power-limited; explicit fleet commissioning override supplied"
                if accepted else
                "candidate is conservative/power-limited; fleet commissioning is blocked by default"
            )
        elif health.status == "anti_conservative":
            reason = "candidate fleet Reference is anti-conservative for this member asset"
        elif health.status.startswith("insufficient"):
            reason = "insufficient trusted-normal sentinel evidence for this member asset"
        elif frac < manager._root.min_supported_fraction:
            accepted = False
            state = "reference_unsupported"
            reason = "candidate fleet Reference has inadequate backend support for this member asset"
        audits.append(FleetReferenceAssetValidation(
            aid, bool(accepted), state, reason, frac, health, normalized_ids[aid]
        ))

    all_accepted = len(audits) == len(required) and all(x.accepted for x in audits)
    promotion_pending = bool(all_accepted and multi_plan)
    if all_accepted:
        bundle.fleet_commissioning_satisfied = True
        bundle.health_status = "conservative" if any_power_limited else "healthy"
        bundle.power_limited = bool(any_power_limited)
        root_ref.lifecycle_state = "shadow"
        root_ref.health_status = bundle.health_status
        root_ref.validation_supported_fraction = float(min_supported_fraction)
        if multi_plan:
            bundle.validated_for_promotion = False
            bundle.lifecycle_state = "robustness_pending"
            state = (
                "validated_power_limited_robustness_pending"
                if any_power_limited else "validated_robustness_pending"
            )
            reason = (
                "every declared fleet asset passed this Reference construction; "
                f"promotion is pending robustness plan {plan_id!r} across {plan_n} predeclared variants"
            )
        else:
            bundle.validated_for_promotion = True
            bundle.lifecycle_state = "validated"
            state = "validated_power_limited" if any_power_limited else "validated"
            reason = "every declared fleet asset passed an independent trusted-normal sentinel audit"
    else:
        bundle.fleet_commissioning_satisfied = False
        bundle.validated_for_promotion = False
        bundle.lifecycle_state = "rejected"
        bundle.power_limited = bool(any_power_limited)
        if any(x.health is not None and x.health.status == "anti_conservative" for x in audits):
            bundle.health_status = "anti_conservative"
        elif any(x.state == "reference_unsupported" for x in audits):
            bundle.health_status = "blocked"
        elif any(x.health is not None and x.health.status == "conservative" for x in audits):
            bundle.health_status = "conservative"
        else:
            bundle.health_status = "blocked"
        root_ref.lifecycle_state = "rejected"
        root_ref.health_status = bundle.health_status
        root_ref.validation_supported_fraction = float(min_supported_fraction)
        state = "blocked"
        failed = [x.asset_id for x in audits if not x.accepted]
        reason = f"fleet candidate failed one or more required asset sentinel audits: {failed}"

    manager._history.append({
        "event": "validate_fleet_bundle",
        "key": repr(key),
        "bundle_id": bundle.meta.bundle_id,
        "fleet_group_id": gid,
        "accepted": bool(all_accepted),
        "health_status": bundle.health_status,
        "power_limited": bool(any_power_limited),
        "promotion_guard_id": plan_id if multi_plan else None,
        "planned_variant_count": plan_n if multi_plan else None,
        "promotion_pending": promotion_pending,
        "asset_validations": [x.to_dict() for x in audits],
    })
    manager._root._history.append({
        "event": "validate_fleet_candidate",
        "key": repr(key),
        "reference_id": root_ref.meta.reference_id,
        "accepted": bool(all_accepted),
        "health": root_ref.health_status,
        "supported_fraction": root_ref.validation_supported_fraction,
        "assets": sorted(required),
        "promotion_pending": promotion_pending,
    })
    return FleetReferenceCommissioningDecision(
        gid, bundle.meta.bundle_id, bool(all_accepted), state, reason,
        tuple(sorted(required)), tuple(audits), bool(any_power_limited),
        promotion_guard_id=plan_id if multi_plan else None,
        planned_variant_count=plan_n if multi_plan else None,
        promotion_pending=promotion_pending,
    )


@dataclass(frozen=True)
class FleetReferenceCommissioningRobustnessDecision:
    fleet_group_id: str
    accepted: bool
    state: str
    reason: str
    planned_variant_count: int
    audited_variant_count: int
    accepted_variant_count: int
    accepted_variant_fraction: float
    blocked_variant_count: int
    bundle_ids: tuple[str, ...]
    accepted_bundle_ids: tuple[str, ...] = ()
    promotion_guard_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "fleet_group_id": self.fleet_group_id,
            "accepted": self.accepted,
            "state": self.state,
            "reason": self.reason,
            "planned_variant_count": self.planned_variant_count,
            "audited_variant_count": self.audited_variant_count,
            "accepted_variant_count": self.accepted_variant_count,
            "accepted_variant_fraction": self.accepted_variant_fraction,
            "blocked_variant_count": self.blocked_variant_count,
            "bundle_ids": list(self.bundle_ids),
            "accepted_bundle_ids": list(self.accepted_bundle_ids),
            "promotion_guard_id": self.promotion_guard_id,
        }


def robust_fleet_commissioning_from_variants(
    decisions: Iterable[FleetReferenceCommissioningDecision],
    *,
    planned_variant_count: int,
    min_audited_fraction: float = 1.0,
    min_accept_fraction: float = 1.0,
) -> FleetReferenceCommissioningRobustnessDecision:
    """Aggregate multi-asset fleet commissioning across Reference constructions.

    Defaults are intentionally strict: every predeclared Reference construction
    must be audited and every one must pass.  If the variant decisions declare
    a multi-construction promotion guard, all decisions must carry the same
    guard id and planned count.  A passing aggregate decision can then be bound
    to one accepted staged bundle with :func:`attest_fleet_reference_robustness`.
    """
    ds = tuple(decisions)
    if not ds:
        raise ValueError("at least one fleet commissioning decision is required")
    gids = {d.fleet_group_id for d in ds}
    if len(gids) != 1:
        raise ValueError(f"all fleet commissioning variants must belong to one fleet group: {sorted(gids)}")
    bundle_ids = tuple(d.bundle_id for d in ds)
    if len(set(bundle_ids)) != len(bundle_ids):
        raise ValueError("fleet Reference variant bundle_ids must be unique")
    planned = int(planned_variant_count)
    if planned <= 0 or planned < len(ds):
        raise ValueError("planned_variant_count must be positive and >= audited variants")
    for name,val in (("min_audited_fraction",min_audited_fraction),("min_accept_fraction",min_accept_fraction)):
        if not (0.0 < float(val) <= 1.0):
            raise ValueError(f"{name} must be in (0,1]")

    guard_values = [d.promotion_guard_id for d in ds]
    any_guard = any(x is not None for x in guard_values)
    guard_id: str | None = None
    if any_guard:
        if any(x is None for x in guard_values):
            raise ValueError("all Reference variants in one robustness plan must declare the same promotion_guard_id")
        guards = {str(x) for x in guard_values}
        if len(guards) != 1:
            raise ValueError(f"Reference variants belong to different promotion guards: {sorted(guards)}")
        guard_id = next(iter(guards))
        counts = {d.planned_variant_count for d in ds}
        if counts != {planned}:
            raise ValueError(
                f"variant-declared planned counts {sorted(str(x) for x in counts)} do not match aggregate planned_variant_count={planned}"
            )

    audited=len(ds); accepted=sum(d.accepted for d in ds)
    audited_fraction=audited/planned
    accept_fraction=accepted/audited
    accepted_bundle_ids=tuple(sorted(d.bundle_id for d in ds if d.accepted))
    common=dict(
        fleet_group_id=next(iter(gids)), planned_variant_count=planned,
        audited_variant_count=audited, accepted_variant_count=accepted,
        accepted_variant_fraction=float(accept_fraction),
        blocked_variant_count=audited-accepted,
        bundle_ids=tuple(sorted(bundle_ids)),
        accepted_bundle_ids=accepted_bundle_ids,
        promotion_guard_id=guard_id,
    )
    if audited_fraction + 1e-15 < float(min_audited_fraction):
        return FleetReferenceCommissioningRobustnessDecision(
            accepted=False,state="reference_uncertainty_insufficient",
            reason=f"only {audited}/{planned} predeclared fleet Reference variants were audited",
            **common,
        )
    if accept_fraction + 1e-15 < float(min_accept_fraction):
        return FleetReferenceCommissioningRobustnessDecision(
            accepted=False,state="blocked_reference_sensitive",
            reason=f"only {accepted}/{audited} audited fleet Reference variants passed all member assets",
            **common,
        )
    return FleetReferenceCommissioningRobustnessDecision(
        accepted=True,state="robustly_commissioned",
        reason=f"all required fleet Reference constructions passed ({accepted}/{audited})",
        **common,
    )


def attest_fleet_reference_robustness(
    manager: ReferenceBundleManager,
    key: Hashable,
    decision: FleetReferenceCommissioningRobustnessDecision,
) -> str:
    """Bind a passing multi-construction robustness decision to one staged bundle.

    The staged bundle must already have passed its own atomic member-sentinel
    audit under the same declared promotion guard.  A blocked/partial aggregate
    decision cannot be used as an attestation.  On success the bundle becomes
    eligible for the ordinary ``ReferenceBundleManager.promote`` call.
    """
    bundle = manager._shadow.get(key)
    if bundle is None:
        raise KeyError(f"no staged reference bundle for {key!r}")
    if bundle.promotion_guard_id is None:
        raise ValueError("staged Reference Bundle did not declare a multi-construction promotion guard")
    if not decision.accepted:
        raise RuntimeError(
            f"fleet Reference robustness decision is not promotable: {decision.state}: {decision.reason}"
        )
    if decision.promotion_guard_id != bundle.promotion_guard_id:
        raise ValueError(
            f"promotion guard mismatch: bundle={bundle.promotion_guard_id!r} decision={decision.promotion_guard_id!r}"
        )
    if decision.planned_variant_count != bundle.promotion_guard_planned_variants:
        raise ValueError(
            "planned Reference-variant count does not match the staged bundle promotion guard"
        )
    compat = bundle.meta.compatibility
    scope = None if compat is None else compat.asset_scope
    gid = None if scope is None else scope.fleet_group_id
    if gid != decision.fleet_group_id:
        raise ValueError(
            f"fleet group mismatch: bundle={gid!r} decision={decision.fleet_group_id!r}"
        )
    if bundle.meta.bundle_id not in decision.accepted_bundle_ids:
        raise RuntimeError(
            f"staged bundle {bundle.meta.bundle_id!r} was not an accepted Reference variant in the robustness decision"
        )
    if bundle.lifecycle_state != "robustness_pending" or bundle.health_status not in {"healthy", "conservative"}:
        raise RuntimeError("staged bundle has not passed its atomic member-sentinel validation")

    bundle.promotion_guard_satisfied = True
    bundle.validated_for_promotion = True
    bundle.lifecycle_state = "validated"
    manager._history.append({
        "event": "attest_fleet_reference_robustness",
        "key": repr(key),
        "bundle_id": bundle.meta.bundle_id,
        "fleet_group_id": decision.fleet_group_id,
        "promotion_guard_id": decision.promotion_guard_id,
        "planned_variant_count": decision.planned_variant_count,
        "accepted_bundle_ids": list(decision.accepted_bundle_ids),
    })
    return bundle.meta.bundle_id
