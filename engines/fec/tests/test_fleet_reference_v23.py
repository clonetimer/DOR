from __future__ import annotations

import numpy as np
import pytest

from fec.core.fec_core import ContactSpec
from fec.extensions.evidence_provenance import (
    AssetReferenceScope,
    ModalityAvailability,
    ProviderIdentity,
    ReferenceCompatibilityContract,
    SourceDomain,
)
from fec.extensions.fleet_reference import audit_asset_for_fleet_reference, fleet_scope_from_admissions
from fec.extensions.reference_bundle import ReferenceBundleManager, ReferenceBundleMeta


SPEC = (ContactSpec("bearing_rms", ("bearing",)),)
AVAIL = ModalityAvailability.from_mapping({"vibration": "observed"})
PROVIDER = ProviderIdentity.build("bearing_contact", "1")


def rt(asset: str, provider=PROVIDER):
    return ReferenceCompatibilityContract.from_runtime(
        availability=AVAIL,
        provider_fingerprints={provider.fingerprint},
        source_domain=SourceDomain.REAL,
        asset_id=asset,
    )


def commissioned_a():
    expected = ReferenceCompatibilityContract.commissioned(
        availability=AVAIL,
        provider_fingerprints={PROVIDER.fingerprint},
        source_domain=SourceDomain.REAL,
        asset_scope=AssetReferenceScope.asset_local("A"),
    )
    rng = np.random.default_rng(2303)
    mgr = ReferenceBundleManager(
        alpha=0.05,
        min_sentinel_samples=100,
        sentinel_bootstrap_reps=300,
        random_state=0,
    )
    meta = ReferenceBundleMeta(
        "A-reference", "ctx", "trusted-A", frozenset({"healthy"}),
        frozenset({"A-leaf"}), frozenset({"A-root"}), compatibility=expected,
    )
    assert mgr.stage_candidate(
        meta, rng.normal(size=(1000, 1)), rng.normal(size=(1000, 1)), SPEC,
        trusted_claims={"healthy"},
    )
    v = mgr.validate_candidate(
        "ctx", rng.normal(size=(1000, 1)), sentinel_data_ids={"A-sentinel"},
        sentinel_compatibility=rt("A"),
    )
    assert v.accepted_for_promotion
    mgr.promote("ctx")
    return mgr, rng


def test_same_distribution_candidate_asset_can_pass_independent_fleet_gate():
    mgr, rng = commissioned_a()
    d = audit_asset_for_fleet_reference(
        mgr, "ctx", rng.normal(size=(1200, 1)), asset_id="B", fleet_group_id="line-1",
        sentinel_data_ids={"B-independent-normal"}, runtime_compatibility=rt("B"),
    )
    assert d.accepted
    assert d.state == "admitted"
    assert d.health is not None and d.health.status == "healthy"


def test_shifted_candidate_asset_is_blocked_as_anti_conservative():
    mgr, rng = commissioned_a()
    d = audit_asset_for_fleet_reference(
        mgr, "ctx", rng.normal(loc=2.0, size=(2000, 1)), asset_id="B", fleet_group_id="line-1",
        sentinel_data_ids={"B-shifted-normal"}, runtime_compatibility=rt("B"),
    )
    assert not d.accepted
    assert d.state == "blocked"
    assert d.health is not None
    assert d.health.status == "anti_conservative"
    assert d.health.empirical_fpr > 0.4


def test_fleet_gate_keeps_non_asset_reference_contract_hard():
    mgr, rng = commissioned_a()
    changed = ProviderIdentity.build("bearing_contact", "2")
    d = audit_asset_for_fleet_reference(
        mgr, "ctx", rng.normal(size=(500, 1)), asset_id="B", fleet_group_id="line-1",
        sentinel_data_ids={"B-normal"}, runtime_compatibility=rt("B", changed),
    )
    assert not d.accepted
    assert d.state == "reference_unsupported"
    assert d.health is None
    assert "provider" in d.reason


def test_fleet_scope_can_only_be_built_from_accepted_admissions():
    mgr, rng = commissioned_a()
    b = audit_asset_for_fleet_reference(
        mgr, "ctx", rng.normal(size=(1500, 1)), asset_id="B", fleet_group_id="line-1",
        sentinel_data_ids={"B-normal"}, runtime_compatibility=rt("B"),
    )
    assert b.accepted
    scope = fleet_scope_from_admissions(
        "line-1", [b], base_scope=AssetReferenceScope.asset_local("A")
    )
    assert scope.admitted_asset_ids == frozenset({"A", "B"})
    c = audit_asset_for_fleet_reference(
        mgr, "ctx", rng.normal(loc=2.0, size=(1500, 1)), asset_id="C", fleet_group_id="line-1",
        sentinel_data_ids={"C-normal"}, runtime_compatibility=rt("C"),
    )
    assert not c.accepted
    with pytest.raises(ValueError, match="rejected/unverified"):
        fleet_scope_from_admissions("line-1", [b, c])


def test_fleet_sentinel_must_be_disjoint_from_reference_data():
    mgr, rng = commissioned_a()
    with pytest.raises(ValueError, match="overlaps active reference data"):
        audit_asset_for_fleet_reference(
            mgr, "ctx", rng.normal(size=(500, 1)), asset_id="B", fleet_group_id="line-1",
            sentinel_data_ids={"A-leaf"}, runtime_compatibility=rt("B"),
        )


def _variant_decision(bundle_id: str, *, accepted: bool, status: str = "healthy", asset: str = "B", group: str = "line-1"):
    from fec.extensions.drift_guard import CalibrationHealth
    from fec.extensions.fleet_reference import FleetAssetAdmissionDecision
    health = CalibrationHealth(
        n=100, alpha=0.05,
        exceedances=5 if status == "healthy" else (20 if status == "anti_conservative" else 0),
        empirical_fpr=0.05 if status == "healthy" else (0.20 if status == "anti_conservative" else 0.0),
        ci_low=0.02 if status == "healthy" else (0.12 if status == "anti_conservative" else 0.0),
        ci_high=0.11 if status == "healthy" else (0.29 if status == "anti_conservative" else 0.04),
        status=status,
        action="continue" if status == "healthy" else "review",
    )
    return FleetAssetAdmissionDecision(
        asset_id=asset,
        fleet_group_id=group,
        accepted=accepted,
        state="admitted" if accepted else "blocked",
        reason="synthetic variant vote",
        supported_fraction=1.0,
        health=health,
        sentinel_data_ids=frozenset({f"{bundle_id}-sentinel"}),
        bundle_id=bundle_id,
    )


def test_reference_variant_robustness_requires_unanimity_by_default():
    from fec.extensions.fleet_reference import robust_fleet_admission_from_variants
    ds = [_variant_decision(f"ref-{i}", accepted=True) for i in range(4)]
    r = robust_fleet_admission_from_variants(ds, planned_variant_count=4)
    assert r.accepted
    assert r.state == "robustly_admitted"
    assert r.accepted_variant_fraction == 1.0


def test_reference_variant_robustness_blocks_any_anti_conservative_variant():
    from fec.extensions.fleet_reference import robust_fleet_admission_from_variants
    ds = [
        _variant_decision("ref-1", accepted=True),
        _variant_decision("ref-2", accepted=True),
        _variant_decision("ref-3", accepted=False, status="anti_conservative"),
        _variant_decision("ref-4", accepted=True),
    ]
    r = robust_fleet_admission_from_variants(ds, planned_variant_count=4)
    assert not r.accepted
    assert r.state == "blocked_reference_sensitive"
    assert r.anti_conservative_variant_count == 1


def test_reference_variant_robustness_accounts_for_predeclared_reference_failures():
    from fec.extensions.fleet_reference import robust_fleet_admission_from_variants
    ds = [_variant_decision(f"ref-{i}", accepted=True) for i in range(3)]
    r = robust_fleet_admission_from_variants(
        ds,
        planned_variant_count=6,
        min_valid_variants=3,
        min_valid_fraction=0.8,
    )
    assert not r.accepted
    assert r.state == "reference_uncertainty_insufficient"
    assert r.valid_variant_fraction == 0.5


def test_reference_variant_robustness_rejects_duplicate_votes_and_mixed_assets():
    from fec.extensions.fleet_reference import robust_fleet_admission_from_variants
    with pytest.raises(ValueError, match="unique"):
        robust_fleet_admission_from_variants([
            _variant_decision("same", accepted=True),
            _variant_decision("same", accepted=True),
            _variant_decision("other", accepted=True),
        ], planned_variant_count=3)
    with pytest.raises(ValueError, match="one asset"):
        robust_fleet_admission_from_variants([
            _variant_decision("r1", accepted=True, asset="B"),
            _variant_decision("r2", accepted=True, asset="C"),
            _variant_decision("r3", accepted=True, asset="B"),
        ], planned_variant_count=3)


def test_reference_variant_robustness_relaxed_vote_still_cannot_override_anti_conservative_default():
    from fec.extensions.fleet_reference import robust_fleet_admission_from_variants
    ds = [
        _variant_decision("r1", accepted=True),
        _variant_decision("r2", accepted=True),
        _variant_decision("r3", accepted=False, status="anti_conservative"),
        _variant_decision("r4", accepted=True),
    ]
    r = robust_fleet_admission_from_variants(ds, planned_variant_count=4, min_accept_fraction=0.75)
    assert not r.accepted
    assert r.state == "blocked_reference_sensitive"


def test_fleet_scope_from_robust_admissions_only_accepts_robust_records():
    from fec.extensions.fleet_reference import (
        robust_fleet_admission_from_variants,
        fleet_scope_from_robust_admissions,
    )
    good = robust_fleet_admission_from_variants(
        [_variant_decision(f"g{i}", accepted=True) for i in range(4)],
        planned_variant_count=4,
    )
    scope = fleet_scope_from_robust_admissions(
        "line-1", [good], base_scope=AssetReferenceScope.asset_local("A")
    )
    assert scope.admitted_asset_ids == frozenset({"A", "B"})

    bad = robust_fleet_admission_from_variants([
        _variant_decision("b1", accepted=True),
        _variant_decision("b2", accepted=True),
        _variant_decision("b3", accepted=False, status="anti_conservative"),
    ], planned_variant_count=3)
    with pytest.raises(ValueError, match="non-robust/rejected"):
        fleet_scope_from_robust_admissions(
            "line-1", [bad], base_scope=AssetReferenceScope.asset_local("A")
        )
