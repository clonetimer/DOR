from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fec.core.fec_core import ContactSpec
from fec.extensions.evidence_provenance import (
    AssetReferenceScope,
    AssetScopeKind,
    EvidenceProvenance,
    ModalityAvailability,
    ProviderIdentity,
    ReferenceCompatibilityContract,
    SourceDomain,
    SourceKind,
    resolve_evidence_availability,
)
from fec.extensions.reference_bundle import ReferenceBundleManager, ReferenceBundleMeta
from fec.extensions.reference_bundle_store import load_reference_bundle_manager, save_reference_bundle_manager


AVAIL = ModalityAvailability.from_mapping({"vibration": "observed"})
PROVIDER = ProviderIdentity.build("bearing_contact_pack", "1", configuration={"contact": "rms"})
SPECS = (ContactSpec("vibration_rms", ("bearing",)),)


def expected_local(asset_id: str) -> ReferenceCompatibilityContract:
    return ReferenceCompatibilityContract.commissioned(
        availability=AVAIL,
        provider_fingerprints={PROVIDER.fingerprint},
        source_domain=SourceDomain.REAL,
        asset_scope=AssetReferenceScope.asset_local(asset_id),
    )


def runtime(asset_id: str | None, *, fleet_group_id: str | None = None) -> ReferenceCompatibilityContract:
    return ReferenceCompatibilityContract.from_runtime(
        availability=AVAIL,
        provider_fingerprints={PROVIDER.fingerprint},
        source_domain=SourceDomain.REAL,
        asset_id=asset_id,
        fleet_group_id=fleet_group_id,
    )


def commission(expected: ReferenceCompatibilityContract, sentinel: ReferenceCompatibilityContract):
    rng = np.random.default_rng(2301)
    mgr = ReferenceBundleManager(
        alpha=0.05,
        min_sentinel_samples=50,
        sentinel_bootstrap_reps=300,
        random_state=0,
    )
    meta = ReferenceBundleMeta(
        "bundle-v23",
        "ctx",
        "trusted-normal",
        frozenset({"bearing:healthy"}),
        frozenset({"leaf-A"}),
        frozenset({"root-A"}),
        compatibility=expected,
    )
    assert mgr.stage_candidate(
        meta,
        rng.normal(size=(500, 1)),
        rng.normal(size=(500, 1)),
        SPECS,
        trusted_claims={"bearing:healthy"},
    )
    validation = mgr.validate_candidate(
        "ctx",
        rng.normal(size=(500, 1)),
        sentinel_data_ids={"sent-A"},
        sentinel_compatibility=sentinel,
    )
    assert validation.accepted_for_promotion
    mgr.promote("ctx")
    return mgr, rng


def test_asset_local_scope_accepts_only_commissioned_asset():
    scope = AssetReferenceScope.asset_local("bearing-A")
    assert scope.kind is AssetScopeKind.ASSET_LOCAL
    assert scope.mismatch_reasons(asset_id="bearing-A") == ()
    assert "not admitted" in " ".join(scope.mismatch_reasons(asset_id="bearing-B"))
    assert "not provided" in " ".join(scope.mismatch_reasons(asset_id=None))


def test_fleet_group_is_explicit_membership_not_schema_equivalence():
    scope = AssetReferenceScope.fleet_group("line-7", ["A", "B"])
    assert scope.mismatch_reasons(asset_id="A", fleet_group_id="line-7") == ()
    assert scope.mismatch_reasons(asset_id="B", fleet_group_id="line-7") == ()
    assert "not admitted" in " ".join(scope.mismatch_reasons(asset_id="C", fleet_group_id="line-7"))
    assert "fleet_group_id" in " ".join(scope.mismatch_reasons(asset_id="A", fleet_group_id="line-8"))


def test_asset_scope_roundtrip_and_runtime_identity_is_not_persisted():
    expected = expected_local("bearing-A")
    raw = expected.to_dict()
    restored = ReferenceCompatibilityContract.from_dict(raw)
    assert restored == expected
    rt = runtime("bearing-A")
    assert "runtime_asset_id" not in rt.to_dict()
    assert rt.to_runtime_dict()["runtime_asset_id"] == "bearing-A"
    assert rt.runtime_asset_id == "bearing-A"


def test_reference_bundle_rejects_cross_asset_before_evidence_use():
    expected = expected_local("bearing-A")
    mgr, rng = commission(expected, runtime("bearing-A"))
    X = rng.normal(size=(40, 1))
    ok = mgr.evaluate("ctx", X, runtime_compatibility=runtime("bearing-A"))
    assert ok.state in {"valid", "power_limited"}
    bad = mgr.evaluate("ctx", X, runtime_compatibility=runtime("bearing-B"))
    assert bad.state == "reference_unsupported"
    assert not bad.usable_for_policy.any()
    assert "asset_id" in " ".join(bad.diagnostics["compatibility_mismatch"])


def test_candidate_validation_rejects_sentinel_from_wrong_asset():
    expected = expected_local("bearing-A")
    rng = np.random.default_rng(2302)
    mgr = ReferenceBundleManager(min_sentinel_samples=40, sentinel_bootstrap_reps=200)
    meta = ReferenceBundleMeta(
        "candidate-v23", "ctx", "trusted", frozenset({"healthy"}),
        frozenset({"leaf-A"}), frozenset({"root-A"}), compatibility=expected,
    )
    assert mgr.stage_candidate(
        meta, rng.normal(size=(250, 1)), rng.normal(size=(250, 1)), SPECS,
        trusted_claims={"healthy"},
    )
    v = mgr.validate_candidate(
        "ctx", rng.normal(size=(250, 1)), sentinel_data_ids={"sent-B"},
        sentinel_compatibility=runtime("bearing-B"),
    )
    assert not v.accepted_for_promotion
    assert "incompatible" in v.reason


def test_persistence_restart_preserves_asset_scope(tmp_path: Path):
    expected = expected_local("bearing-A")
    mgr, rng = commission(expected, runtime("bearing-A"))
    path = tmp_path / "state.json"
    save_reference_bundle_manager(mgr, path)
    loaded = load_reference_bundle_manager(path)
    info = loaded.active_info("ctx")
    assert info["compatibility"]["asset_scope"]["admitted_asset_ids"] == ["bearing-A"]
    X = rng.normal(size=(30, 1))
    assert loaded.evaluate("ctx", X, runtime_compatibility=runtime("bearing-A")).state in {"valid", "power_limited"}
    assert loaded.evaluate("ctx", X, runtime_compatibility=runtime("bearing-B")).state == "reference_unsupported"


def test_legacy_unscoped_contract_remains_backward_compatible():
    expected = ReferenceCompatibilityContract.commissioned(
        availability=AVAIL,
        provider_fingerprints={PROVIDER.fingerprint},
        source_domain=SourceDomain.REAL,
    )
    assert expected.compatible_with(runtime("bearing-A"))
    assert expected.compatible_with(runtime("bearing-B"))


def test_lineage_runtime_contract_derives_asset_and_rejects_explicit_mismatch():
    p = EvidenceProvenance(
        "sensor:vib", SourceKind.OBSERVED, "vibration", "bearing", "bearing-A", "run-1",
        PROVIDER, SourceDomain.REAL,
    )
    resolution = resolve_evidence_availability((p,), {"vibration": "observed"})
    rt = resolution.reference_compatibility(
        {"vibration": "sensor:vib"}, source_domain=SourceDomain.REAL
    )
    assert rt.runtime_asset_id == "bearing-A"
    with pytest.raises(ValueError, match="does not match selected-source asset_id"):
        resolution.reference_compatibility(
            {"vibration": "sensor:vib"}, source_domain=SourceDomain.REAL, asset_id="bearing-B"
        )


def test_lineage_selection_cannot_mix_sources_from_different_assets():
    a = EvidenceProvenance(
        "sensor:a", SourceKind.OBSERVED, "vibration", "bearing", "A", "run-a",
        ProviderIdentity.build("a", "1"), SourceDomain.REAL,
    )
    b = EvidenceProvenance(
        "sensor:b", SourceKind.OBSERVED, "current", "motor", "B", "run-b",
        ProviderIdentity.build("b", "1"), SourceDomain.REAL,
    )
    resolution = resolve_evidence_availability((a, b), {"vibration": "observed", "current": "observed"})
    with pytest.raises(ValueError, match="span multiple asset_ids"):
        resolution.reference_compatibility(
            {"vibration": "sensor:a", "current": "sensor:b"}, source_domain=SourceDomain.REAL
        )


def test_cli_config_builds_separate_commissioned_scope_and_runtime_identity():
    from fec.cli import _reference_compatibility_from_config

    cfg = {
        "reference_contract": {
            "modalities": {"vibration": "observed"},
            "provider_fingerprints": [PROVIDER.fingerprint],
            "source_domain": "real",
            "asset_scope": {
                "kind": "asset_local",
                "admitted_asset_ids": ["bearing-A"],
                "fleet_group_id": None,
            },
            "runtime_asset_id": "bearing-A",
        }
    }
    expected = _reference_compatibility_from_config(cfg, commissioned=True)
    actual = _reference_compatibility_from_config(cfg)
    assert expected is not None and actual is not None
    assert expected.asset_scope == AssetReferenceScope.asset_local("bearing-A")
    assert expected.runtime_asset_id is None
    assert actual.asset_scope is None
    assert actual.runtime_asset_id == "bearing-A"
    assert expected.compatible_with(actual)


def test_v23_store_can_load_legacy_v1_unscoped_state(tmp_path: Path):
    import json
    from fec.extensions import reference_bundle_store as store

    legacy = ReferenceCompatibilityContract.commissioned(
        availability=AVAIL,
        provider_fingerprints={PROVIDER.fingerprint},
        source_domain=SourceDomain.REAL,
    )
    mgr, _ = commission(legacy, runtime("bearing-A"))
    path = tmp_path / "legacy.json"
    save_reference_bundle_manager(mgr, path)
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["schema_version"] = "fec.reference_bundle_store.v1"
    envelope["payload"]["schema_version"] = "fec.reference_bundle_store.v1"
    envelope["payload_sha256"] = store._sha256_payload(envelope["payload"])
    path.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    loaded = load_reference_bundle_manager(path)
    assert loaded.active_info("ctx")["compatibility"]["asset_scope"] is None
