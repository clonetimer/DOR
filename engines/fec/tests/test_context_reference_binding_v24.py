from __future__ import annotations

from pathlib import Path

import json
import numpy as np
import pytest

from fec.core.fec_core import ContactSpec
from fec.extensions.evidence_provenance import (
    AssetReferenceScope,
    ModalityAvailability,
    OperatingContextIdentity,
    ProviderIdentity,
    ReferenceCompatibilityContract,
    SourceDomain,
)
from fec.extensions.reference_bundle import ReferenceBundleManager, ReferenceBundleMeta
from fec.extensions.reference_bundle_store import (
    SCHEMA_VERSION,
    load_reference_bundle_manager,
    save_reference_bundle_manager,
)


AVAIL = ModalityAvailability.from_mapping({"acoustic_score": "derived"})
PROVIDER = ProviderIdentity.build("published-score-provider", "1", configuration={"output": "anomaly_score"})
SPECS = (ContactSpec("score", ("machine",)),)
SOURCE_CTX = OperatingContextIdentity.from_mapping(
    "dcase2021.task2", {"machine_type": "valve", "section": "01", "domain": "source"}
)
TARGET_CTX = OperatingContextIdentity.from_mapping(
    "dcase2021.task2", {"machine_type": "valve", "section": "01", "domain": "target"}
)


def contract(ctx: OperatingContextIdentity | None, *, asset: str = "valve-section-01", commissioned: bool = False):
    kw = dict(
        availability=AVAIL,
        provider_fingerprints={PROVIDER.fingerprint},
        source_domain=SourceDomain.REAL,
        operating_context=ctx,
    )
    if commissioned:
        return ReferenceCompatibilityContract.commissioned(
            **kw, asset_scope=AssetReferenceScope.asset_local(asset)
        )
    return ReferenceCompatibilityContract.from_runtime(**kw, asset_id=asset)


def commission(expected: ReferenceCompatibilityContract, sentinel: ReferenceCompatibilityContract):
    rng = np.random.default_rng(2401)
    mgr = ReferenceBundleManager(alpha=0.05, min_sentinel_samples=50, sentinel_bootstrap_reps=300)
    meta = ReferenceBundleMeta(
        "bundle-v24", "valve-01-source", "trusted-normal", frozenset({"normal"}),
        frozenset({"leaf-source"}), frozenset({"root-source"}), compatibility=expected,
    )
    assert mgr.stage_candidate(
        meta, rng.normal(size=(400, 1)), rng.normal(size=(400, 1)), SPECS,
        trusted_claims={"normal"},
    )
    v = mgr.validate_candidate(
        "valve-01-source", rng.normal(size=(400, 1)), sentinel_data_ids={"sentinel-source"},
        sentinel_compatibility=sentinel,
    )
    assert v.accepted_for_promotion
    mgr.promote("valve-01-source")
    return mgr, rng


def test_operating_context_identity_is_canonical_and_order_invariant():
    a = OperatingContextIdentity.from_mapping(
        "dcase", {"section": "01", "domain": "source", "load": 1.0}
    )
    b = OperatingContextIdentity.from_mapping(
        "dcase", {"load": 1.0, "domain": "source", "section": "01"}
    )
    assert a == b
    assert a.fingerprint == b.fingerprint
    assert a.to_dict()["attributes"]["domain"] == "source"
    assert OperatingContextIdentity.from_dict(a.to_dict()) == a


def test_operating_context_identity_rejects_tampered_persisted_fingerprint():
    raw = SOURCE_CTX.to_dict()
    raw["fingerprint"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="fingerprint"):
        OperatingContextIdentity.from_dict(raw)


def test_reference_contract_exact_context_match_and_mismatch():
    expected = contract(SOURCE_CTX, commissioned=True)
    assert expected.compatible_with(contract(SOURCE_CTX))
    reasons = expected.mismatch_reasons(contract(TARGET_CTX))
    assert any("operating_context" in x for x in reasons)
    missing = expected.mismatch_reasons(contract(None))
    assert any("not provided" in x for x in missing)


def test_context_mismatch_is_blocked_before_policy_evidence_use():
    expected = contract(SOURCE_CTX, commissioned=True)
    mgr, rng = commission(expected, contract(SOURCE_CTX))
    X = rng.normal(size=(60, 1))
    ok = mgr.evaluate("valve-01-source", X, runtime_compatibility=contract(SOURCE_CTX))
    assert ok.state in {"valid", "power_limited"}
    bad = mgr.evaluate("valve-01-source", X, runtime_compatibility=contract(TARGET_CTX))
    assert bad.state == "reference_unsupported"
    assert not bad.usable_for_policy.any()
    assert "operating_context" in " ".join(bad.diagnostics["compatibility_mismatch"])


def test_context_missing_is_fail_closed_when_reference_is_bound():
    expected = contract(SOURCE_CTX, commissioned=True)
    mgr, rng = commission(expected, contract(SOURCE_CTX))
    bad = mgr.evaluate("valve-01-source", rng.normal(size=(20, 1)), runtime_compatibility=contract(None))
    assert bad.state == "reference_unsupported"
    assert not bad.usable_for_policy.any()



def test_new_context_aware_runtime_cannot_silently_use_legacy_unbound_reference():
    legacy = ReferenceCompatibilityContract.commissioned(
        availability=AVAIL,
        provider_fingerprints={PROVIDER.fingerprint},
        source_domain=SourceDomain.REAL,
        asset_scope=AssetReferenceScope.asset_local("valve-section-01"),
    )
    reasons = legacy.mismatch_reasons(contract(SOURCE_CTX))
    assert any("no operating_context binding" in x for x in reasons)
    # A truly legacy runtime can still inspect/use the legacy state deliberately.
    assert legacy.compatible_with(contract(None))

def test_store_v3_roundtrip_preserves_context_binding(tmp_path: Path):
    expected = contract(SOURCE_CTX, commissioned=True)
    mgr, rng = commission(expected, contract(SOURCE_CTX))
    path = tmp_path / "state.json"
    save_reference_bundle_manager(mgr, path)
    envelope = json.loads(path.read_text(encoding="utf-8"))
    assert envelope["schema_version"] == "fec.reference_bundle_store.v3" == SCHEMA_VERSION
    loaded = load_reference_bundle_manager(path)
    info = loaded.active_info("valve-01-source")
    assert info["compatibility"]["operating_context"]["fingerprint"] == SOURCE_CTX.fingerprint
    X = rng.normal(size=(40, 1))
    assert loaded.evaluate("valve-01-source", X, runtime_compatibility=contract(SOURCE_CTX)).state in {"valid", "power_limited"}
    assert loaded.evaluate("valve-01-source", X, runtime_compatibility=contract(TARGET_CTX)).state == "reference_unsupported"


def test_v24_store_reads_v2_state_as_legacy_context_unscoped(tmp_path: Path):
    from fec.extensions import reference_bundle_store as store

    # Deliberately emulate a v2 state: no operating_context field existed.
    legacy_expected = ReferenceCompatibilityContract.commissioned(
        availability=AVAIL,
        provider_fingerprints={PROVIDER.fingerprint},
        source_domain=SourceDomain.REAL,
        asset_scope=AssetReferenceScope.asset_local("valve-section-01"),
    )
    mgr, _ = commission(legacy_expected, contract(None))
    path = tmp_path / "legacy-v2.json"
    save_reference_bundle_manager(mgr, path)
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["schema_version"] = "fec.reference_bundle_store.v2"
    envelope["payload"]["schema_version"] = "fec.reference_bundle_store.v2"
    for entry in envelope["payload"]["bundles"]:
        for b in entry.get("retired", []):
            if b["meta"].get("compatibility") is not None:
                b["meta"]["compatibility"].pop("operating_context", None)
        if entry.get("active") and entry["active"]["meta"].get("compatibility") is not None:
            entry["active"]["meta"]["compatibility"].pop("operating_context", None)
    envelope["payload_sha256"] = store._sha256_payload(envelope["payload"])
    path.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    loaded = load_reference_bundle_manager(path)
    assert loaded.active_info("valve-01-source")["compatibility"]["operating_context"] is None


def test_cli_builds_same_context_identity_for_commissioned_and_runtime_contracts():
    from fec.cli import _reference_compatibility_from_config

    cfg = {
        "reference_contract": {
            "modalities": {"acoustic_score": "derived"},
            "provider_fingerprints": [PROVIDER.fingerprint],
            "source_domain": "real",
            "asset_scope": {
                "kind": "asset_local",
                "admitted_asset_ids": ["valve-section-01"],
                "fleet_group_id": None,
            },
            "runtime_asset_id": "valve-section-01",
            "operating_context": {
                "namespace": "dcase2021.task2",
                "attributes": {"machine_type": "valve", "section": "01", "domain": "source"},
            },
        }
    }
    expected = _reference_compatibility_from_config(cfg, commissioned=True)
    runtime = _reference_compatibility_from_config(cfg)
    assert expected is not None and runtime is not None
    assert expected.operating_context == SOURCE_CTX
    assert runtime.operating_context == SOURCE_CTX
    assert expected.compatible_with(runtime)
