from __future__ import annotations

from pathlib import Path

import numpy as np

from fec.core.fec_core import ContactSpec
from fec.extensions.evidence_provenance import (
    ModalityAvailability,
    ProviderIdentity,
    ReferenceCompatibilityContract,
    SourceDomain,
)
from fec.extensions.reference_bundle import ReferenceBundleManager, ReferenceBundleMeta
from fec.extensions.reference_bundle_store import load_reference_bundle_manager, save_reference_bundle_manager


SPECS = (ContactSpec("pressure", ("hydraulic",)), ContactSpec("vibration", ("mechanical",)))


def _contract(provider_version="1", pressure="observed"):
    p = ProviderIdentity.build("device_contact_pack", provider_version, configuration={"contacts": 2})
    availability = ModalityAvailability.from_mapping({"pressure": pressure, "vibration": "observed"})
    return ReferenceCompatibilityContract.from_runtime(
        availability=availability,
        provider_fingerprints={p.fingerprint},
        source_domain=SourceDomain.REAL,
    )


def _commission(contract):
    rng = np.random.default_rng(21)
    mgr = ReferenceBundleManager(alpha=0.05, min_sentinel_samples=40, sentinel_bootstrap_reps=300, random_state=0)
    meta = ReferenceBundleMeta(
        "bundle-v21", "ctx", "trusted", frozenset({"asset:healthy"}),
        frozenset({"leaf"}), frozenset({"root"}), compatibility=contract,
    )
    assert mgr.stage_candidate(
        meta, rng.normal(size=(300, 2)), rng.normal(size=(300, 2)), SPECS,
        trusted_claims={"asset:healthy"},
    )
    validation = mgr.validate_candidate(
        "ctx", rng.normal(size=(300, 2)), sentinel_data_ids={"sentinel"},
        sentinel_compatibility=contract,
    )
    assert validation.accepted_for_promotion
    mgr.promote("ctx")
    return mgr, rng


def test_runtime_provider_or_modality_mismatch_returns_reference_unsupported_before_evidence_use():
    contract = _contract()
    mgr, rng = _commission(contract)
    X = rng.normal(size=(40, 2))
    ok = mgr.evaluate("ctx", X, runtime_compatibility=contract)
    assert ok.state in {"valid", "power_limited"}
    missing_contract = mgr.evaluate("ctx", X)
    assert missing_contract.state == "reference_unsupported"
    assert not missing_contract.usable_for_policy.any()
    provider_changed = mgr.evaluate("ctx", X, runtime_compatibility=_contract(provider_version="2"))
    assert provider_changed.state == "reference_unsupported"
    modality_changed = mgr.evaluate("ctx", X, runtime_compatibility=_contract(pressure="derived"))
    assert modality_changed.state == "reference_unsupported"


def test_candidate_sentinel_contract_mismatch_blocks_promotion():
    contract = _contract()
    rng = np.random.default_rng(22)
    mgr = ReferenceBundleManager(min_sentinel_samples=40, sentinel_bootstrap_reps=200)
    meta = ReferenceBundleMeta(
        "candidate", "ctx", "trusted", frozenset({"healthy"}),
        frozenset({"leaf"}), frozenset({"root"}), compatibility=contract,
    )
    assert mgr.stage_candidate(meta, rng.normal(size=(200,2)), rng.normal(size=(200,2)), SPECS, trusted_claims={"healthy"})
    v = mgr.validate_candidate(
        "ctx", rng.normal(size=(200,2)), sentinel_data_ids={"sent"},
        sentinel_compatibility=_contract(provider_version="2"),
    )
    assert not v.accepted_for_promotion
    assert "incompatible" in v.reason


def test_persistence_roundtrip_preserves_compatibility_contract(tmp_path: Path):
    contract = _contract()
    mgr, rng = _commission(contract)
    path = tmp_path / "state.json"
    save_reference_bundle_manager(mgr, path)
    loaded = load_reference_bundle_manager(path)
    assert loaded.active_info("ctx")["compatibility"] == contract.to_dict()
    X = rng.normal(size=(20,2))
    assert loaded.evaluate("ctx", X, runtime_compatibility=contract).state in {"valid", "power_limited"}
    assert loaded.evaluate("ctx", X, runtime_compatibility=_contract(provider_version="9")).state == "reference_unsupported"
