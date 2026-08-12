from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fec.core.fec_core import ContactSpec
from fec.extensions.reference_bundle import ReferenceBundleManager, ReferenceBundleMeta
from fec.extensions.reference_bundle_store import (
    ReferenceBundleIntegrityError,
    load_reference_bundle_manager,
    save_reference_bundle_manager,
)


def _build_manager(seed=0):
    rng = np.random.default_rng(seed)
    specs = (ContactSpec("x", ("mechanical",)), ContactSpec("y", ("hydraulic",)))
    mgr = ReferenceBundleManager(alpha=0.05, min_sentinel_samples=20, sentinel_bootstrap_reps=200, random_state=3)
    X_leaf = rng.normal(0, 1, (160, 2))
    X_root = rng.normal(0, 1, (160, 2))
    X_sent = rng.normal(0, 1, (120, 2))
    meta = ReferenceBundleMeta(
        "bundle-a", "ctx", "trusted-commissioning", frozenset({"machine_healthy"}),
        frozenset({"leaf-a"}), frozenset({"root-a"}),
    )
    assert mgr.stage_candidate(meta, X_leaf, X_root, specs, trusted_claims={"machine_healthy"})
    v = mgr.validate_candidate("ctx", X_sent, sentinel_data_ids={"sent-a"}, allow_conservative=True)
    assert v.accepted_for_promotion
    mgr.promote("ctx")
    return mgr, specs, rng


def _promote_second(mgr, specs, rng):
    X_leaf = rng.normal(0.35, 1, (180, 2))
    X_root = rng.normal(0.35, 1, (180, 2))
    X_sent = rng.normal(0.35, 1, (140, 2))
    meta = ReferenceBundleMeta(
        "bundle-b", "ctx", "trusted-refresh", frozenset({"machine_healthy"}),
        frozenset({"leaf-b"}), frozenset({"root-b"}),
    )
    assert mgr.stage_candidate(meta, X_leaf, X_root, specs, trusted_claims={"machine_healthy"})
    v = mgr.validate_candidate(
        "ctx", X_sent, sentinel_data_ids={"sent-b"}, allow_conservative=True, protect_active_power=False
    )
    assert v.accepted_for_promotion
    mgr.promote("ctx")


def test_restart_roundtrip_preserves_inference(tmp_path: Path):
    mgr, specs, rng = _build_manager()
    X = rng.normal(0.2, 1, (50, 2))
    before = mgr.evaluate("ctx", X).p_values.copy()
    path = tmp_path / "fec_reference_state.json"
    digest = save_reference_bundle_manager(mgr, path)
    assert len(digest) == 64
    loaded = load_reference_bundle_manager(path)
    after = loaded.evaluate("ctx", X).p_values
    np.testing.assert_array_equal(after, before)
    assert loaded.active_info("ctx")["bundle_id"] == "bundle-a"


def test_restart_preserves_rollback_chain(tmp_path: Path):
    mgr, specs, rng = _build_manager()
    X = rng.normal(0.0, 1, (60, 2))
    p_a = mgr.evaluate("ctx", X).p_values.copy()
    _promote_second(mgr, specs, rng)
    assert mgr.active_info("ctx")["bundle_id"] == "bundle-b"
    path = tmp_path / "fec_reference_state.json"
    save_reference_bundle_manager(mgr, path)
    loaded = load_reference_bundle_manager(path)
    assert loaded.active_info("ctx")["bundle_id"] == "bundle-b"
    rolled = loaded.rollback("ctx")
    assert rolled == "bundle-a"
    np.testing.assert_array_equal(loaded.evaluate("ctx", X).p_values, p_a)


def test_tampering_is_rejected(tmp_path: Path):
    mgr, _, _ = _build_manager()
    path = tmp_path / "fec_reference_state.json"
    save_reference_bundle_manager(mgr, path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    obj["payload"]["manager_settings"]["alpha"] = 0.5
    path.write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(ReferenceBundleIntegrityError):
        load_reference_bundle_manager(path)


def test_shadow_candidate_is_not_persisted(tmp_path: Path):
    mgr, specs, rng = _build_manager()
    meta = ReferenceBundleMeta(
        "bundle-shadow", "ctx2", "trusted", frozenset({"ok"}), frozenset({"l2"}), frozenset({"r2"})
    )
    assert mgr.stage_candidate(meta, rng.normal(size=(80,2)), rng.normal(size=(80,2)), specs, trusted_claims={"ok"})
    path = tmp_path / "fec_reference_state.json"
    save_reference_bundle_manager(mgr, path)
    loaded = load_reference_bundle_manager(path)
    assert loaded.active_info("ctx2") is None
    assert "ctx2" not in loaded._shadow
