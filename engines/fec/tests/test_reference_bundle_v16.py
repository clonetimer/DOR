from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from fec.core.fec_core import ContactSpec
from fec.extensions.reference_bundle import ReferenceBundleManager, ReferenceBundleMeta

CORE_SHA = "b31a166a385a4287cdac17fc05888b52270a329ec2b6d72b1d1384ad25ad206e"
SPECS = [
    ContactSpec("x", ("mechanism",)),
    ContactSpec("y", ("mechanism",)),
    ContactSpec("z", ("aux",)),
]
KEY = ("demo", "context")
D_CAL = frozenset({"trusted_device_healthy"})


def _meta(bundle_id: str) -> ReferenceBundleMeta:
    return ReferenceBundleMeta(
        bundle_id=bundle_id,
        key=KEY,
        provenance="synthetic trusted normal",
        dependency_set=D_CAL,
        leaf_data_ids=frozenset({f"{bundle_id}-leaf"}),
        root_data_ids=frozenset({f"{bundle_id}-root"}),
    )


def test_frozen_core_hash_unchanged_v16():
    p = Path(__file__).resolve().parents[1] / "fec" / "core" / "fec_core.py"
    assert hashlib.sha256(p.read_bytes()).hexdigest() == CORE_SHA


def test_bundle_meta_requires_leaf_root_run_disjointness():
    with pytest.raises(ValueError, match="leaf/root reference data must be disjoint"):
        ReferenceBundleMeta(
            "bad", KEY, "p", D_CAL,
            frozenset({"same-run"}), frozenset({"same-run"}),
        )


def test_bundle_stage_respects_dcal_and_sentinel_disjointness():
    rng = np.random.default_rng(11)
    leaf = rng.normal(size=(100, 3))
    root = rng.normal(size=(100, 3))
    sent = rng.normal(size=(120, 3))
    mgr = ReferenceBundleManager(min_sentinel_samples=40, random_state=0)
    assert not mgr.stage_candidate(_meta("b1"), leaf, root, SPECS, trusted_claims=set())
    assert mgr.stage_candidate(_meta("b1"), leaf, root, SPECS, trusted_claims=D_CAL)
    with pytest.raises(ValueError, match="sentinel data overlaps"):
        mgr.validate_candidate(KEY, sent, sentinel_data_ids={"b1-leaf"})


def test_atomic_refresh_repairs_shifted_normal_and_rollback_restores_both_levels():
    # This is a functional counterfactual, not a claim about universal drift.
    # Both leaf and root normal mechanisms shift together from mean 0 to mean 3.
    rng = np.random.default_rng(2)
    gen = lambda mu, n: rng.normal(mu, 1.0, size=(n, 3))
    old_leaf = gen(0.0, 800)
    old_root = gen(0.0, 800)
    old_sentinel = gen(0.0, 1200)
    old_heldout = gen(0.0, 1000)
    new_leaf = gen(3.0, 800)
    new_root = gen(3.0, 800)
    new_sentinel = gen(3.0, 1200)
    new_heldout = gen(3.0, 1000)

    mgr = ReferenceBundleManager(
        alpha=0.05,
        min_sentinel_samples=300,
        min_supported_fraction=1.0,
        sentinel_bootstrap_reps=500,
        random_state=0,
    )

    assert mgr.stage_candidate(_meta("bundle-v1"), old_leaf, old_root, SPECS, trusted_claims=D_CAL)
    v1 = mgr.validate_candidate(KEY, old_sentinel, sentinel_data_ids={"old-sentinel"})
    assert v1.accepted_for_promotion and v1.health is not None
    assert v1.health.status in {"healthy", "conservative"}
    assert mgr.promote(KEY) == "bundle-v1"
    assert np.mean(mgr.evaluate(KEY, old_heldout).p_values <= 0.05) < 0.10

    shifted_under_old = mgr.evaluate(KEY, new_heldout).p_values.copy()
    assert np.mean(shifted_under_old <= 0.05) > 0.90

    assert mgr.stage_candidate(_meta("bundle-v2"), new_leaf, new_root, SPECS, trusted_claims=D_CAL)
    v2 = mgr.validate_candidate(KEY, new_sentinel, sentinel_data_ids={"new-sentinel"})
    assert v2.accepted_for_promotion and v2.health is not None
    assert v2.health.status in {"healthy", "conservative"}
    assert mgr.promote(KEY) == "bundle-v2"
    shifted_under_new = mgr.evaluate(KEY, new_heldout).p_values
    assert np.mean(shifted_under_new <= 0.05) < 0.10
    assert mgr.active_info(KEY)["bundle_id"] == "bundle-v2"

    assert mgr.rollback(KEY) == "bundle-v1"
    shifted_after_rollback = mgr.evaluate(KEY, new_heldout).p_values
    assert mgr.active_info(KEY)["bundle_id"] == "bundle-v1"
    assert np.allclose(shifted_after_rollback, shifted_under_old)


def test_manifest_persists_leaf_and_root_provenance():
    rng = np.random.default_rng(4)
    mgr = ReferenceBundleManager(min_sentinel_samples=40, random_state=0)
    assert mgr.stage_candidate(
        _meta("bundle-v1"), rng.normal(size=(100, 3)), rng.normal(size=(100, 3)),
        SPECS, trusted_claims=D_CAL,
    )
    shadow = mgr.manifest()["shadow"][repr(KEY)]
    assert shadow["bundle_id"] == "bundle-v1"
    assert shadow["leaf_data_ids"] == ["bundle-v1-leaf"]
    assert shadow["root_data_ids"] == ["bundle-v1-root"]


def test_power_limited_refresh_cannot_silently_replace_healthy_active_bundle():
    rng = np.random.default_rng(2)
    mgr = ReferenceBundleManager(
        alpha=0.05,
        min_sentinel_samples=300,
        min_supported_fraction=1.0,
        sentinel_bootstrap_reps=500,
        random_state=0,
    )
    old_leaf = rng.normal(size=(800, 3))
    old_root = rng.normal(size=(800, 3))
    old_sentinel = rng.normal(size=(1200, 3))
    assert mgr.stage_candidate(_meta("active"), old_leaf, old_root, SPECS, trusted_claims=D_CAL)
    active_v = mgr.validate_candidate(KEY, old_sentinel, sentinel_data_ids={"active-sentinel"})
    assert active_v.accepted_for_promotion and active_v.health.status == "healthy"
    mgr.promote(KEY)

    new_leaf = rng.normal(size=(800, 3))
    new_root = rng.normal(size=(800, 3))
    # A valid-but-extremely-conservative sentinel deliberately exposes a power
    # regression. The root manager considers it validity-safe, but the bundle
    # manager must not silently replace a currently healthy active reference.
    center = np.median(new_leaf, axis=0)
    conservative_sentinel = np.tile(center, (1000, 1))
    assert mgr.stage_candidate(_meta("candidate"), new_leaf, new_root, SPECS, trusted_claims=D_CAL)
    v = mgr.validate_candidate(KEY, conservative_sentinel, sentinel_data_ids={"candidate-sentinel"})
    assert v.health.status == "conservative" and v.power_limited
    assert not v.accepted_for_promotion
    with pytest.raises(RuntimeError, match="bundle-level promotion policy"):
        mgr.promote(KEY)

    # An explicit operator override exists for cases where a power-limited
    # replacement is intentionally approved after separate evidence/review.
    v_override = mgr.validate_candidate(
        KEY,
        conservative_sentinel,
        sentinel_data_ids={"candidate-sentinel"},
        protect_active_power=False,
    )
    assert v_override.accepted_for_promotion


def test_bundle_passes_backend_specific_fit_kwargs():
    from fec.extensions.conformal_backend import WeightedRootBackend

    rng = np.random.default_rng(9)
    leaf = rng.normal(size=(120, 3))
    root = rng.normal(size=(120, 3))
    mgr = ReferenceBundleManager(
        backend_factory=lambda: WeightedRootBackend(min_effective_n=2),
        min_sentinel_samples=40,
    )
    assert mgr.stage_candidate(
        _meta("weighted"), leaf, root, SPECS, trusted_claims=D_CAL,
        backend_fit_kwargs={"reference_weights": np.ones(len(root))},
    )
    shadow = mgr.manifest()["shadow"][repr(KEY)]
    assert shadow["bundle_id"] == "weighted"
