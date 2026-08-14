from __future__ import annotations

import numpy as np
import pytest

from fec.core.fec_core import ContactSpec
from fec.extensions.evidence_provenance import (
    AssetReferenceScope, ModalityAvailability, ProviderIdentity,
    ReferenceCompatibilityContract, SourceDomain,
)
from fec.extensions.fleet_reference import (
    FleetReferenceCommissioningDecision,
    robust_fleet_commissioning_from_variants,
    validate_fleet_reference_candidate,
    attest_fleet_reference_robustness,
)
from fec.extensions.reference_bundle import ReferenceBundleManager, ReferenceBundleMeta
from fec.extensions.reference_bundle_store import save_reference_bundle_manager, load_reference_bundle_manager

SPEC=(ContactSpec('rms',('machine',)),)
AV=ModalityAvailability.from_mapping({'vibration':'observed'})
P=ProviderIdentity.build('fleet-test-provider','1')
GROUP='fleet-1'


def runtime(asset: str):
    return ReferenceCompatibilityContract.from_runtime(
        availability=AV, provider_fingerprints={P.fingerprint},
        source_domain=SourceDomain.REAL, asset_id=asset, fleet_group_id=GROUP,
    )


def staged_manager(*, seed=1):
    rng=np.random.default_rng(seed)
    mgr=ReferenceBundleManager(alpha=.05,min_sentinel_samples=200,sentinel_bootstrap_reps=300,random_state=3)
    c=ReferenceCompatibilityContract.commissioned(
        availability=AV, provider_fingerprints={P.fingerprint}, source_domain=SourceDomain.REAL,
        asset_scope=AssetReferenceScope.fleet_group(GROUP,{'A','B'}),
    )
    meta=ReferenceBundleMeta(
        bundle_id='fleet-ref', key='ctx', provenance='two trusted assets',
        dependency_set=frozenset({'healthy'}),
        leaf_data_ids=frozenset({'A-leaf','B-leaf'}),
        root_data_ids=frozenset({'A-root','B-root'}), compatibility=c,
    )
    # Balanced asset commissioning data; the manager sees the pooled Contact rows,
    # while fleet validation below keeps the independent sentinels asset-stratified.
    leaf=np.vstack([rng.normal(size=(1000,1)),rng.normal(size=(1000,1))])
    root=np.vstack([rng.normal(size=(1000,1)),rng.normal(size=(1000,1))])
    assert mgr.stage_candidate(meta,leaf,root,SPEC,trusted_claims={'healthy'})
    return mgr,rng


def test_atomic_fleet_candidate_requires_every_declared_asset_and_promotes_only_after_all_pass():
    mgr,rng=staged_manager(seed=11)
    d=validate_fleet_reference_candidate(
        mgr,'ctx',{'A':rng.normal(size=(1200,1)),'B':rng.normal(size=(1200,1))},
        sentinel_data_ids_by_asset={'A':{'A-sentinel'},'B':{'B-sentinel'}},
        runtime_compatibility_by_asset={'A':runtime('A'),'B':runtime('B')},
        min_samples=200,
    )
    assert d.accepted and d.state=='validated'
    assert {x.asset_id for x in d.asset_validations}=={'A','B'}
    assert all(x.health is not None and x.health.status=='healthy' for x in d.asset_validations)
    assert mgr._shadow['ctx'].validated_for_promotion
    mgr.promote('ctx')
    assert mgr.active_info('ctx')['compatibility']['asset_scope']['admitted_asset_ids']==['A','B']
    assert mgr.evaluate('ctx',rng.normal(size=(10,1)),runtime_compatibility=runtime('A')).state=='valid'
    assert mgr.evaluate('ctx',rng.normal(size=(10,1)),runtime_compatibility=runtime('B')).state=='valid'


def test_one_anti_conservative_member_blocks_entire_candidate_without_last_write_wins():
    mgr,rng=staged_manager(seed=12)
    d=validate_fleet_reference_candidate(
        mgr,'ctx',{'A':rng.normal(size=(1500,1)),'B':rng.normal(loc=2.0,size=(1500,1))},
        sentinel_data_ids_by_asset={'A':{'A-sentinel'},'B':{'B-sentinel'}},
        runtime_compatibility_by_asset={'A':runtime('A'),'B':runtime('B')},
        min_samples=200,
    )
    assert not d.accepted and d.state=='blocked'
    by={x.asset_id:x for x in d.asset_validations}
    assert by['A'].accepted
    assert not by['B'].accepted and by['B'].health is not None and by['B'].health.status=='anti_conservative'
    assert not mgr._shadow['ctx'].validated_for_promotion
    with pytest.raises(RuntimeError,match='promotion policy'):
        mgr.promote('ctx')


def test_fleet_candidate_rejects_missing_member_and_cross_asset_sentinel_overlap():
    mgr,rng=staged_manager(seed=13)
    d=validate_fleet_reference_candidate(
        mgr,'ctx',{'A':rng.normal(size=(500,1))},
        sentinel_data_ids_by_asset={'A':{'A-sentinel'}},
        runtime_compatibility_by_asset={'A':runtime('A')},
        min_samples=200,
    )
    assert not d.accepted and d.state=='reference_unsupported'
    assert 'B' in d.reason

    mgr,rng=staged_manager(seed=14)
    with pytest.raises(ValueError,match='overlap across assets'):
        validate_fleet_reference_candidate(
            mgr,'ctx',{'A':rng.normal(size=(500,1)),'B':rng.normal(size=(500,1))},
            sentinel_data_ids_by_asset={'A':{'same'},'B':{'same'}},
            runtime_compatibility_by_asset={'A':runtime('A'),'B':runtime('B')},
            min_samples=200,
        )


def test_fleet_candidate_persist_restart_preserves_scope_and_audit_history(tmp_path):
    mgr,rng=staged_manager(seed=15)
    d=validate_fleet_reference_candidate(
        mgr,'ctx',{'A':rng.normal(size=(1400,1)),'B':rng.normal(size=(1400,1))},
        sentinel_data_ids_by_asset={'A':{'A-sentinel'},'B':{'B-sentinel'}},
        runtime_compatibility_by_asset={'A':runtime('A'),'B':runtime('B')},
        min_samples=200,
    )
    assert d.accepted
    mgr.promote('ctx')
    path=tmp_path/'fleet-state.json'
    save_reference_bundle_manager(mgr,path)
    loaded=load_reference_bundle_manager(path)
    info=loaded.active_info('ctx')
    assert info['compatibility']['asset_scope']['kind']=='fleet_group'
    assert info['compatibility']['asset_scope']['admitted_asset_ids']==['A','B']
    assert any(h.get('event')=='validate_fleet_bundle' for h in loaded.history)


def synthetic_decision(bundle_id: str, accepted: bool) -> FleetReferenceCommissioningDecision:
    return FleetReferenceCommissioningDecision(
        fleet_group_id=GROUP,bundle_id=bundle_id,accepted=accepted,
        state='validated' if accepted else 'blocked',reason='synthetic',
        required_asset_ids=('A','B'),asset_validations=(),power_limited=False,
    )


def test_fleet_commissioning_reference_robustness_is_unanimous_by_default():
    good=[synthetic_decision(f'r{i}',True) for i in range(6)]
    r=robust_fleet_commissioning_from_variants(good,planned_variant_count=6)
    assert r.accepted and r.state=='robustly_commissioned'

    mixed=[synthetic_decision(f'r{i}',i!=2) for i in range(6)]
    r=robust_fleet_commissioning_from_variants(mixed,planned_variant_count=6)
    assert not r.accepted and r.state=='blocked_reference_sensitive'
    assert r.accepted_variant_count==5



def guarded_synthetic_decision(bundle_id: str, accepted: bool, *, guard_id: str, planned: int):
    return FleetReferenceCommissioningDecision(
        fleet_group_id=GROUP,bundle_id=bundle_id,accepted=accepted,
        state='validated_robustness_pending' if accepted else 'blocked',reason='synthetic guarded variant',
        required_asset_ids=('A','B'),asset_validations=(),power_limited=False,
        promotion_guard_id=guard_id,planned_variant_count=planned,promotion_pending=accepted,
    )


def test_multi_construction_plan_blocks_direct_promotion_when_robustness_fails():
    mgr,rng=staged_manager(seed=31)
    d1=validate_fleet_reference_candidate(
        mgr,'ctx',{'A':rng.normal(size=(1200,1)),'B':rng.normal(size=(1200,1))},
        sentinel_data_ids_by_asset={'A':{'A-sentinel'},'B':{'B-sentinel'}},
        runtime_compatibility_by_asset={'A':runtime('A'),'B':runtime('B')},
        min_samples=200,
        reference_construction_plan_id='plan-2',planned_variant_count=2,
    )
    assert d1.accepted and d1.promotion_pending
    assert d1.state=='validated_robustness_pending'
    with pytest.raises(RuntimeError,match='promotion guard'):
        mgr.promote('ctx')

    d2=guarded_synthetic_decision('fleet-ref-2',False,guard_id='plan-2',planned=2)
    robust=robust_fleet_commissioning_from_variants([d1,d2],planned_variant_count=2)
    assert not robust.accepted and robust.state=='blocked_reference_sensitive'
    with pytest.raises(RuntimeError,match='not promotable'):
        attest_fleet_reference_robustness(mgr,'ctx',robust)
    with pytest.raises(RuntimeError,match='promotion guard'):
        mgr.promote('ctx')


def test_multi_construction_plan_requires_passing_attestation_before_promotion():
    mgr,rng=staged_manager(seed=32)
    d1=validate_fleet_reference_candidate(
        mgr,'ctx',{'A':rng.normal(size=(1200,1)),'B':rng.normal(size=(1200,1))},
        sentinel_data_ids_by_asset={'A':{'A-sentinel'},'B':{'B-sentinel'}},
        runtime_compatibility_by_asset={'A':runtime('A'),'B':runtime('B')},
        min_samples=200,
        reference_construction_plan_id='plan-2',planned_variant_count=2,
    )
    assert d1.accepted and not mgr._shadow['ctx'].validated_for_promotion
    d2=guarded_synthetic_decision('fleet-ref-2',True,guard_id='plan-2',planned=2)
    robust=robust_fleet_commissioning_from_variants([d1,d2],planned_variant_count=2)
    assert robust.accepted and robust.promotion_guard_id=='plan-2'
    assert robust.accepted_bundle_ids==('fleet-ref','fleet-ref-2')
    assert attest_fleet_reference_robustness(mgr,'ctx',robust)=='fleet-ref'
    assert mgr._shadow['ctx'].validated_for_promotion
    mgr.promote('ctx')
    assert mgr.active_info('ctx')['bundle_id']=='fleet-ref'


def test_declared_multi_construction_guard_cannot_be_silently_cleared_or_changed():
    mgr,rng=staged_manager(seed=33)
    d=validate_fleet_reference_candidate(
        mgr,'ctx',{'A':rng.normal(size=(1200,1)),'B':rng.normal(size=(1200,1))},
        sentinel_data_ids_by_asset={'A':{'A-sentinel'},'B':{'B-sentinel'}},
        runtime_compatibility_by_asset={'A':runtime('A'),'B':runtime('B')},
        min_samples=200,
        reference_construction_plan_id='plan-2',planned_variant_count=2,
    )
    assert d.accepted and d.promotion_pending
    with pytest.raises(ValueError,match='already declares'):
        validate_fleet_reference_candidate(
            mgr,'ctx',{'A':rng.normal(size=(1200,1)),'B':rng.normal(size=(1200,1))},
            sentinel_data_ids_by_asset={'A':{'A-sentinel-2'},'B':{'B-sentinel-2'}},
            runtime_compatibility_by_asset={'A':runtime('A'),'B':runtime('B')},
            min_samples=200,
        )
    with pytest.raises(ValueError,match='cannot be changed'):
        validate_fleet_reference_candidate(
            mgr,'ctx',{'A':rng.normal(size=(1200,1)),'B':rng.normal(size=(1200,1))},
            sentinel_data_ids_by_asset={'A':{'A-sentinel-3'},'B':{'B-sentinel-3'}},
            runtime_compatibility_by_asset={'A':runtime('A'),'B':runtime('B')},
            min_samples=200,
            reference_construction_plan_id='different-plan',planned_variant_count=2,
        )
    with pytest.raises(RuntimeError,match='promotion guard'):
        mgr.promote('ctx')


def test_rc3_generic_bundle_validation_cannot_commission_fleet_scope():
    """Fleet scope must never be qualified by one unstratified sentinel."""
    mgr,rng=staged_manager(seed=31)
    with pytest.raises(RuntimeError,match='fleet-scope Reference Bundle cannot use generic validate_candidate'):
        mgr.validate_candidate(
            'ctx',rng.normal(size=(1200,1)),
            sentinel_data_ids={'A-only-sentinel'},
            sentinel_compatibility=runtime('A'),
            min_samples=200,
            allow_conservative=False,
        )
    assert not mgr._shadow['ctx'].fleet_commissioning_satisfied
    assert not mgr._shadow['ctx'].validated_for_promotion
    with pytest.raises(RuntimeError,match='asset-stratified fleet commissioning'):
        mgr.promote('ctx')


def test_rc3_fleet_specific_validation_sets_hard_commissioning_attestation():
    mgr,rng=staged_manager(seed=32)
    d=validate_fleet_reference_candidate(
        mgr,'ctx',{'A':rng.normal(size=(1200,1)),'B':rng.normal(size=(1200,1))},
        sentinel_data_ids_by_asset={'A':{'A-sentinel'},'B':{'B-sentinel'}},
        runtime_compatibility_by_asset={'A':runtime('A'),'B':runtime('B')},
        min_samples=200,
    )
    assert d.accepted
    assert mgr._shadow['ctx'].fleet_commissioning_satisfied
    assert mgr.promote('ctx')=='fleet-ref'
