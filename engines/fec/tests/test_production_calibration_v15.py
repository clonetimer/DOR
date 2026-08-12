import numpy as np
import pytest

from fec.core.fec_core import ContactSpec
from fec.extensions.conformal_backend import EmpiricalRootBackend, LocalizedRootBackend
from fec.extensions.production_calibration import CalibrationReferenceMeta, ProductionCalibrationManager

SPECS = [ContactSpec('a', ('x',)), ContactSpec('b', ('y',))]
A = np.ones(2); G = np.ones(2)


def _emp(scores):
    return EmpiricalRootBackend().fit(np.asarray(scores,float), SPECS, A=A, G=G)


def test_missing_reference_abstains():
    m=ProductionCalibrationManager(min_sentinel_samples=5)
    r=m.evaluate('ctx', [1,2], SPECS, A=A, G=G)
    assert r.state=='reference_unsupported' and not r.usable_for_policy.any()


def test_dependency_contract_blocks_candidate():
    m=ProductionCalibrationManager(min_sentinel_samples=5)
    meta=CalibrationReferenceMeta('r2','ctx','maintenance',frozenset({'pump','valve'}),reference_data_id='cand')
    assert not m.stage_candidate(meta,_emp(np.linspace(0,1,20)),SPECS,trusted_claims={'pump'},A=A,G=G)


def test_shadow_validation_then_atomic_promotion():
    m=ProductionCalibrationManager(min_sentinel_samples=10,min_supported_fraction=.8)
    m.install_active(CalibrationReferenceMeta('r1','ctx','commissioning'),_emp(np.linspace(0,1,50)),SPECS,A=A,G=G,health_status='healthy')
    meta=CalibrationReferenceMeta('r2','ctx','maintenance',frozenset({'healthy'}),reference_data_id='cand')
    assert m.stage_candidate(meta,_emp(np.linspace(1,2,50)),SPECS,trusted_claims={'healthy'},A=A,G=G)
    # Sentinel is compatible with candidate and therefore should pass.
    v=m.validate_candidate('ctx',np.linspace(1.05,1.95,30),SPECS,A=A,G=G,min_samples=10,sentinel_id='sent')
    assert v.accepted_for_promotion
    assert m.promote('ctx')=='r2'
    assert m.active_info('ctx')['reference_id']=='r2'


def test_anti_conservative_active_reference_is_blocked():
    m=ProductionCalibrationManager(min_sentinel_samples=20)
    m.install_active(CalibrationReferenceMeta('r1','ctx','commissioning'),_emp(np.linspace(0,1,50)),SPECS,A=A,G=G)
    h=m.audit_active('ctx',np.linspace(2,3,40),SPECS,A=A,G=G,min_samples=20)
    assert h.status=='anti_conservative'
    r=m.evaluate('ctx',[0.5],SPECS,A=A,G=G)
    assert r.state=='calibration_blocked' and not r.usable_for_policy.any()


def test_topology_mismatch_is_unsupported_not_recalibrated_silently():
    m=ProductionCalibrationManager(min_sentinel_samples=5)
    m.install_active(CalibrationReferenceMeta('r1','ctx','commissioning'),_emp(np.linspace(0,1,50)),SPECS,A=A,G=G,health_status='healthy')
    r=m.evaluate('ctx',[0.5],SPECS,A=np.array([1.0,.1]),G=G)
    assert r.state=='topology_unsupported'


def test_localized_backend_low_ess_is_not_policy_usable():
    rng=np.random.default_rng(0)
    ref_s=rng.normal(size=80); X=rng.normal(size=(80,1))
    b=LocalizedRootBackend(bandwidth=.15,min_effective_n=20,random_state=0).fit(ref_s,X,SPECS,A=A,G=G)
    m=ProductionCalibrationManager(min_sentinel_samples=5,min_supported_fraction=.8)
    m.install_active(CalibrationReferenceMeta('r1','ctx','commissioning'),b,SPECS,A=A,G=G)
    q=m.evaluate('ctx',[0.0],SPECS,A=A,G=G,backend_kwargs={'covariates':np.array([[20.0]])})
    assert not q.supported[0] and not q.usable_for_policy[0]


def test_unverified_active_reference_cannot_drive_policy():
    m=ProductionCalibrationManager(min_sentinel_samples=5)
    m.install_active(CalibrationReferenceMeta('r0','ctx','commissioning'),_emp(np.linspace(0,1,50)),SPECS,A=A,G=G)
    r=m.evaluate('ctx',[0.5],SPECS,A=A,G=G)
    assert r.state=='calibration_unverified' and not r.usable_for_policy.any()


def test_serial_sentinel_block_audit_supported():
    from fec.extensions.drift_guard import audit_p_value_series
    p=np.r_[np.full(20,.01),np.full(20,.5)]
    h=audit_p_value_series(p,alpha=.05,min_samples=20,block_size=4,n_bootstrap=300,random_state=0)
    assert h.n==40 and 0<=h.ci_low<=h.ci_high<=1


def test_candidate_cannot_validate_on_its_own_data():
    m=ProductionCalibrationManager(min_sentinel_samples=5)
    meta=CalibrationReferenceMeta('r2','ctx','maintenance',frozenset({'healthy'}),reference_data_id='same')
    assert m.stage_candidate(meta,_emp(np.linspace(0,1,30)),SPECS,trusted_claims={'healthy'},A=A,G=G)
    v=m.validate_candidate('ctx',np.linspace(0,1,20),SPECS,A=A,G=G,sentinel_id='same')
    assert not v.accepted_for_promotion and 'disjoint' in v.reason

def test_promotion_can_rollback():
    m=ProductionCalibrationManager(min_sentinel_samples=10)
    m.install_active(CalibrationReferenceMeta('r1','ctx','commissioning'),_emp(np.linspace(0,1,50)),SPECS,A=A,G=G,health_status='healthy')
    meta=CalibrationReferenceMeta('r2','ctx','maintenance',frozenset({'healthy'}),reference_data_id='cand')
    assert m.stage_candidate(meta,_emp(np.linspace(1,2,50)),SPECS,trusted_claims={'healthy'},A=A,G=G)
    v=m.validate_candidate('ctx',np.linspace(1.05,1.95,30),SPECS,A=A,G=G,min_samples=10,sentinel_id='sent')
    assert v.accepted_for_promotion
    m.promote('ctx')
    assert m.rollback('ctx')=='r1'
