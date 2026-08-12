import numpy as np
from fec.extensions.drift_guard import audit_p_values, calibration_strategy_decision

def test_audit_flags_anti_conservative():
    p=np.r_[np.full(20,.01),np.full(80,.5)]
    r=audit_p_values(p,alpha=.05,min_samples=40)
    assert r.status=='anti_conservative'
    d=calibration_strategy_decision(sentinel_health=r.status)
    assert not d['ready'] and d['mode']=='unsupported'

def test_audit_can_choose_safe_contract():
    d=calibration_strategy_decision(sentinel_health='anti_conservative',has_external_group=True,groups_adequate=True)
    assert d['ready'] and d['mode']=='mondrian'
