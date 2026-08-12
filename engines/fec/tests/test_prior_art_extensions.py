import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest

from fec.core.fec_core import ContactSpec, make_root_reference
from fec.extensions.conditional_calibration import MondrianRootRegistry, CalibrationGroupUnavailable
from fec.extensions.safe_admission import FrozenReserveAdmissionGate
from fec.extensions.credibility import combine_contract_and_credibility


def specs():
    return [ContactSpec('a', ('g',)), ContactSpec('b', ('g',))]


def test_mondrian_groups_differ_and_topology_guard():
    s=specs(); A=np.ones(2); G=np.ones(2)
    r=MondrianRootRegistry(min_group_size=4)
    r.fit_group('a', np.array([0.1,0.2,0.3,0.4]), s, A=A, G=G)
    r.fit_group('b', np.array([1.0,1.1,1.2,1.3]), s, A=A, G=G)
    pa=float(r.p_value('a', np.array([0.9]), s, A=A, G=G)[0])
    pb=float(r.p_value('b', np.array([0.9]), s, A=A, G=G)[0])
    assert pa < pb
    with pytest.raises(ValueError):
        r.p_value('a', np.array([0.9]), s, A=A, G=np.array([1,0]))
    with pytest.raises(CalibrationGroupUnavailable):
        r.p_value('missing', np.array([0.9]), s, A=A, G=G)


def test_frozen_reserve_admission_enforces_dependencies_and_reserve():
    s=specs(); A=np.ones(2); G=np.ones(2)
    ref=make_root_reference(np.array([0.1,0.2,0.3,0.4]),s,A=A,G=G,dependency_set={'healthy','mode:A'})
    gate=FrozenReserveAdmissionGate(np.linspace(0.1,0.5,40),alpha=.05)
    assert gate.check(.2,trusted_claims={'healthy','mode:A'},target_reference=ref).allowed
    assert not gate.check(.2,trusted_claims={'mode:A'},target_reference=ref).allowed
    assert not gate.check(99,trusted_claims={'healthy','mode:A'},target_reference=ref).allowed


def test_credibility_does_not_overwrite_contract_a():
    a=np.array([1.0,.2]); q=np.array([.3,.9])
    a2,q2=combine_contract_and_credibility(a,q,mode='separate')
    assert np.allclose(a2,a) and np.allclose(q2,q)
    assert np.allclose(combine_contract_and_credibility(a,q,mode='product'), a*q)
