import numpy as np
import pytest
from fec.core.fec_core import ContactSpec
from fec.extensions.conformal_backend import EmpiricalRootBackend, LocalizedRootBackend, WeightedRootBackend

SPECS=[ContactSpec('x',('b',),'x')]

def test_empirical_backend_topology_guard():
    b=EmpiricalRootBackend().fit(np.arange(20.),SPECS,A=np.array([1.]),G=np.array([1.]))
    assert np.all((b.p_values([2.,10.],SPECS,A=np.array([1.]),G=np.array([1.])).p_values>=0))
    with pytest.raises(ValueError):
        b.p_values([2.],SPECS,A=np.array([.5]),G=np.array([1.]))

def test_localized_backend_valid_shapes_and_ess():
    rng=np.random.default_rng(1); x=rng.normal(size=(100,1)); s=np.abs(rng.normal(size=100))
    b=LocalizedRootBackend(random_state=4,min_effective_n=5).fit(s,x,SPECS)
    r=b.p_values([.2,2.],[[0.],[.5]],SPECS,random_state=5)
    assert r.p_values.shape==(2,)
    assert np.all((r.p_values>=0)&(r.p_values<=1))
    assert len(r.diagnostics['effective_n'])==2

def test_weighted_backend_reports_low_ess():
    r=np.arange(50,dtype=float); w=np.r_[1.,np.zeros(49)]
    b=WeightedRootBackend(min_effective_n=5).fit(r,w,SPECS)
    out=b.p_values([10.],SPECS)
    assert not out.supported[0]
