import sys, types
import numpy as np
import pytest

from isgn.backends import backend_status
from isgn.simulation import simulate
from isgn.api import recover_graph, certify_completion, save_checkpoint, load_checkpoint


def test_reference_backend_roundtrip(tmp_path):
    tr,_=simulate('thermal',8,T=70,seed=1)
    va,_=simulate('thermal',5,T=70,seed=2)
    r=recover_graph(tr,va,backend='reference-sindy')
    assert r.summary.backend=='reference-sindy'
    assert set(r.summary.graph['T'])=={'T','P','Ta'}
    assert r.summary.val_nrmse < 1e-2
    c=certify_completion(r,va,coverage_floor=1e-4)
    assert c['backend']=='reference-sindy'
    p=tmp_path/'r.pt'; save_checkpoint(r,p); rr=load_checkpoint(p)
    assert rr.summary.backend=='reference-sindy'
    assert rr.model.effective_coefficients()==r.model.effective_coefficients()


def test_pysindy_missing_is_explicit():
    if backend_status()['pysindy']:
        pytest.skip('official PySINDy is installed')
    tr,_=simulate('tank',4,T=40,seed=1); va,_=simulate('tank',3,T=40,seed=2)
    with pytest.raises(RuntimeError,match='optional dependency'):
        recover_graph(tr,va,backend='pysindy')


class _FakeIdentityLibrary: pass
class _FakeSTLSQ:
    def __init__(self,**kw): self.kw=kw
class _FakeConstrainedSR3(_FakeSTLSQ): pass
class _FakeStableLinearSR3(_FakeSTLSQ): pass
class _FakeSINDy:
    def __init__(self,feature_library=None,optimizer=None): self.optimizer=optimizer
    def fit(self,x,t=None,x_dot=None,u=None,**kw):
        X=np.concatenate([np.asarray(a) for a in x],axis=0)
        U=np.concatenate([np.asarray(a) for a in u],axis=0)
        Y=np.concatenate([np.asarray(a) for a in x_dot],axis=0)
        P=np.concatenate([X,U],axis=1)
        self._coef=np.linalg.lstsq(P,Y,rcond=None)[0].T
        self._d=X.shape[1]; self._u=U.shape[1]
        return self
    def get_feature_names(self): return [f'x{i}' for i in range(self._d)]+[f'u{i}' for i in range(self._u)]
    def coefficients(self): return self._coef
    def predict(self,x,u=None): return np.concatenate([np.asarray(x),np.asarray(u)],axis=1)@self._coef.T


def test_official_adapter_contract_with_api_compatible_fake(monkeypatch):
    fake=types.SimpleNamespace(__version__='2.1.0-test',IdentityLibrary=_FakeIdentityLibrary,STLSQ=_FakeSTLSQ,
                               ConstrainedSR3=_FakeConstrainedSR3,StableLinearSR3=_FakeStableLinearSR3,SINDy=_FakeSINDy)
    monkeypatch.setitem(sys.modules,'pysindy',fake)
    tr,_=simulate('motor',12,T=80,seed=3); va,_=simulate('motor',6,T=80,seed=4)
    r=recover_graph(tr,va,backend='pysindy',backend_options={'thresholds':(0.0,)})
    assert r.summary.backend=='pysindy'
    assert r.summary.backend_metadata['official'] is True
    assert r.summary.backend_metadata['pysindy_version']=='2.1.0-test'
    assert set(r.summary.graph['i'])=={'i','w','V','TL'}  # dense OLS fake: TL may be numerical near-zero but floor is tiny
    assert r.summary.val_nrmse < 0.05
