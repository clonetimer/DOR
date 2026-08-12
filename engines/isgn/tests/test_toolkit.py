import numpy as np
from isgn.io import from_arrays, save_npz, load_npz
from isgn.simulation import simulate
from isgn.api import recover_graph, save_checkpoint, load_checkpoint, fit_normal_phm, extract_mechanism_features

def test_npz_roundtrip(tmp_path):
    ds,_=simulate('thermal',2,T=20,seed=1)
    p=tmp_path/'d.npz'; save_npz(ds,p); d2=load_npz(p)
    assert d2.X.shape==ds.X.shape and d2.U.shape==ds.U.shape
    assert d2.schema==ds.schema

def test_recovery_checkpoint_and_phm(tmp_path):
    tr,_=simulate('tank',6,T=50,seed=1); va,_=simulate('tank',4,T=50,seed=2)
    r=recover_graph(tr,va,seed=1,epochs=60,refit_epochs=40)
    p=tmp_path/'m.pt'; save_checkpoint(r,p); r2=load_checkpoint(p)
    assert r2.summary.graph.keys()==r.summary.graph.keys()
    nom=fit_normal_phm(tr,r.summary.graph); f=extract_mechanism_features(va,nom)
    assert f.shape[0]==len(va.X) and np.isfinite(f).all()
