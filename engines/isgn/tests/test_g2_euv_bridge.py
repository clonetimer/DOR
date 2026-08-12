import numpy as np
from isgn.bridges.euv import load_euv_scope


def test_euv_bridge_is_algebraic_and_marks_simulator_not_real(tmp_path):
    names=np.array(['fired','ce','E_in','E_IB','E_IF','meas_E'],dtype=object)
    n=40; X=np.zeros((n,len(names))); X[:,0]=1; X[:,1]=.2; X[:,2]=np.arange(n)+1; X[:,3]=X[:,1]*X[:,2]; X[:,4]=.5*X[:,3]; X[:,5]=X[:,4]
    split=np.array(['train']*20+['val']*10+['test']*10,dtype=object); normal=np.ones(n,int)
    p=tmp_path/'shots.npz'; np.savez(p,X=X,feature_names=names,split=split,is_normal=normal)
    ds=load_euv_scope(p,scope='plasma-energy',split='train')
    assert ds.variable_names==['ce','E_in','E_IB']
    assert ds.schema['production_real_world'] is False
