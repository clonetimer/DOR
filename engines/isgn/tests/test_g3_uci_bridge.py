import io, zipfile, numpy as np
from isgn.bridges.uci_hydraulic import load_uci_hydraulic


def _write_matrix(z,name,a):
    buf=io.StringIO(); np.savetxt(buf,a); z.writestr(name+'.txt',buf.getvalue())


def test_uci_bridge_resamples_by_cycle_and_prevents_cycle_leakage(tmp_path):
    p=tmp_path/'uci.zip'; n=30
    profile=np.column_stack([np.full(n,100),np.full(n,100),np.zeros(n),np.full(n,130),np.zeros(n)])
    t=np.tile(np.linspace(30,40,60),(n,1)); ts3=t; ts4=t-2
    fs2=np.repeat(np.full((n,60),4.0),10,axis=1)
    cp=.027*4*(ts3-ts4)
    with zipfile.ZipFile(p,'w') as z:
        b=io.StringIO(); np.savetxt(b,profile); z.writestr('profile.txt',b.getvalue())
        _write_matrix(z,'TS3',ts3); _write_matrix(z,'TS4',ts4); _write_matrix(z,'FS2',fs2); _write_matrix(z,'CP',cp)
    tr=load_uci_hydraulic(p,scope='cooling-power',split='train',seed=2)
    va=load_uci_hydraulic(p,scope='cooling-power',split='val',seed=2)
    te=load_uci_hydraulic(p,scope='cooling-power',split='test',seed=2)
    tc=set(tr.metadata['cycle_ids']); vc=set(va.metadata['cycle_ids']); ec=set(te.metadata['cycle_ids'])
    assert not (tc&vc or tc&ec or vc&ec)
    assert tr.schema['production_real_world'] is True
    assert tr.values.shape[1]==4
