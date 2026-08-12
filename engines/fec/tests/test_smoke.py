import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fec.core.fec_core import ContactSpec,fit_reference,evidence,coordinate,make_root_reference,calibrated_p

def test_smoke():
    rng=np.random.default_rng(1)
    specs=[ContactSpec('a',('g1',)),ContactSpec('b',('g2',))]
    leaf=rng.normal(size=(100,2)); root=rng.normal(size=(100,2))
    ref=fit_reference(leaf)
    stc=coordinate(evidence(root,ref),specs,operators=('mean','mean'))
    rr=make_root_reference(stc.root_E,specs)
    st=coordinate(evidence(np.array([[0.,0.]]),ref),specs,operators=('mean','mean'))
    p=calibrated_p(rr,st.root_E,specs)[0]
    assert 0<p<=1
    assert st.C_presence==1.0
