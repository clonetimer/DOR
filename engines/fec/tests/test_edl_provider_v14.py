import numpy as np
import pytest
pytest.importorskip("torch")
from fec.extensions.edl_provider import EvidentialMLPProvider

def test_edl_provider_emits_nonnegative_class_evidence():
    rng=np.random.default_rng(0)
    X=np.r_[rng.normal(-2,.3,(30,2)),rng.normal(2,.3,(30,2))]
    y=np.r_[np.zeros(30,int),np.ones(30,int)]
    p=EvidentialMLPProvider(('a','b'),hidden=8,seed=0).fit(X,y,epochs=40,lr=.02,annealing_step=15)
    model=EvidentialMLPProvider(('a','b'),hidden=8,seed=0)
    model.fit(X,y,epochs=40,lr=.02,annealing_step=15)
    e=model.evidence(X[:4])
    assert e.shape==(4,2) and np.all(e>=0)
    assert p.train_accuracy>.9
