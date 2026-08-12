import numpy as np
from isgn.contracts import AlgebraicDataset
from isgn.physics_reducer import dimensionally_consistent_monomials, probe_pairwise_reductions


def _ds(seed=0,n=1200):
    rng=np.random.default_rng(seed)
    p1=rng.uniform(2e6,8e6,n); p2=rng.uniform(1e6,5e6,n); q=rng.uniform(.001,.01,n)
    power=(p1-p2)*q
    A=np.column_stack([p1,p2,q,power])
    schema={'variables':[{'name':'p1','dim':'pressure'},{'name':'p2','dim':'pressure'},{'name':'q','dim':'volume_flow'},{'name':'P','dim':'power'}]}
    return AlgebraicDataset(A,np.arange(n),schema)


def test_dimensional_pruning_finds_pressure_flow_power_terms():
    ds=_ds()
    c=dimensionally_consistent_monomials(ds,inputs=['p1','p2','q'],target='P',max_total_order=2)
    expr={x.expression for x in c}
    assert 'p1*q' in expr and 'p2*q' in expr
    assert 'p1' not in expr and 'q' not in expr


def test_pairwise_reducer_recovers_pressure_difference_coordinate():
    tr=_ds(1,1000); va=_ds(2,500)
    r=probe_pairwise_reductions(tr,va,inputs=['p1','p2','q'],target='P',degree=2,max_terms=2)
    best=r[0]
    assert best.transform=='difference'
    assert best.variables==('p1','p2')
    assert len(best.reduced_terms)==1
    assert best.reduced_validation_nrmse < 1e-6
