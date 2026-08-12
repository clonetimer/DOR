"""Small physics-reducer example: P=(p1-p2)q."""
import numpy as np
from isgn.contracts import AlgebraicDataset
from isgn.physics_reducer import reduce_algebraic_search_space

rng=np.random.default_rng(0)
def make(seed,n):
    r=np.random.default_rng(seed); p1=r.uniform(2e6,8e6,n); p2=r.uniform(1e6,5e6,n); q=r.uniform(.001,.01,n); P=(p1-p2)*q
    schema={'variables':[{'name':'p1','dim':'pressure'},{'name':'p2','dim':'pressure'},{'name':'q','dim':'volume_flow'},{'name':'P','dim':'power'}]}
    return AlgebraicDataset(np.column_stack([p1,p2,q,P]),np.arange(n),schema)
tr,va=make(1,1000),make(2,500)
r=reduce_algebraic_search_space(tr,va,inputs=['p1','p2','q'],target='P',max_total_order=2,max_terms=2)
print(r.to_dict())
