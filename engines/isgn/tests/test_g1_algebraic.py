import numpy as np
from isgn.contracts import AlgebraicDataset
from isgn.algebraic import discover_algebraic_relation, discover_algebraic_relation_subset


def _ds(seed,n=1000):
    r=np.random.default_rng(seed); a=r.uniform(.1,2,n); b=r.uniform(.2,3,n); y=1.7*a*b
    vals=np.column_stack([a,b,y])
    return AlgebraicDataset(vals,np.arange(n),{'variables':[{'name':'a'},{'name':'b'},{'name':'y'}]})


def test_algebraic_backends_recover_product_without_device_knowledge():
    tr=_ds(1); va=_ds(2)
    for result in [discover_algebraic_relation(tr,va,inputs=['a','b'],target='y'),
                   discover_algebraic_relation_subset(tr,va,inputs=['a','b'],target='y',max_terms=2)]:
        assert 'a*b' in result.selected_terms
        assert abs(result.coefficients['a*b']-1.7)<2e-3
        assert result.validation_nrmse<2e-3
