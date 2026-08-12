import numpy as np
from isgn.contracts import PublicDataset, AlgebraicDataset
from isgn.observation_gate import assess_observation
from isgn.atlas_synthesis import variable_evidence_from_public_dataset, relation_evidence_from_algebraic
from isgn.algebraic import discover_algebraic_relation_subset


def test_public_dataset_and_gate_convert_without_device_names():
    t=np.linspace(0,1,101); x=np.sin(2*np.pi*t)[None,:,None]; u=np.cos(2*np.pi*t[:-1])[None,:,None]
    ds=PublicDataset(x,u,np.array([0]),{'states':[{'name':'q','dim':'x'}],'inputs':[{'name':'cmd','role':'input'}]},.01,'demo')
    ev=variable_evidence_from_public_dataset(ds,assess_observation(ds))
    assert {e.name for e in ev}=={'q','cmd'}
    assert next(e for e in ev if e.name=='cmd').role=='input'


def test_algebraic_result_converts_to_relation_evidence():
    r=np.random.default_rng(3); a=r.normal(size=300); b=r.normal(size=300); y=2*a*b
    ds=AlgebraicDataset(np.c_[a,b,y],np.arange(300),{'variables':[{'name':'a'},{'name':'b'},{'name':'y'}]})
    res=discover_algebraic_relation_subset(ds,ds,inputs=['a','b'],target='y',max_terms=2)
    e=relation_evidence_from_algebraic(res)
    assert e.kind=='algebraic' and 'a*b' in e.terms
