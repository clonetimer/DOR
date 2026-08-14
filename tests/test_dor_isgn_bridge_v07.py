import numpy as np
from dor.isgn_identification_bridge import build_isgn_identification_evidence
from isgn.contracts import PublicDataset

def ds(coeff=2.0,n=6,seed=1,const_u=False):
    rng=np.random.RandomState(seed); dt=.01; T=201
    X=np.zeros((n,T,1),np.float32); U=np.zeros((n,T-1,1),np.float32)
    for k in range(n):
        u=np.zeros(T-1) if const_u else rng.normal(size=T-1); U[k,:,0]=u
        for t in range(T-1): X[k,t+1,0]=X[k,t,0]+dt*(coeff*u[t]+.001*rng.normal())
    return PublicDataset(X,U,np.arange(n),{'states':[{'name':'x'}],'inputs':[{'name':'u'}]},dt,'t','t')

def inputs(cert='COMPLETE'):
    spec={'contract':'DOR-ISGN-MECHANISM-BASELINE/0.7','certificate_status':cert,'graph':{'x':['u']},'provenance':{'source':'unit'}}
    ont={'contract':'DOR-FAULT-ONTOLOGY/0.5','ontology_id':'x','domain':'x','components':[{'component_id':'A','kind':'PHYSICAL'}],'sensors':[],'commands':[],'topology_edges':[],'faults':[{'fault_id':'A__d','target_component_id':'A','fault_mode':'parameter_drift','category':'PARAMETRIC','expected_evidence':[{'scope_type':'MECHANISM','scope_id':'x<-u','expected_state':'ABNORMAL'}]}],'provenance':{'sources':['unit']}}
    mmap={'contract':'DOR-MECHANISM-COMPONENT-MAP/0.5','bindings':[{'mechanism_id':'x<-u','component_ids':['A'],'sensor_ids':[],'provenance':'unit binding'}]}
    return spec,ont,mmap

def test_bridge_detects_supported_anchored_drift():
    spec,ont,mmap=inputs(); r=build_isgn_identification_evidence(spec,ds(2,10,1),ds(3,5,2),ont,mmap)
    assert r['status']=='SUPPORTED'; c=r['changes'][0]
    assert c['identifiable'] and c['supported'] and abs(c['delta']-1)<.04

def test_bridge_rejects_noncomplete_baseline():
    spec,ont,mmap=inputs('UNCERTIFIED'); r=build_isgn_identification_evidence(spec,ds(2,6,1),ds(3,5,2),ont,mmap)
    assert r['status']=='UNSUPPORTED' and not r['changes']

def test_bridge_rejects_invalid_window():
    spec,ont,mmap=inputs(); r=build_isgn_identification_evidence(spec,ds(2,6,1),ds(3,5,2),ont,mmap,window_evidence_validity='CONTEXT_UNSUPPORTED')
    assert r['status']=='UNSUPPORTED'

def test_bridge_fails_closed_when_isgn23_observation_gate_blocks_unexcited_fault_window():
    spec,ont,mmap=inputs(); r=build_isgn_identification_evidence(spec,ds(2,8,1),ds(3,4,2,const_u=True),ont,mmap)
    assert r['status']=='UNIDENTIFIABLE'
    assert r['changes']==[]
    assert r['observation_gate']['abnormal']['status']=='OBSERVATION_INSUFFICIENT'


def test_supported_drift_with_ambiguous_component_mapping_is_alternatives():
    """A numerically supported drift cannot be attributed to one component when the map is non-unique."""
    spec,ont,mmap=inputs()
    ont['components'].append({'component_id':'B','kind':'PHYSICAL'})
    ont['faults'].append({
      'fault_id':'B__d','target_component_id':'B','fault_mode':'parameter_drift','category':'PARAMETRIC',
      'expected_evidence':[{'scope_type':'MECHANISM','scope_id':'x<-u','expected_state':'ABNORMAL'}]
    })
    mmap['bindings'].append({'mechanism_id':'x<-u','component_ids':['B'],'sensor_ids':[],'provenance':'unit binding B'})
    rep=build_isgn_identification_evidence(spec,ds(2,10,1),ds(3,5,2),ont,mmap)
    assert rep['status']=='ALTERNATIVES'
    assert len(rep['changes'])==2
    assert all(not c['supported'] for c in rep['changes'])
