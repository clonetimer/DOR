import numpy as np
import pytest
from isgn.contracts import AlgebraicDataset, PublicDataset
from isgn.hybrid import (
    RegimePartition, threshold_regime_partition, project_algebraic_regime,
    project_public_regime_windows, discover_regime_algebraic_relations,
)


def _alg(seed=0,n=800):
    rng=np.random.default_rng(seed); x=rng.uniform(-2,2,n); mode=np.where(np.arange(n)%2==0,'A','B')
    y=np.where(mode=='A',2.0*x+1.0,-1.0*x+0.5)+rng.normal(0,.01,n)
    A=np.column_stack([x,y]); schema={'variables':[{'name':'x','dim':'dimensionless'},{'name':'y','dim':'dimensionless'}]}
    return AlgebraicDataset(A,np.arange(n),schema),mode


def test_observed_regime_recovery_recovers_two_distinct_relations():
    tr,mt=_alg(1,1000); va,mv=_alg(2,500)
    pt=RegimePartition(mt,'synthetic observed mode','observed_control'); pv=RegimePartition(mv,'synthetic observed mode','observed_control')
    r=discover_regime_algebraic_relations(tr,va,train_partition=pt,validation_partition=pv,inputs=['x'],target='y',degree=1,max_terms=1)
    assert r.status=='HYBRID_RELATION_EVIDENCE' and r.certification_allowed is False
    a=r.regimes['A'].relation; b=r.regimes['B'].relation
    assert abs(a.coefficients['x']-2.0)<.02 and abs(a.intercept-1.0)<.02
    assert abs(b.coefficients['x']+1.0)<.02 and abs(b.intercept-.5)<.02
    assert max(a.validation_nrmse,b.validation_nrmse)<.02
    assert r.global_baseline.validation_nrmse > 0.8


def test_threshold_partition_masks_switch_neighborhood_and_projects():
    v=np.array([0,0,0,10,10,10,10,10,0,0],float)
    p=threshold_regime_partition(v,5,source='flow',source_role='observed_process',low_label='off',high_label='on',transition_margin=1)
    assert p.regimes==['off','on']
    assert p.transition_mask.tolist()==[False,False,True,True,True,False,False,True,True,True]
    ds=AlgebraicDataset(np.column_stack([np.arange(10.),np.arange(10.)]),np.arange(10),{'variables':[{'name':'x'},{'name':'y'}]})
    on=project_algebraic_regime(ds,p,regime='on')
    assert on.values[:,0].tolist()==[5.0,6.0]


def test_fault_label_partition_is_rejected_by_default():
    tr,mt=_alg(1,100); va,mv=_alg(2,100)
    pt=RegimePartition(mt,'fault class','fault_label'); pv=RegimePartition(mv,'fault class','fault_label')
    with pytest.raises(ValueError,match='not eligible'):
        discover_regime_algebraic_relations(tr,va,train_partition=pt,validation_partition=pv,inputs=['x'],target='y',degree=1,max_terms=1)


def test_degenerate_mode_is_explicitly_blocked():
    x=np.linspace(0,1,200); mode=np.where(np.arange(200)<100,'flat','vary')
    y=np.where(mode=='flat',3.0,2*x+1)
    ds=AlgebraicDataset(np.column_stack([x,y]),np.arange(200),{'variables':[{'name':'x'},{'name':'y'}]})
    p=RegimePartition(mode,'observed mode','observed_control')
    r=discover_regime_algebraic_relations(ds,ds,train_partition=p,validation_partition=p,inputs=['x'],target='y',degree=1,max_terms=1,min_target_std=.01)
    assert r.status=='PARTIAL_HYBRID_RELATION_EVIDENCE'
    assert r.regimes['flat'].status=='DEGENERATE_TARGET_VARIATION'
    assert r.regimes['flat'].relation is None
    assert r.regimes['vary'].status=='RECOVERED'


def test_public_regime_projection_extracts_only_complete_dwell_windows():
    N,T=2,13; X=np.zeros((N,T,1)); U=np.zeros((N,T-1,1))
    for k in range(N): X[k,:,0]=np.arange(T)+k; U[k,:,0]=k
    schema={'states':[{'name':'x'}],'inputs':[{'name':'u'}]}
    ds=PublicDataset(X,U,np.arange(N),schema,.1,'parent')
    labels=np.array([['A']*6+['B']*6,['A']*4+['B']*8],dtype=object)
    trans=np.zeros_like(labels,dtype=bool); trans[:,5:7]=True
    p=RegimePartition(labels,'controller','observed_control',trans)
    a=project_public_regime_windows(ds,p,regime='A',window_steps=3,stride_steps=3,min_windows=2)
    assert a.X.shape[1:]==(4,1) and a.U.shape[1:]==(3,1)
    assert a.regime=='A'
    assert all(w['end_interval']<=5 for w in a.schema['regime_projection']['parent_windows'])
