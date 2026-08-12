"""Cross-domain BLINDED_FOR_VALIDATION example using the compact EUV T1 normal subset."""
from __future__ import annotations
from pathlib import Path
import copy, math
import numpy as np
import torch
from scipy.optimize import least_squares
from m2n import (
    Dim, KnowledgeLevel, KnowledgeSource, MechanismSlot, SlotEvidence,
    plan_neural_architecture, PeakAxisSpec, SeparablePeakIR,
    FieldAssignment, FamilyPatch, SlotBinding, run_design_session,
    ArchitectureSelectionPolicy, compile_executable_ir,
)

HERE=Path(__file__).resolve().parent
DATA=HERE.parent/'data'/'euv_le07_t1_normal_subset.npz'
I_DIM=Dim.of(mass=1,time=-3); SIGMA_DIM=Dim.of(mass=1,length=-2); LENGTH=Dim.of(length=1); DLESS=Dim()
FAMILIES={
    'irradiance_peak':('I_on_target',),
    'irradiance_target_peak':('I_on_target','target_sigma'),
    'irradiance_target_alignment_peak':('I_on_target','target_sigma','hit_offset'),
}
slot=MechanismSlot(
    id='ce_response_surface',level=KnowledgeLevel.K2,role='local conversion-efficiency response surface',
    input_dims=(I_DIM,SIGMA_DIM,LENGTH),output_dim=DLESS,finite_candidates=tuple(FAMILIES),
    hard_constraints=('ce>=0','nominal alignment optimum at zero offset'),
    knowledge_source=KnowledgeSource.BLINDED_FOR_VALIDATION,
)
plan=plan_neural_architecture((slot,),{'ce_response_surface':SlotEvidence(
    data_qualification='READY',observation_qualification='SUPPORTED',hard_mechanism_status='COMPLETE',finite_status='PENDING')})
skeleton=SeparablePeakIR(
    axes=(
        PeakAxisSpec('I_on_target',I_DIM,'log',1e14,math.log(6.0),0.5,True),
        PeakAxisSpec('target_sigma',SIGMA_DIM,'identity',1e-3,1.17,0.7,True),
        PeakAxisSpec('hit_offset',LENGTH,'identity',1e-5,0.0,5.0,False),
    ),
    active_axes=('I_on_target',),output_name='ce',output_dim=DLESS,amplitude_init=0.04,
)
binding=SlotBinding('ce_response_surface',tuple(
    FamilyPatch(f,(FieldAssignment('active_axes',axes),)) for f,axes in FAMILIES.items()
))
z=np.load(DATA,allow_pickle=False); data={k:z[k] for k in z.files}; tr=data['split']=='train'; va=data['split']=='val'; te=data['split']=='test'


def transform(mask):
    return {
        'I_on_target':np.log(data['I_on_target'][mask]/1e14),
        'target_sigma':data['target_sigma'][mask]/1e-3,
        'hit_offset':data['hit_offset'][mask]/1e-5,
    }, data['ce'][mask]


def layout(active):
    x0=[0.04,math.log(6.0),0.5]; lo=[0.005,1.0,0.05]; hi=[0.08,3.0,2.0]
    if 'target_sigma' in active: x0 += [1.17,0.7]; lo += [0.5,0.05]; hi += [2.0,2.0]
    if 'hit_offset' in active: x0 += [5.0]; lo += [0.1]; hi += [20.0]
    return np.array(x0),np.array(lo),np.array(hi)


def np_pred(p,active,x):
    q=((x['I_on_target']-p[1])/p[2])**2; k=3
    if 'target_sigma' in active: q += ((x['target_sigma']-p[k])/p[k+1])**2; k+=2
    if 'hit_offset' in active: q += (x['hit_offset']/p[k])**2
    return p[0]*np.exp(-0.5*q)


def load_model(model,active,p):
    centers={'I_on_target':p[1]}; widths={'I_on_target':p[2]}; k=3
    if 'target_sigma' in active: centers['target_sigma']=p[k]; widths['target_sigma']=p[k+1]; k+=2
    if 'hit_offset' in active: centers['hit_offset']=0.0; widths['hit_offset']=p[k]
    model.set_normalized_parameters(amplitude=p[0],centers=centers,widths=widths)


def compiled_pred(model,mask):
    kw={k:torch.as_tensor(data[k][mask],dtype=torch.float64) for k in ('I_on_target','target_sigma','hit_offset')}
    with torch.no_grad(): return model(**kw)['ce'].numpy()


def evaluator(candidate,model):
    family=dict(candidate.family_choices)['ce_response_surface']; active=FAMILIES[family]; xtr,ytr=transform(tr); xv,yv=transform(va)
    vals=[]
    for seed in (21,22,23):
        rng=np.random.default_rng(seed); x0,lo,hi=layout(active); start=np.clip(x0*(1+rng.normal(0,0.06,len(x0))),lo+1e-9,hi-1e-9)
        opt=least_squares(lambda p:np_pred(p,active,xtr)-ytr,start,bounds=(lo,hi),max_nfev=700)
        m=copy.deepcopy(model); load_model(m,active,opt.x); pred=compiled_pred(m,va)
        vals.append(float(np.sqrt(np.mean((pred-yv)**2))/(np.std(yv)+1e-15)))
    return vals

result=run_design_session(plan,skeleton,(binding,),evaluator,selection_policy=ArchitectureSelectionPolicy())
print('BLINDING NOTICE:')
for r in plan.reasons: print(' ',r)
print('result   :',result.status)
print('selected :',result.selected_candidate_id)
for k,v in result.candidate_validation_risks: print(' ',k,'median NRMSE =',float(np.median(v)))
print('test remains unopened in this quick example:',int(np.sum(te)),'rows')
