"""Complete K2 workflow: knowledge contract -> candidate networks -> normal-data selection."""
from __future__ import annotations
import copy
import numpy as np
import torch
from m2n import (
    Dim, KnowledgeLevel, KnowledgeSource, MechanismSlot, SlotEvidence,
    plan_neural_architecture, CoordinateSpec, InputSpec, DriveSpec, ResistanceSpec,
    SecondOrderMechanicsIR, FieldAssignment, FamilyPatch, SlotBinding,
    ArchitectureSelectionPolicy, run_design_session,
)

L=Dim.of(length=1); T=Dim.of(time=1); V=L/T; A=L/(T**2)
slot=MechanismSlot(
    id='friction', level=KnowledgeLevel.K2, role='local mechanical resistance',
    input_dims=(V,), output_dim=A,
    finite_candidates=('viscous','viscous_coulomb'),
    knowledge_source=KnowledgeSource.FINITE_ENGINEERING_HYPOTHESIS,
)
plan=plan_neural_architecture((slot,),{'friction':SlotEvidence(
    data_qualification='READY',observation_qualification='SUPPORTED',
    hard_mechanism_status='COMPLETE',finite_status='PENDING')})

skeleton=SecondOrderMechanicsIR(
    coordinate=CoordinateSpec('q','v',L),
    inputs=(InputSpec('u',A),),
    drive=DriveSpec('drive','u',Dim(),init=1.0),
    resistance=ResistanceSpec('friction','viscous',viscous_gain_init=0.1,coulomb_gain_init=0.1),
    dt=0.01,
)
binding=SlotBinding('friction',(
    FamilyPatch('viscous',(FieldAssignment('resistance.law','viscous'),)),
    FamilyPatch('viscous_coulomb',(FieldAssignment('resistance.law','viscous_coulomb'),)),
))

rng=np.random.default_rng(20260811)
v=rng.uniform(-1.2,1.2,900); v[np.abs(v)<0.06]+=0.1
u=rng.uniform(-0.9,0.9,900)
a=1.4*u-0.25*v-0.55*np.sign(v)  # normal-data generator; not shown to M2N as a family label
idx=np.arange(len(v)); rng.shuffle(idx); tr=idx[:650]; va=idx[650:]
vt=torch.tensor(v,dtype=torch.float64); ut=torch.tensor(u,dtype=torch.float64); at=torch.tensor(a,dtype=torch.float64)
state=torch.stack([torch.zeros_like(vt),vt],dim=1)


def one_restart(model,seed):
    torch.manual_seed(seed); m=copy.deepcopy(model).double()
    for p in m.parameters():
        if p.requires_grad: p.data.add_(0.02*torch.randn_like(p))
    opt=torch.optim.Adam(m.parameters(),lr=0.06)
    tri=torch.as_tensor(tr); vai=torch.as_tensor(va)
    for _ in range(500):
        opt.zero_grad(); pred=m.rate(state[tri],u=ut[tri])[:,1]
        loss=torch.mean((pred-at[tri])**2); loss.backward(); opt.step()
    with torch.no_grad():
        pred=m.rate(state[vai],u=ut[vai])[:,1]
        return float(torch.sqrt(torch.mean((pred-at[vai])**2)))


def evaluator(candidate,model):
    return [one_restart(model,s) for s in (11,12,13)]

result=run_design_session(
    plan,skeleton,(binding,),evaluator,
    selection_policy=ArchitectureSelectionPolicy(practical_ratio=0.8,restart_stability=0.75),
)
print('plan status :',plan.planning_status)
print('notice      :',plan.reasons)
print('result      :',result.status)
print('selected    :',result.selected_candidate_id)
print('validation risks:')
for k,vv in result.candidate_validation_risks: print(' ',k, [f'{x:.6g}' for x in vv])
