"""Runnable starter. Replace synthetic data with your normal device data."""
from __future__ import annotations
import copy
import numpy as np
import torch
from m2n import run_design_session, ArchitectureSelectionPolicy
from mechanism_contract import build_contract, build_ir_and_bindings


def load_data(seed=7):
    rng=np.random.default_rng(seed)
    v=rng.uniform(-1.0,1.0,700)
    v[np.abs(v)<0.08]+=0.12
    u=rng.uniform(-0.8,0.8,700)
    a=1.35*u-0.22*v-0.48*np.sign(v)
    idx=np.arange(len(v)); rng.shuffle(idx)
    return v,u,a,idx[:500],idx[500:]


def fit_and_risk(model,v,u,a,train,val,seed):
    torch.manual_seed(seed)
    m=copy.deepcopy(model).double()
    for p in m.parameters():
        if p.requires_grad: p.data.add_(0.03*torch.randn_like(p))
    vt=torch.tensor(v,dtype=torch.float64); ut=torch.tensor(u,dtype=torch.float64); at=torch.tensor(a,dtype=torch.float64)
    state=torch.stack([torch.zeros_like(vt),vt],dim=1)
    opt=torch.optim.Adam(m.parameters(),lr=0.06)
    tri=torch.tensor(train,dtype=torch.long); vai=torch.tensor(val,dtype=torch.long)
    for _ in range(450):
        opt.zero_grad(); pred=m.rate(state[tri],u=ut[tri])[:,1]
        loss=torch.mean((pred-at[tri])**2); loss.backward(); opt.step()
    with torch.no_grad():
        pred=m.rate(state[vai],u=ut[vai])[:,1]
        return float(torch.sqrt(torch.mean((pred-at[vai])**2)))


def main():
    plan=build_contract(); skeleton,bindings=build_ir_and_bindings(); v,u,a,tr,va=load_data()
    def evaluator(candidate,model):
        return [fit_and_risk(model,v,u,a,tr,va,s) for s in (101,102,103)]
    result=run_design_session(
        plan,skeleton,bindings,evaluator,
        selection_policy=ArchitectureSelectionPolicy(practical_ratio=0.8,restart_stability=0.75),
    )
    print('planning_status =',result.planning_status)
    print('status          =',result.status)
    print('selected        =',result.selected_candidate_id)
    print('risks           =',result.to_dict()['candidate_validation_risks'])

if __name__=='__main__': main()
