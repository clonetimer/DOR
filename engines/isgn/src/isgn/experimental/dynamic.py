#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import math, numpy as np, torch
from ..contracts import PublicDataset

@dataclass(frozen=True)
class DynamicChoice:
    op: str
    params: Dict[str,float]
    score: float = float('nan')


def transform_numpy(u: np.ndarray, dt: float, choice: DynamicChoice) -> np.ndarray:
    """Transform one command channel [N,T] into its latent physical actuation."""
    x=np.asarray(u,dtype=np.float64); N,T=x.shape; op=choice.op
    if op=='Identity': return x.copy()
    if op=='FirstOrderLag':
        tau=float(choice.params['tau']); a=math.exp(-dt/tau); z=np.empty_like(x); z[:,0]=x[:,0]
        for t in range(T-1): z[:,t+1]=a*z[:,t]+(1-a)*x[:,t]
        return z
    if op=='Delay':
        d=int(round(choice.params['steps'])); z=np.empty_like(x)
        for t in range(T): z[:,t]=x[:,max(0,t-d)]
        return z
    if op=='SecondOrderOscillator':
        wn=float(choice.params['wn']); ze=float(choice.params['zeta']); z=np.empty_like(x); v=np.zeros(N); q=x[:,0].copy(); z[:,0]=q
        def rhs(q,v,cmd): return v, wn*wn*(cmd-q)-2*ze*wn*v
        for t in range(T-1):
            cmd=x[:,t]
            k1q,k1v=rhs(q,v,cmd); k2q,k2v=rhs(q+.5*dt*k1q,v+.5*dt*k1v,cmd)
            k3q,k3v=rhs(q+.5*dt*k2q,v+.5*dt*k2v,cmd); k4q,k4v=rhs(q+dt*k3q,v+dt*k3v,cmd)
            q=q+dt*(k1q+2*k2q+2*k3q+k4q)/6; v=v+dt*(k1v+2*k2v+2*k3v+k4v)/6; z[:,t+1]=q
        return z
    raise ValueError(op)


def transformed_public(public: PublicDataset, choices: Dict[int,DynamicChoice]) -> PublicDataset:
    U=np.asarray(public.U,dtype=np.float64).copy()
    # U has T-1 command samples; the transform is defined on exactly those interval samples.
    for idx,ch in choices.items(): U[:,:,idx]=transform_numpy(U[:,:,idx],public.dt,ch)
    return PublicDataset(X=public.X.copy(),U=U.astype(np.float32),trajectory_id=public.trajectory_id.copy(),schema=public.schema,dt=public.dt,regime=public.regime,variant=public.variant)


def _discrete_ls_score(train:PublicDataset,val:PublicDataset,input_idx:int,choice:DynamicChoice)->float:
    tr=transformed_public(train,{input_idx:choice}); va=transformed_public(val,{input_idx:choice})
    # Fit each state equation in sampled increment form; all states and all inputs are candidate sources.
    Atr=np.concatenate([tr.X[:,:-1,:],tr.U],axis=-1).reshape(-1,tr.X.shape[-1]+tr.U.shape[-1]).astype(np.float64)
    Ava=np.concatenate([va.X[:,:-1,:],va.U],axis=-1).reshape(-1,va.X.shape[-1]+va.U.shape[-1]).astype(np.float64)
    dtr=(tr.X[:,1:,:]-tr.X[:,:-1,:]).reshape(-1,tr.X.shape[-1]).astype(np.float64)
    dva=(va.X[:,1:,:]-va.X[:,:-1,:]).reshape(-1,va.X.shape[-1]).astype(np.float64)
    # add intercept for fairness; it is not part of final graph, only evidence scoring.
    A1=np.concatenate([Atr,np.ones((len(Atr),1))],axis=1); V1=np.concatenate([Ava,np.ones((len(Ava),1))],axis=1)
    coef=np.linalg.lstsq(A1,dtr,rcond=None)[0]
    pred=V1@coef
    scale=np.std(dva,axis=0); scale[scale<1e-8]=1.0
    return float(np.mean(np.sqrt(np.mean((pred-dva)**2,axis=0))/scale))


def candidate_bank():
    out=[DynamicChoice('Identity',{})]
    out += [DynamicChoice('FirstOrderLag',{'tau':x}) for x in (0.2,0.4,0.6,0.8,1.2)]
    out += [DynamicChoice('Delay',{'steps':float(x)}) for x in (1,2,3,4,5,6)]
    out += [DynamicChoice('SecondOrderOscillator',{'wn':wn,'zeta':z}) for wn in (0.8,1.1,1.4,1.8) for z in (0.2,0.35,0.6)]
    return out


def select_dynamic(train:PublicDataset,val:PublicDataset,input_idx:int=0)->Tuple[DynamicChoice,list]:
    rows=[]
    for c in candidate_bank():
        s=_discrete_ls_score(train,val,input_idx,c); rows.append(DynamicChoice(c.op,c.params,s))
    rows.sort(key=lambda c:c.score)
    best=rows[0]
    # Complexity equivalence band: if within 1% of the best, prefer simpler family.
    comp={'Identity':0,'FirstOrderLag':1,'Delay':1,'SecondOrderOscillator':2}
    equiv=[r for r in rows if r.score <= best.score*1.01 + 1e-10]
    best=min(equiv,key=lambda r:(comp[r.op],r.score))
    return best,rows

class DynamicISGNModel:
    def __init__(self, graph_model, choices:Dict[int,DynamicChoice]): self.graph=graph_model; self.choices=dict(choices); self.dt=graph_model.dt
    def rollout(self,x0:torch.Tensor,u_raw:torch.Tensor):
        arr=u_raw.detach().cpu().numpy(); U=arr.copy()
        for idx,ch in self.choices.items(): U[:,:,idx]=transform_numpy(arr[:,:,idx],self.dt,ch)
        ut=torch.as_tensor(U,dtype=u_raw.dtype,device=u_raw.device)
        return self.graph.rollout(x0,ut)
