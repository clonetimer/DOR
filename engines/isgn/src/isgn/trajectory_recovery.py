"""Trajectory-level recovery of missing dynamics under partial physics.

The design absorbs the central UDE idea useful to ISGN without reimplementing
SciML/UDE: known mechanism terms are fixed inside an integrator, a small
universal approximator learns only the unexplained vector field from complete
trajectory mismatch, and the learned residual is then distilled back into the
finite ISGN mechanism-term grammar.

This is intentionally a candidate-generation path.  A distilled mechanism must
still pass normal ISGN validation/certification before it is trusted.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Mapping, Sequence
import math
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Lasso

from .contracts import PublicDataset
from .mechanism_recovery import evaluate_term


def _names(ds: PublicDataset):
    return ([str(x["name"]) for x in ds.schema.get("states",[])],
            [str(x["name"]) for x in ds.schema.get("inputs",[])])


def _parse_term_torch(term: str, x: torch.Tensor, u: torch.Tensor, state_names: Sequence[str], input_names: Sequence[str]) -> torch.Tensor:
    term=str(term).strip()
    lookup={n:x[...,i] for i,n in enumerate(state_names)}
    lookup.update({n:u[...,i] for i,n in enumerate(input_names)})
    if term=="__constant__": return torch.ones_like(x[...,0])
    if term in lookup: return lookup[term]
    for op in ("sin","cos","square","cube"):
        prefix=op+"("
        if term.startswith(prefix) and term.endswith(")"):
            name=term[len(prefix):-1].strip(); z=lookup[name]
            if op=="sin": return torch.sin(z)
            if op=="cos": return torch.cos(z)
            if op=="square": return z*z
            if op=="cube": return z*z*z
    if term.startswith("product(") and term.endswith(")"):
        body=term[len("product("):-1]; a,b=[s.strip() for s in body.split(",",1)]
        return lookup[a]*lookup[b]
    raise ValueError(f"unsupported trajectory mechanism term {term!r}")


class _ResidualMLP(nn.Module):
    def __init__(self, nin: int, nout: int, hidden: int, depth: int):
        super().__init__(); layers=[]; d=nin
        for _ in range(int(depth)):
            layers += [nn.Linear(d,int(hidden)),nn.Tanh()]; d=int(hidden)
        layers.append(nn.Linear(d,nout)); self.net=nn.Sequential(*layers)
    def forward(self,z): return self.net(z)


@dataclass
class TrajectoryMissingDynamicsResult:
    unknown_targets: List[str]
    known_terms: Dict[str, Dict[str, float]]
    distilled_terms: Dict[str, List[str]]
    distilled_coefficients: Dict[str, Dict[str, float]]
    neural_training_loss: float
    validation_rollout_nrmse_per_state: Dict[str, float]
    validation_rollout_nrmse: float
    distilled_validation_rollout_nrmse_per_state: Dict[str, float]
    distilled_validation_rollout_nrmse: float
    distillation_nrmse_per_target: Dict[str, float]
    candidate_terms: Dict[str, List[str]]
    metadata: Dict[str, object]
    backend: str = "trajectory-residual-ua+distillation"
    def to_dict(self): return asdict(self)


class TrajectoryMissingDynamicsModel:
    """Internal fitted model returned only when explicitly requested by callers."""
    def __init__(self, net, state_mean, state_std, input_mean, input_std, state_names, input_names, unknown_targets, known_terms, dt):
        self.net=net; self.state_mean=state_mean; self.state_std=state_std; self.input_mean=input_mean; self.input_std=input_std
        self.state_names=list(state_names); self.input_names=list(input_names); self.unknown_targets=list(unknown_targets); self.known_terms=known_terms; self.dt=float(dt)


def _rhs_known(x,u,state_names,input_names,known_terms):
    cols=[]
    for target in state_names:
        y=torch.zeros_like(x[...,0])
        for term,c in known_terms.get(target,{}).items():
            y=y+float(c)*_parse_term_torch(term,x,u,state_names,input_names)
        cols.append(y)
    return torch.stack(cols,-1)


def _full_rhs(model: TrajectoryMissingDynamicsModel, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    known=_rhs_known(x,u,model.state_names,model.input_names,model.known_terms)
    z=torch.cat([(x-model.state_mean)/model.state_std,(u-model.input_mean)/model.input_std],dim=-1)
    residual=model.net(z)
    out=known.clone()
    for j,t in enumerate(model.unknown_targets): out[...,model.state_names.index(t)] += residual[...,j]
    return out


def _rk4(model,x,u):
    h=model.dt; k1=_full_rhs(model,x,u); k2=_full_rhs(model,x+.5*h*k1,u); k3=_full_rhs(model,x+.5*h*k2,u); k4=_full_rhs(model,x+h*k3,u)
    return x+h*(k1+2*k2+2*k3+k4)/6.0


def _rollout_nrmse(model: TrajectoryMissingDynamicsModel, ds: PublicDataset) -> tuple[float,Dict[str,float]]:
    X=np.asarray(ds.X,float); U=np.asarray(ds.U,float); preds=[]
    with torch.no_grad():
        for k in range(len(X)):
            x=torch.tensor(X[k,0:1],dtype=torch.float32); rows=[X[k,0].copy()]
            for j in range(U.shape[1]):
                uu=torch.tensor(U[k,j:j+1],dtype=torch.float32); x=_rk4(model,x,uu); rows.append(x.cpu().numpy()[0])
            preds.append(np.asarray(rows))
    P=np.asarray(preds); err=np.sqrt(np.mean((P-X)**2,axis=(0,1))); scale=np.std(X,axis=(0,1))+1e-12; vals=err/scale
    return float(np.mean(vals)),{model.state_names[i]:float(vals[i]) for i in range(len(vals))}


def _rollout_distilled(ds: PublicDataset, state_names: Sequence[str], input_names: Sequence[str], known_terms: Mapping[str,Mapping[str,float]], distilled: Mapping[str,Mapping[str,float]]) -> tuple[float,Dict[str,float]]:
    X=np.asarray(ds.X,float); U=np.asarray(ds.U,float); dt=float(ds.dt); preds=[]
    full={str(t):{str(k):float(v) for k,v in row.items()} for t,row in known_terms.items()}
    for target,row in distilled.items():
        full.setdefault(str(target),{}).update({str(k):float(v) for k,v in row.items()})
    def rhs_np(z,uu):
        xt=torch.tensor(np.asarray(z,float)[None,:],dtype=torch.float32); ut=torch.tensor(np.asarray(uu,float)[None,:],dtype=torch.float32)
        with torch.no_grad(): return _rhs_known(xt,ut,state_names,input_names,full).cpu().numpy()[0].astype(float)
    for k in range(len(X)):
        z=X[k,0].copy(); rows=[z.copy()]
        for j in range(U.shape[1]):
            uu=U[k,j]; h=dt
            k1=rhs_np(z,uu); k2=rhs_np(z+.5*h*k1,uu); k3=rhs_np(z+.5*h*k2,uu); k4=rhs_np(z+h*k3,uu)
            z=z+h*(k1+2*k2+2*k3+k4)/6.0; rows.append(z.copy())
        preds.append(np.asarray(rows))
    P=np.asarray(preds); err=np.sqrt(np.mean((P-X)**2,axis=(0,1))); scale=np.std(X,axis=(0,1))+1e-12; vals=err/scale
    return float(np.mean(vals)),{str(state_names[i]):float(vals[i]) for i in range(len(vals))}


def _network_residual(model: TrajectoryMissingDynamicsModel, ds: PublicDataset) -> np.ndarray:
    xm=.5*(np.asarray(ds.X[:,:-1],float)+np.asarray(ds.X[:,1:],float)); uu=np.asarray(ds.U,float)
    with torch.no_grad():
        xt=torch.tensor(xm.reshape(-1,xm.shape[-1]),dtype=torch.float32)
        ut=torch.tensor(uu.reshape(-1,uu.shape[-1]),dtype=torch.float32)
        z=torch.cat([(xt-model.state_mean)/model.state_std,(ut-model.input_mean)/model.input_std],dim=-1)
        return model.net(z).cpu().numpy()


def _distill(model: TrajectoryMissingDynamicsModel, train: PublicDataset, validation: PublicDataset,
             candidate_terms: Mapping[str,Sequence[str]], *, max_terms: int, complexity_penalty: float):
    """Distill the learned universal residual into a small declared mechanism basis.

    Candidate supports are compared on a held-out trajectory set, so the
    distillation step does not reward a larger expression merely because it can
    imitate small neural residual artifacts on the training trajectories.
    """
    from itertools import combinations
    R=_network_residual(model,train); RV=_network_residual(model,validation)
    selected={}; coeffs={}; nrmses={}
    for jj,target in enumerate(model.unknown_targets):
        candidates=[str(x) for x in candidate_terms[target]]
        A=np.stack([evaluate_term(train,t) for t in candidates],axis=1); V=np.stack([evaluate_term(validation,t) for t in candidates],axis=1)
        y=R[:,jj]; vy=RV[:,jj]
        good=np.isfinite(A).all(axis=1)&np.isfinite(y); vgood=np.isfinite(V).all(axis=1)&np.isfinite(vy)
        A,y,V,vy=A[good],y[good],V[vgood],vy[vgood]
        if len(A)<10 or len(V)<5: raise ValueError("insufficient finite samples for trajectory residual distillation")
        best=None
        upto=min(int(max_terms),len(candidates))
        for k in range(1,upto+1):
            for idx in combinations(range(len(candidates)),k):
                idx=list(idx); beta=np.linalg.lstsq(A[:,idx],y,rcond=None)[0]
                pred=V[:,idx]@beta
                nrmse=float(np.sqrt(np.mean((pred-vy)**2))/(np.std(vy)+1e-12))
                score=nrmse+float(complexity_penalty)*k
                key=(score,k,nrmse,tuple(idx))
                if best is None or key<best[0]: best=(key,idx,beta,nrmse)
        _,idx,beta,nrmse=best
        selected[target]=[candidates[i] for i in idx]
        coeffs[target]={candidates[i]:float(beta[j]) for j,i in enumerate(idx)}
        nrmses[target]=float(nrmse)
    return selected,coeffs,nrmses

def recover_missing_dynamics_trajectory(
    train: PublicDataset,
    validation: PublicDataset,
    *,
    known_terms: Mapping[str,Mapping[str,float]],
    unknown_targets: Sequence[str],
    candidate_terms: Mapping[str,Sequence[str]],
    hidden: int = 32,
    depth: int = 2,
    epochs: int = 800,
    batch_windows: int = 32,
    window_steps: int = 12,
    learning_rate: float = 2e-3,
    weight_decay: float = 1e-5,
    seed: int = 0,
    distill_max_terms: int = 3,
    distill_complexity_penalty: float = 5e-2,
    return_model: bool = False,
):
    """Learn unexplained dynamics from trajectory mismatch and distill to terms.

    All state equations must be represented in ``known_terms``; targets listed in
    ``unknown_targets`` receive an additional universal residual.  Known terms are
    fixed throughout training.  This routine is intended for sparse/noisy sampling
    where differentiating observations is unreliable.
    """
    train.validate(); validation.validate()
    if abs(float(train.dt)-float(validation.dt))>1e-12: raise ValueError("train/validation dt mismatch")
    state_names,input_names=_names(train); vs,vi=_names(validation)
    if state_names!=vs or input_names!=vi: raise ValueError("train/validation schema mismatch")
    unknown=[str(x) for x in unknown_targets]
    if not unknown: raise ValueError("unknown_targets must not be empty")
    for t in unknown:
        if t not in state_names: raise KeyError(f"unknown target {t!r} is not a state")
        if t not in candidate_terms or not candidate_terms[t]: raise ValueError(f"candidate terms required for {t!r}")
    known={str(t):{str(k):float(v) for k,v in row.items()} for t,row in known_terms.items()}
    missing_eq=[s for s in state_names if s not in known]
    if missing_eq: raise ValueError(f"known_terms must include every state equation (empty mapping allowed); missing {missing_eq}")

    torch.manual_seed(int(seed)); np_rng=np.random.RandomState(int(seed))
    X=torch.tensor(np.asarray(train.X,float),dtype=torch.float32); U=torch.tensor(np.asarray(train.U,float),dtype=torch.float32)
    sm=X[:,:-1].reshape(-1,len(state_names)).mean(0); ss=X[:,:-1].reshape(-1,len(state_names)).std(0)+1e-6
    if len(input_names):
        im=U.reshape(-1,len(input_names)).mean(0); ins=U.reshape(-1,len(input_names)).std(0)+1e-6
    else:
        im=torch.zeros(0); ins=torch.ones(0)
    net=_ResidualMLP(len(state_names)+len(input_names),len(unknown),int(hidden),int(depth))
    model=TrajectoryMissingDynamicsModel(net,sm,ss,im,ins,state_names,input_names,unknown,known,float(train.dt))
    opt=torch.optim.Adam(net.parameters(),lr=float(learning_rate),weight_decay=float(weight_decay))
    max_start=X.shape[1]-int(window_steps)-1
    if max_start<1: raise ValueError("trajectory too short for requested window_steps")
    loss_value=float("nan")
    for _ in range(int(epochs)):
        ti=torch.tensor(np_rng.randint(0,len(X),size=int(batch_windows)),dtype=torch.long)
        starts=torch.tensor(np_rng.randint(0,max_start+1,size=int(batch_windows)),dtype=torch.long)
        x=torch.stack([X[ti[i],starts[i]] for i in range(len(ti))]); loss=torch.tensor(0.0)
        for s in range(int(window_steps)):
            if len(input_names): uu=torch.stack([U[ti[i],starts[i]+s] for i in range(len(ti))])
            else: uu=torch.zeros((len(ti),0),dtype=torch.float32)
            x=_rk4(model,x,uu); y=torch.stack([X[ti[i],starts[i]+s+1] for i in range(len(ti))])
            loss=loss+torch.mean(((x-y)/ss)**2)
        loss=loss/float(window_steps); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),5.0); opt.step(); loss_value=float(loss.detach())
    selected,coeffs,dnrmse=_distill(model,train,validation,candidate_terms,max_terms=distill_max_terms,complexity_penalty=distill_complexity_penalty)
    valmean,valper=_rollout_nrmse(model,validation)
    dvalmean,dvalper=_rollout_distilled(validation,state_names,input_names,known,coeffs)
    result=TrajectoryMissingDynamicsResult(
        unknown_targets=unknown,known_terms=known,distilled_terms=selected,distilled_coefficients=coeffs,
        neural_training_loss=loss_value,validation_rollout_nrmse_per_state=valper,validation_rollout_nrmse=valmean,
        distilled_validation_rollout_nrmse_per_state=dvalper,distilled_validation_rollout_nrmse=dvalmean,
        distillation_nrmse_per_target=dnrmse,candidate_terms={str(k):[str(x) for x in v] for k,v in candidate_terms.items()},
        metadata={
            "known_terms_frozen":True,"training_objective":"multiple-shooting trajectory mismatch","derivatives_used_for_training":False,
            "universal_approximator":"small tanh MLP","distillation":"scale-normalized sparse regression over declared mechanism terms",
            "candidate_only":True,"requires_downstream_validation_and_certification":True,
            "inspiration":"UDE knowledge-enhanced missing-dynamics recovery; not a SciML/UDE implementation",
            "seed":int(seed),"epochs":int(epochs),"window_steps":int(window_steps),"batch_windows":int(batch_windows),
        })
    return (result,model) if return_model else result
