from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional
import json
import numpy as np
from .contracts import PublicDataset


class DiscreteLinearStateSpaceModel:
    """Engineering fallback: x[k+1] = [x[k],u[k],1] W.

    This model is intentionally *not* a certified Mechanism Graph. It exists so
    PHM/monitoring workflows can retain a predictive residual baseline when
    mechanistic discovery backends disagree or abstain.
    """
    def __init__(self,W:np.ndarray,schema:Dict[str,Any],dt:float):
        self.W=np.asarray(W,float); self.schema=schema; self.dt=float(dt)
    @property
    def state_names(self): return [v['name'] for v in self.schema['states']]
    @property
    def input_names(self): return [v['name'] for v in self.schema['inputs']]
    def one_step_numpy(self,x,u):
        x=np.asarray(x,float); u=np.asarray(u,float)
        z=np.concatenate([x,u,np.ones((*x.shape[:-1],1),float)],axis=-1)
        return z@self.W
    def rollout_numpy(self,x0,u):
        x=np.asarray(x0,float).copy(); u=np.asarray(u,float); out=[x.copy()]
        for k in range(u.shape[-2]):
            x=self.one_step_numpy(x,u[...,k,:]); out.append(x.copy())
        return np.stack(out,axis=-2)
    def to_dict(self): return {'W':self.W.tolist(),'schema':self.schema,'dt':self.dt,'model_kind':'discrete-linear-state-space'}
    @classmethod
    def from_dict(cls,d): return cls(np.asarray(d['W'],float),d['schema'],d['dt'])


def _nrmse(pred,truth):
    pred=np.asarray(pred,float); truth=np.asarray(truth,float)
    rmse=np.sqrt(np.mean((pred-truth)**2,axis=(0,1))); sd=np.maximum(np.std(truth,axis=(0,1)),1e-9)
    return float(np.mean(rmse/sd))


@dataclass
class FallbackSummary:
    status: str
    model_type: str
    train_one_step_nrmse: float
    validation_rollout_nrmse: float
    n_parameters: int
    mechanism_certifiable: bool
    note: str
    def to_dict(self): return asdict(self)

@dataclass
class FallbackResult:
    model: DiscreteLinearStateSpaceModel
    summary: FallbackSummary


def fit_linear_state_space_fallback(train:PublicDataset,val:PublicDataset,*,ridge:float=1e-8,max_steps:int=200)->FallbackResult:
    X=np.asarray(train.X,float); U=np.asarray(train.U,float); D=X.shape[-1]
    Z=np.concatenate([X[:,:-1],U,np.ones((*U.shape[:2],1),float)],axis=-1).reshape(-1,D+U.shape[-1]+1)
    Y=X[:,1:].reshape(-1,D)
    W=np.linalg.solve(Z.T@Z+ridge*np.eye(Z.shape[1]),Z.T@Y)
    model=DiscreteLinearStateSpaceModel(W,train.schema,train.dt)
    yp=Z@W; one=_nrmse(yp[:,None,:],Y[:,None,:])
    steps=min(int(max_steps),val.U.shape[1]); pr=model.rollout_numpy(val.X[:,0],val.U[:,:steps]); truth=val.X[:,:steps+1]
    roll=_nrmse(pr,truth)
    summ=FallbackSummary('PREDICTIVE_FALLBACK_ONLY','discrete-linear-state-space',one,roll,int(W.size),False,
                         'Useful for predictive residuals/baseline only; do not interpret coefficients as a certified Operational Mechanism Graph.')
    return FallbackResult(model,summ)


def save_fallback(result:FallbackResult,path:str|Path):
    Path(path).write_text(json.dumps({'model':result.model.to_dict(),'summary':result.summary.to_dict()},indent=2),encoding='utf-8')

def load_fallback(path:str|Path)->FallbackResult:
    d=json.loads(Path(path).read_text(encoding='utf-8')); model=DiscreteLinearStateSpaceModel.from_dict(d['model']); return FallbackResult(model,FallbackSummary(**d['summary']))
