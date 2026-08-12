"""Scale-normalized sparse linear discovery using scikit-learn Lasso.

This is a solver backend, not a new mechanism grammar.  It exists to provide an
independent sparse optimizer for industrial data whose engineering units span
many orders of magnitude.  Final outputs are mapped back to physical units and
the common CanonicalMechanismModel contract.
"""
from __future__ import annotations
from typing import Iterable, Optional
import numpy as np
from sklearn.linear_model import Lasso

from .base import CanonicalMechanismModel, DiscoveryResult, rollout_nrmse_canonical


def _names(ds):
    return [x['name'] for x in ds.schema['states']], [x['name'] for x in ds.schema['inputs']]


def _design(ds):
    states,inputs=_names(ds)
    xm=.5*(ds.X[:,:-1].astype(float)+ds.X[:,1:].astype(float))
    P=np.concatenate([xm,ds.U.astype(float)],axis=2).reshape(-1,len(states)+len(inputs))
    Y=((ds.X[:,1:].astype(float)-ds.X[:,:-1].astype(float))/ds.dt).reshape(-1,len(states))
    return P,Y,states+inputs,states


def _allowed_columns(features, target, known_absent):
    forbidden=set((known_absent or {}).get(target,[]))
    return np.asarray([i for i,n in enumerate(features) if n not in forbidden],dtype=int)


def _fit(train,val,*,known_present=None,known_absent=None,
         alphas: Iterable[float]=(1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1),
         complexity_penalty: float=5e-4, max_iter: int=20000, tol: float=1e-8,
         constitutive_search: bool=False, **kwargs):
    if constitutive_search:
        raise ValueError('sklearn-lasso backend currently supports Identity terms only')
    P,Y,features,targets=_design(train); Pv,Yv,_,_=_design(val)
    # RMS feature scaling preserves zero and avoids intercept injection. Target
    # scaling makes one alpha grid meaningful across heterogeneous state units.
    ps=np.sqrt(np.mean(P*P,axis=0)); ps=np.maximum(ps,1e-12)
    ys=np.std(Y,axis=0); ys=np.maximum(ys,1e-12)
    Pn=P/ps; Pvn=Pv/ps; Yn=Y/ys; Yvn=Yv/ys
    Xi_n=np.zeros((len(features),len(targets)),dtype=float)
    chosen={}; per_target_score={}
    for k,t in enumerate(targets):
        cols=_allowed_columns(features,t,known_absent)
        if not len(cols):
            chosen[t]=None; per_target_score[t]=1.0; continue
        best=None
        for alpha in alphas:
            model=Lasso(alpha=float(alpha),fit_intercept=False,max_iter=int(max_iter),tol=float(tol))
            model.fit(Pn[:,cols],Yn[:,k])
            c=np.asarray(model.coef_,dtype=float)
            pred=Pvn[:,cols]@c
            score=float(np.sqrt(np.mean((Yvn[:,k]-pred)**2))/(np.std(Yvn[:,k])+1e-12))
            nz=int(np.count_nonzero(np.abs(c)>1e-10))
            key=(score+float(complexity_penalty)*nz,nz,float(alpha))
            if best is None or key<best[0]: best=(key,float(alpha),c,score)
        _,alpha,c,score=best
        Xi_n[cols,k]=c; chosen[t]=alpha; per_target_score[t]=score
    Xi=Xi_n*ys[None,:]/ps[:,None]
    graph={}; coeff={}; ops={}; params={}
    for k,t in enumerate(targets):
        idx=np.where(np.abs(Xi_n[:,k])>1e-10)[0]
        graph[t]=[features[j] for j in idx]
        coeff[t]={features[j]:float(Xi[j,k]) for j in idx}
        ops[t]={s:'Identity' for s in graph[t]}; params[t]={s:None for s in graph[t]}
    can=CanonicalMechanismModel(schema=train.schema,dt=train.dt,coefficients=coeff,operators=ops,operator_parameters=params)
    val_roll=rollout_nrmse_canonical(can,val)
    derr=float(np.mean(list(per_target_score.values()))) if per_target_score else float('inf')
    meta={'optimizer':'sklearn.linear_model.Lasso','normalize_features':'rms','normalize_targets':'std',
          'alphas':chosen,'per_target_derivative_nrmse':per_target_score,'derivative_nrmse':derr,
          'known_present':known_present or {},'known_absent':known_absent or {},'official_external':True}
    return DiscoveryResult(graph,coeff,ops,params,val_roll,'sklearn-lasso',meta,can)


class SklearnLassoBackend:
    name='sklearn-lasso'
    @classmethod
    def available(cls): return True
    def recover(self,train,validation,**kwargs): return _fit(train,validation,**kwargs)
    def complete(self,train,validation,*,known_present=None,known_absent=None,**kwargs):
        return _fit(train,validation,known_present=known_present,known_absent=known_absent,**kwargs)
