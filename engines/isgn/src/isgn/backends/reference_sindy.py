"""Small independent STLSQ-SINDYc reference backend.

This is NOT official PySINDy. It exists for offline QA and for validating the
solver-agnostic ISGN backend contract when the optional ``pysindy`` dependency
cannot be installed. Production users should prefer backend="pysindy".
"""
from __future__ import annotations
from itertools import product
from typing import Dict, Iterable, Optional
import numpy as np

from .base import CanonicalMechanismModel, DiscoveryResult, rollout_nrmse_canonical
from ..contracts import PublicDataset


def _names(ds):
    return [x['name'] for x in ds.schema['states']],[x['name'] for x in ds.schema['inputs']]

def _candidate_grid(ds,name):
    _,ins=_names(ds); a=np.abs(ds.U[:,:,ins.index(name)].astype(float)).reshape(-1)
    q=max(float(np.quantile(a,.995)),1e-6)
    return [('Identity',None)]+[('Saturation',float(v)) for v in np.linspace(.3*q,.9*q,4)]+[('DeadZone',float(v)) for v in np.linspace(.08*q,.55*q,4)]

def _trans(z,op,p):
    if op=='Identity': return z
    if op=='Saturation': return np.clip(z,-p,p)
    if op=='DeadZone': return np.sign(z)*np.maximum(np.abs(z)-p,0)
    raise ValueError(op)

def _design(ds,spec):
    states,inputs=_names(ds)
    xm=.5*(ds.X[:,:-1].astype(float)+ds.X[:,1:].astype(float))
    cols=[]; names=[]
    for j,n in enumerate(states): cols.append(xm[:,:,j].reshape(-1)); names.append(n)
    for j,n in enumerate(inputs):
        op,p=spec.get(n,('Identity',None)); cols.append(_trans(ds.U[:,:,j].astype(float),op,p).reshape(-1)); names.append(n)
    P=np.stack(cols,axis=1)
    Y=((ds.X[:,1:].astype(float)-ds.X[:,:-1].astype(float))/ds.dt).reshape(-1,len(states))
    return P,Y,names,states

def _ridge(P,Y,alpha):
    return np.linalg.solve(P.T@P+alpha*np.eye(P.shape[1]),P.T@Y)

def _stlsq(P,Y,threshold,alpha,allowed):
    Xi=np.zeros((P.shape[1],Y.shape[1]))
    for k in range(Y.shape[1]):
        cols=np.where(allowed[:,k])[0]
        if not len(cols): continue
        X=_ridge(P[:,cols],Y[:,[k]],alpha)
        for _ in range(20):
            scale=max(float(np.max(np.abs(X))),1e-12); keep=np.abs(X[:,0])>=threshold*scale
            if not np.any(keep): X[:]=0; break
            new=np.zeros_like(X); new[keep]=_ridge(P[:,cols[keep]],Y[:,[k]],alpha)
            if np.max(np.abs(new-X))<1e-10: X=new; break
            X=new
        Xi[cols,k]=X[:,0]
    return Xi

def _fit(train,val,*,constitutive_search=False,known_absent=None,thresholds=(.005,.02,.05),alpha=1e-8,normalize_columns=False,**kwargs):
    states,inputs=_names(train); known_absent=known_absent or {}
    candidates=[]
    for meta,n in zip(train.schema['inputs'],inputs):
        candidates.append(_candidate_grid(train,n) if (constitutive_search and meta.get('role','input')!='context') else [('Identity',None)])
    best=None
    for combo in product(*candidates):
        spec={n:v for n,v in zip(inputs,combo)}
        P,Y,fn,targets=_design(train,spec); Pv,Yv,_,_=_design(val,spec)
        if normalize_columns:
            scale=np.sqrt(np.mean(P*P,axis=0)); scale=np.maximum(scale,1e-12)
            Pf=P/scale; Pvf=Pv/scale
        else:
            scale=np.ones(P.shape[1],dtype=float); Pf=P; Pvf=Pv
        allowed=np.ones((len(fn),len(targets)),bool)
        for k,t in enumerate(targets):
            for s in known_absent.get(t,[]):
                if s in fn: allowed[fn.index(s),k]=False
        for th in thresholds:
            Xi_fit=_stlsq(Pf,Y,th,alpha,allowed)
            Xi=Xi_fit/scale[:,None]
            score=float(np.mean(np.sqrt(np.mean((Yv-Pv@Xi)**2,axis=0))/(np.std(Yv,axis=0)+1e-12)))
            complexity=int(np.count_nonzero(Xi_fit)); op_complex=sum(op!='Identity' for op,_ in combo)
            key=(score+2e-4*op_complex+1e-7*complexity,op_complex,complexity)
            if best is None or key<best[0]: best=(key,spec,Xi,Xi_fit,score,fn,targets,th)
    _,spec,Xi,Xi_support,score,fn,targets,th=best
    graph={}; coeff={}; ops={}; params={}
    for k,t in enumerate(targets):
        idx=np.where(np.abs(Xi_support[:,k])>1e-12)[0]
        graph[t]=[fn[j] for j in idx]
        coeff[t]={fn[j]:float(Xi[j,k]) for j in idx}
        ops[t]={}; params[t]={}
        for s in graph[t]:
            op,p=spec.get(s,('Identity',None)); ops[t][s]=op; params[t][s]=p
    model=CanonicalMechanismModel(schema=train.schema,dt=train.dt,coefficients=coeff,operators=ops,operator_parameters=params)
    val_roll=rollout_nrmse_canonical(model,val)
    return DiscoveryResult(graph,coeff,ops,params,val_roll,'reference-sindy',{'derivative_nrmse':score,'threshold':th,'transforms':spec,'normalize_columns':bool(normalize_columns),'official':False},model)

class ReferenceSINDyBackend:
    name='reference-sindy'
    @classmethod
    def available(cls): return True
    def recover(self,train,validation,**kwargs): return _fit(train,validation,**kwargs)
    def complete(self,train,validation,*,known_present=None,known_absent=None,**kwargs):
        # known_present is audited by the certification layer; known_absent is enforced here.
        r=_fit(train,validation,known_absent=known_absent,**kwargs)
        r.metadata['known_present']=known_present or {}; r.metadata['known_absent']=known_absent or {}
        return r
