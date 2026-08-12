"""Official PySINDy adapter for ISGN Fusion v1.1.

PySINDy is an optional dependency.  The adapter uses continuous SINDYc with
pre-computed midpoint derivatives so trajectory semantics match the frozen ISGN
PHM identification interface.  IdentityLibrary keeps the returned mechanism
terms canonical (states/controls rather than arbitrary polynomial expressions).
Optional Saturation/DeadZone profiling is performed outside PySINDy, while each
candidate is fit by the official PySINDy optimizer.
"""
from __future__ import annotations
from itertools import product
from typing import Any, Dict, Optional
import importlib
import numpy as np

from .base import CanonicalMechanismModel, DiscoveryResult, rollout_nrmse_canonical
from ..contracts import PublicDataset


def _ps():
    try: return importlib.import_module('pysindy')
    except ModuleNotFoundError as e:
        raise RuntimeError('backend="pysindy" requires optional dependency: pip install "isgn-toolkit[pysindy]"') from e

def _names(ds):
    return [x['name'] for x in ds.schema['states']],[x['name'] for x in ds.schema['inputs']]

def _grid(ds,name):
    _,ins=_names(ds); a=np.abs(ds.U[:,:,ins.index(name)].astype(float)).reshape(-1)
    q=max(float(np.quantile(a,.995)),1e-6)
    return [('Identity',None)]+[('Saturation',float(v)) for v in np.linspace(.3*q,.9*q,4)]+[('DeadZone',float(v)) for v in np.linspace(.08*q,.55*q,4)]

def _apply(z,op,p):
    if op=='Identity': return z
    if op=='Saturation': return np.clip(z,-float(p),float(p))
    if op=='DeadZone': return np.sign(z)*np.maximum(np.abs(z)-float(p),0.0)
    raise ValueError(op)

def _trajectory_lists(ds,spec):
    xs=[]; xds=[]; us=[]
    states,inputs=_names(ds)
    for k in range(len(ds.X)):
        x0=ds.X[k,:-1].astype(float); x1=ds.X[k,1:].astype(float)
        xs.append(.5*(x0+x1)); xds.append((x1-x0)/ds.dt)
        u=ds.U[k].astype(float).copy()
        for j,n in enumerate(inputs):
            op,p=spec.get(n,('Identity',None)); u[:,j]=_apply(u[:,j],op,p)
        us.append(u)
    return xs,xds,us

def _mapped_feature_names(raw,states,inputs):
    out=[]
    for n in raw:
        if n.startswith('x') and n[1:].isdigit() and int(n[1:])<len(states): out.append(states[int(n[1:])])
        elif n.startswith('u') and n[1:].isdigit() and int(n[1:])<len(inputs): out.append(inputs[int(n[1:])])
        else: out.append(n)
    return out

def _constraints(known_absent,feature_names,targets):
    rows=[]
    for ti,t in enumerate(targets):
        for s in (known_absent or {}).get(t,[]):
            if s not in feature_names: continue
            r=np.zeros(len(feature_names)*len(targets),float)
            r[ti*len(feature_names)+feature_names.index(s)]=1.0
            rows.append(r)
    if not rows: return None,None
    return np.stack(rows),np.zeros(len(rows))

def _fit_candidate(train,val,spec,*,threshold=.05,alpha=1e-8,known_absent=None,optimizer='stlsq'):
    ps=_ps(); states,inputs=_names(train)
    x,xd,u=_trajectory_lists(train,spec)
    xv,xdv,uv=_trajectory_lists(val,spec)
    library=ps.IdentityLibrary()
    # For IdentityLibrary the feature order is state coordinates followed by controls.
    canonical_features=states+inputs
    C,d=_constraints(known_absent,canonical_features,states)
    if C is not None:
        opt=ps.ConstrainedSR3(reg_weight_lam=max(float(threshold),1e-6),regularizer='l0',
                              constraint_lhs=C,constraint_rhs=d,constraint_order='target',
                              equality_constraints=True,normalize_columns=True,unbias=False,max_iter=100)
        opt_name='ConstrainedSR3'
    elif optimizer=='stable-linear':
        opt=ps.StableLinearSR3(reg_weight_lam=max(float(threshold),1e-6),regularizer='l1',normalize_columns=True,max_iter=100)
        opt_name='StableLinearSR3'
    else:
        opt=ps.STLSQ(threshold=float(threshold),alpha=float(alpha),normalize_columns=True,max_iter=20)
        opt_name='STLSQ'
    m=ps.SINDy(feature_library=library,optimizer=opt)
    m.fit(x,t=float(train.dt),x_dot=xd,u=u)
    raw_names=list(m.get_feature_names()); feat=_mapped_feature_names(raw_names,states,inputs)
    co=np.asarray(m.coefficients(),float)  # targets x features
    # derivative validation on exactly the same transformed coordinates
    rr=[]; yy=[]
    for xx,dd,uu in zip(xv,xdv,uv): rr.append(np.asarray(m.predict(xx,u=uu),float)); yy.append(dd)
    R=np.concatenate(rr); Y=np.concatenate(yy)
    derr=float(np.mean(np.sqrt(np.mean((Y-R)**2,axis=0))/(np.std(Y,axis=0)+1e-12)))
    graph={}; coeff={}; ops={}; params={}
    for ti,t in enumerate(states):
        idx=np.where(np.abs(co[ti])>1e-8)[0]
        graph[t]=[feat[j] for j in idx]
        coeff[t]={feat[j]:float(co[ti,j]) for j in idx}
        ops[t]={}; params[t]={}
        for s in graph[t]:
            op,p=spec.get(s,('Identity',None)); ops[t][s]=op; params[t][s]=p
    can=CanonicalMechanismModel(schema=train.schema,dt=train.dt,coefficients=coeff,operators=ops,operator_parameters=params)
    return can,graph,coeff,ops,params,derr,{'optimizer':opt_name,'raw_feature_names':raw_names,'mapped_feature_names':feat,'pysindy_version':getattr(ps,'__version__','unknown')}

class PySINDyBackend:
    name='pysindy'
    @classmethod
    def available(cls):
        try: importlib.import_module('pysindy'); return True
        except ModuleNotFoundError: return False

    def _run(self,train,val,*,constitutive_search=False,known_present=None,known_absent=None,
             thresholds=(.005,.02,.05),alpha=1e-8,optimizer='stlsq',**kwargs):
        states,inputs=_names(train)
        choices=[]
        for meta,n in zip(train.schema['inputs'],inputs):
            choices.append(_grid(train,n) if (constitutive_search and meta.get('role','input')!='context') else [('Identity',None)])
        best=None
        for combo in product(*choices):
            spec={n:c for n,c in zip(inputs,combo)}
            for th in thresholds:
                can,g,c,o,p,derr,meta=_fit_candidate(train,val,spec,threshold=th,alpha=alpha,known_absent=known_absent,optimizer=optimizer)
                op_complex=sum(v[0]!='Identity' for v in combo); complexity=sum(len(x) for x in g.values())
                key=(derr+2e-4*op_complex+1e-7*complexity,op_complex,complexity)
                if best is None or key<best[0]: best=(key,can,g,c,o,p,derr,meta,spec,th)
        _,can,g,c,o,p,derr,meta,spec,th=best
        val_roll=rollout_nrmse_canonical(can,val)
        meta.update({'derivative_nrmse':derr,'threshold':th,'transforms':spec,'official':True,
                     'known_present':known_present or {},'known_absent':known_absent or {}})
        return DiscoveryResult(g,c,o,p,val_roll,'pysindy',meta,can)

    def recover(self,train,validation,**kwargs): return self._run(train,validation,**kwargs)
    def complete(self,train,validation,*,known_present=None,known_absent=None,**kwargs):
        return self._run(train,validation,known_present=known_present,known_absent=known_absent,**kwargs)
