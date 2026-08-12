from __future__ import annotations
from dataclasses import dataclass, asdict
from itertools import combinations_with_replacement
from typing import Any, Dict, List, Mapping, Sequence
import numpy as np
from sklearn.linear_model import Lasso
from .contracts import AlgebraicDataset

@dataclass
class AlgebraicRelationResult:
    target: str
    inputs: List[str]
    backend: str
    selected_terms: List[str]
    coefficients: Dict[str,float]
    intercept: float
    validation_nrmse: float
    support_threshold: float
    metadata: Dict[str,Any]
    def to_dict(self): return asdict(self)

def _column(ds: AlgebraicDataset, name: str) -> np.ndarray:
    return np.asarray(ds.values[:,ds.variable_names.index(name)],dtype=float)

def polynomial_library(ds: AlgebraicDataset, inputs: Sequence[str], *, degree: int=2):
    terms=[]; cols=[]
    for n in inputs:
        terms.append(n); cols.append(_column(ds,n))
    if degree >= 2:
        for a,b in combinations_with_replacement(inputs,2):
            terms.append(f"{a}*{b}"); cols.append(_column(ds,a)*_column(ds,b))
    return terms,np.stack(cols,axis=1)

def discover_algebraic_relation(train: AlgebraicDataset, validation: AlgebraicDataset, *,
                                inputs: Sequence[str], target: str, degree: int=2,
                                alpha_grid: Sequence[float]=(1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2),
                                support_threshold: float=2e-3, complexity_weight: float=5e-4) -> AlgebraicRelationResult:
    train.validate(); validation.validate()
    terms,A=polynomial_library(train,inputs,degree=degree); _,V=polynomial_library(validation,inputs,degree=degree)
    y=_column(train,target); vy=_column(validation,target)
    good=np.isfinite(A).all(axis=1)&np.isfinite(y); vgood=np.isfinite(V).all(axis=1)&np.isfinite(vy)
    A=A[good]; y=y[good]; V=V[vgood]; vy=vy[vgood]
    if len(A)<20 or len(V)<10: raise ValueError("insufficient finite samples for algebraic discovery")
    mu=A.mean(0); scale=A.std(0); scale=np.where(scale>1e-15,scale,1.0)
    ymu=float(y.mean()); yscale=float(y.std() or 1.0)
    An=(A-mu)/scale; Vn=(V-mu)/scale; yn=(y-ymu)/yscale
    best=None
    for alpha in alpha_grid:
        m=Lasso(alpha=float(alpha),fit_intercept=True,max_iter=20000,tol=1e-8,selection="cyclic")
        m.fit(An,yn)
        predn=m.predict(Vn); nrmse=float(np.sqrt(np.mean((predn-(vy-ymu)/yscale)**2)))
        support=np.abs(m.coef_)>=support_threshold; score=nrmse+complexity_weight*int(support.sum())
        if best is None or score<best[0]: best=(score,nrmse,m,support,float(alpha))
    _,nrmse,m,support,alpha=best
    # physical coefficients from normalized coordinates
    beta=np.asarray(m.coef_,float)*yscale/scale
    intercept=ymu+float(m.intercept_)*yscale-float(mu@beta)
    coeff={terms[i]:float(beta[i]) for i in range(len(terms)) if bool(support[i])}
    selected=[terms[i] for i in range(len(terms)) if bool(support[i])]
    return AlgebraicRelationResult(str(target),list(inputs),"sklearn-polynomial-lasso",selected,coeff,float(intercept),float(nrmse),float(support_threshold),
        {"alpha":alpha,"degree":degree,"train_rows":int(len(A)),"validation_rows":int(len(V)),"normalized_support":{terms[i]:float(m.coef_[i]) for i in range(len(terms)) if bool(support[i])}})


def discover_algebraic_relation_subset(train: AlgebraicDataset, validation: AlgebraicDataset, *,
                                       inputs: Sequence[str], target: str, degree: int=2,
                                       max_terms: int=3, complexity_weight: float=1e-3) -> AlgebraicRelationResult:
    """Small-library exhaustive subset OLS for interpretable constitutive scopes.

    Intended for finite industrial grammars where the candidate term library is
    deliberately small. Selection is performed on held-out validation error plus
    an explicit term-count penalty; coefficients are refit in physical units.
    """
    from itertools import combinations
    train.validate(); validation.validate()
    terms,A=polynomial_library(train,inputs,degree=degree); _,V=polynomial_library(validation,inputs,degree=degree)
    y=_column(train,target); vy=_column(validation,target)
    good=np.isfinite(A).all(axis=1)&np.isfinite(y); vgood=np.isfinite(V).all(axis=1)&np.isfinite(vy)
    A=A[good]; y=y[good]; V=V[vgood]; vy=vy[vgood]
    best=None
    for k in range(1,min(int(max_terms),len(terms))+1):
        for idx in combinations(range(len(terms)),k):
            X=np.column_stack([np.ones(len(A)),A[:,idx]])
            beta=np.linalg.lstsq(X,y,rcond=None)[0]
            pred=np.column_stack([np.ones(len(V)),V[:,idx]])@beta
            nr=float(np.sqrt(np.mean((pred-vy)**2))/(np.std(vy)+1e-12))
            score=nr+float(complexity_weight)*k
            if best is None or score<best[0]: best=(score,nr,idx,beta)
    _,nr,idx,beta=best
    selected=[terms[i] for i in idx]
    coeff={terms[i]:float(beta[j+1]) for j,i in enumerate(idx)}
    return AlgebraicRelationResult(str(target),list(inputs),'subset-ols',selected,coeff,float(beta[0]),float(nr),0.0,
        {'degree':int(degree),'max_terms':int(max_terms),'complexity_weight':float(complexity_weight),
         'train_rows':int(len(A)),'validation_rows':int(len(V))})
