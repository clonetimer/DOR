"""Physics-aware search-space reduction for mechanism recovery.

This module does not implement AI Feynman.  It absorbs two reusable ideas that
are especially valuable for industrial mechanism discovery:

1. dimensional consistency can prune algebraic candidates before symbolic
   search; and
2. simple symmetry coordinates (difference/sum/ratio/product) can replace two
   raw variables when the transformed problem explains held-out data with a
   simpler expression.

Every proposed reduction is evidence, not a discovered law.  Downstream
recovery/certification must still validate the resulting mechanism.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from itertools import product
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple
import numpy as np

from .contracts import AlgebraicDataset
from .operators import Dim
from .algebraic import discover_algebraic_relation_subset

DimVec = Tuple[int, int, int, int, int]


def _dim(name: str) -> DimVec | None:
    if name in ("dimensionless", "none"):
        return tuple(Dim.get("none"))
    if name not in Dim.VECS:
        return None
    return tuple(Dim.get(name))


def _var_meta(ds: AlgebraicDataset) -> Dict[str, Mapping[str, object]]:
    ds.validate()
    return {str(v["name"]): v for v in ds.schema.get("variables", [])}


def _var_dim(ds: AlgebraicDataset, name: str) -> DimVec | None:
    meta = _var_meta(ds).get(str(name))
    if meta is None:
        raise KeyError(f"unknown algebraic variable {name!r}")
    return _dim(str(meta.get("dim", "")))


def _dim_add_scaled(acc: Sequence[int], d: Sequence[int], exponent: int) -> DimVec:
    return tuple(int(a + exponent * b) for a, b in zip(acc, d))  # type: ignore[return-value]


def _format_monomial(inputs: Sequence[str], exponents: Sequence[int]) -> str:
    num: List[str] = []; den: List[str] = []
    for name, e in zip(inputs, exponents):
        if e == 0: continue
        target = num if e > 0 else den
        p = abs(int(e))
        target.append(str(name) if p == 1 else f"{name}^{p}")
    n = "*".join(num) if num else "1"
    if not den: return n
    d = "*".join(den)
    return f"({n})/({d})"


@dataclass(frozen=True)
class DimensionalCandidate:
    expression: str
    exponents: Dict[str, int]
    dimension: DimVec
    total_order: int
    coefficient_dimensionless: bool = True
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class PairwiseReduction:
    variables: Tuple[str, str]
    transform: str
    derived_name: str
    expression: str
    baseline_validation_nrmse: float
    reduced_validation_nrmse: float
    baseline_terms: Tuple[str, ...]
    reduced_terms: Tuple[str, ...]
    baseline_score: float
    reduced_score: float
    score_gain: float
    dimension: DimVec | None
    def to_dict(self): return asdict(self)


@dataclass
class PhysicsReductionReport:
    target: str
    inputs: List[str]
    dimensional_candidates: List[DimensionalCandidate]
    pairwise_reductions: List[PairwiseReduction]
    metadata: Dict[str, object]
    def to_dict(self):
        return {
            "target": self.target,
            "inputs": list(self.inputs),
            "dimensional_candidates": [x.to_dict() for x in self.dimensional_candidates],
            "pairwise_reductions": [x.to_dict() for x in self.pairwise_reductions],
            "metadata": dict(self.metadata),
        }


def dimensionally_consistent_monomials(
    ds: AlgebraicDataset,
    *,
    inputs: Sequence[str],
    target: str,
    max_total_order: int = 3,
    allow_inverse: bool = False,
) -> List[DimensionalCandidate]:
    """Enumerate low-order monomials whose physical dimension matches target.

    This is strict Buckingham/AI-Feynman-style pruning: it assumes the unknown
    multiplicative coefficient is dimensionless.  If a dimensional physical
    constant is omitted from the inputs, callers must not use the returned list
    as a hard exclusion rule.
    """
    ds.validate(); inputs=[str(x) for x in inputs]
    tdim=_var_dim(ds,target)
    dims=[_var_dim(ds,x) for x in inputs]
    if tdim is None or any(d is None for d in dims):
        return []
    lo=-int(max_total_order) if allow_inverse else 0
    hi=int(max_total_order)
    out=[]
    for exps in product(range(lo,hi+1),repeat=len(inputs)):
        if not any(exps): continue
        order=sum(abs(int(x)) for x in exps)
        if order>int(max_total_order): continue
        acc=(0,0,0,0,0)
        for d,e in zip(dims,exps): acc=_dim_add_scaled(acc,d,int(e))  # type: ignore[arg-type]
        if tuple(acc)==tuple(tdim):
            out.append(DimensionalCandidate(
                expression=_format_monomial(inputs,exps),
                exponents={inputs[i]:int(exps[i]) for i in range(len(inputs)) if exps[i]!=0},
                dimension=tuple(acc), total_order=int(order)))
    out.sort(key=lambda x:(x.total_order,x.expression))
    return out


def _transform_arrays(a: np.ndarray, b: np.ndarray, kind: str) -> np.ndarray:
    if kind=="difference": return a-b
    if kind=="sum": return a+b
    if kind=="ratio":
        eps=max(1e-12,1e-9*float(np.nanmedian(np.abs(b))+1.0))
        return a/(np.where(np.abs(b)>eps,b,np.sign(b)*eps + (b==0)*eps))
    if kind=="product": return a*b
    raise KeyError(kind)


def _transform_dim(da: DimVec | None, db: DimVec | None, kind: str) -> DimVec | None:
    if da is None or db is None: return None
    if kind in ("difference","sum"):
        return da if da==db else None
    if kind=="ratio": return tuple(int(x-y) for x,y in zip(da,db))
    if kind=="product": return tuple(int(x+y) for x,y in zip(da,db))
    return None


def _dim_name_from_vec(vec: DimVec | None) -> str:
    if vec is None: return "generic"
    for k,v in Dim.VECS.items():
        if tuple(v)==tuple(vec):
            return "dimensionless" if k=="none" else str(k)
    return "generic"


def _replace_pair(ds: AlgebraicDataset, a: str, b: str, kind: str, derived: str) -> AlgebraicDataset:
    names=ds.variable_names; ia=names.index(a); ib=names.index(b)
    z=_transform_arrays(ds.values[:,ia],ds.values[:,ib],kind)
    keep=[n for n in names if n not in (a,b)]
    cols=[]; metas=[]; source_meta=_var_meta(ds)
    for n in keep:
        cols.append(ds.values[:,names.index(n)]); metas.append(dict(source_meta[n]))
    # derived variable is appended; target is retained because it is in keep.
    da,db=_var_dim(ds,a),_var_dim(ds,b); dv=_transform_dim(da,db,kind)
    cols.append(z); metas.append({"name":derived,"dim":_dim_name_from_vec(dv),"derived_from":[a,b],"transform":kind})
    A=np.stack(cols,axis=1)
    schema=dict(ds.schema); schema["variables"]=metas
    return AlgebraicDataset(A,np.asarray(ds.sample_id),schema,regime=ds.regime,variant=ds.variant,metadata=dict(ds.metadata))


def probe_pairwise_reductions(
    train: AlgebraicDataset,
    validation: AlgebraicDataset,
    *,
    inputs: Sequence[str],
    target: str,
    degree: int = 2,
    max_terms: int = 3,
    complexity_weight: float = 1e-3,
    min_score_gain: float = 1e-6,
) -> List[PairwiseReduction]:
    """Find simpler pairwise coordinates supported by held-out relation fitting.

    A proposal is retained only when it improves the same validation+complexity
    objective used for the baseline relation.  This keeps symmetry reduction an
    evidence-backed search-space simplification rather than an automatic law.
    """
    train.validate(); validation.validate(); inputs=[str(x) for x in inputs]
    base=discover_algebraic_relation_subset(train,validation,inputs=inputs,target=target,degree=degree,max_terms=max_terms,complexity_weight=complexity_weight)
    bscore=float(base.validation_nrmse)+float(complexity_weight)*len(base.selected_terms)
    out=[]
    for i,a in enumerate(inputs):
        for b in inputs[i+1:]:
            da,db=_var_dim(train,a),_var_dim(train,b)
            kinds=["product"]
            if da is not None and db is not None and da==db:
                kinds=["difference","sum","ratio","product"]
            for kind in kinds:
                derived=f"{kind}__{a}__{b}"
                tr=_replace_pair(train,a,b,kind,derived); va=_replace_pair(validation,a,b,kind,derived)
                reduced_inputs=[x for x in inputs if x not in (a,b)]+[derived]
                try:
                    r=discover_algebraic_relation_subset(tr,va,inputs=reduced_inputs,target=target,degree=degree,max_terms=max_terms,complexity_weight=complexity_weight)
                except Exception:
                    continue
                rscore=float(r.validation_nrmse)+float(complexity_weight)*len(r.selected_terms)
                gain=bscore-rscore
                if np.isfinite(rscore) and gain>float(min_score_gain):
                    expr={"difference":f"{a}-{b}","sum":f"{a}+{b}","ratio":f"{a}/{b}","product":f"{a}*{b}"}[kind]
                    out.append(PairwiseReduction(
                        variables=(a,b),transform=kind,derived_name=derived,expression=expr,
                        baseline_validation_nrmse=float(base.validation_nrmse),reduced_validation_nrmse=float(r.validation_nrmse),
                        baseline_terms=tuple(base.selected_terms),reduced_terms=tuple(r.selected_terms),
                        baseline_score=bscore,reduced_score=rscore,score_gain=float(gain),dimension=_transform_dim(da,db,kind)))
    out.sort(key=lambda x:(-x.score_gain,x.reduced_validation_nrmse,x.expression))
    return out


def reduce_algebraic_search_space(
    train: AlgebraicDataset,
    validation: AlgebraicDataset,
    *, inputs: Sequence[str], target: str,
    max_total_order: int = 3,
    allow_inverse: bool = False,
    degree: int = 2,
    max_terms: int = 3,
) -> PhysicsReductionReport:
    dims=dimensionally_consistent_monomials(train,inputs=inputs,target=target,max_total_order=max_total_order,allow_inverse=allow_inverse)
    pairs=probe_pairwise_reductions(train,validation,inputs=inputs,target=target,degree=degree,max_terms=max_terms)
    return PhysicsReductionReport(
        target=str(target), inputs=[str(x) for x in inputs], dimensional_candidates=dims, pairwise_reductions=pairs,
        metadata={
            "role":"pre-discovery search-space reduction",
            "not_a_discovery_certificate":True,
            "dimensional_pruning_assumption":"unknown multiplicative coefficient is dimensionless",
            "pairwise_validation":"held-out relation fit plus complexity penalty",
            "inspiration":"physics-inspired dimensional/symmetry reduction; not an AI Feynman implementation",
        })
