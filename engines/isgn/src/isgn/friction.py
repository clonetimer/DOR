from __future__ import annotations

"""Reusable smooth dry-friction constitutive motif and fixed-topology profiler.

This module intentionally lives above discovery backends.  A sparse backend may
recover a stable source/target topology while Completion Certification reports a
localized Grammar gap.  ``profile_smooth_stribeck`` can then test a mature,
reusable friction constitutive family on an already-supported velocity edge,
without introducing device-specific truth channels or a new sparse solver.

The canonical motif is

    phi(v) = r_v * v
             + tanh(v / eps)
               * (1 + r_s * exp(-(abs(v) / v_s)^2))

and the equation contribution is ``c * phi(v)``.  ``c`` is the smoothed
Coulomb-scale coefficient, ``r_v`` the viscous/Coulomb ratio, and ``r_s`` the
Stribeck-excess/Coulomb ratio.  All parameters are inferred from public data.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, Mapping, Optional, Sequence, TYPE_CHECKING
import numpy as np

from .backends.base import CanonicalMechanismModel
from .contracts import PublicDataset

if TYPE_CHECKING:  # pragma: no cover
    from .api import RecoveryResult


def smooth_stribeck_transform(z: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    """Evaluate the canonical smooth Stribeck/Coulomb/viscous shape."""
    p=dict(params or {})
    eps=float(p.get("smoothing_velocity", p.get("eps", 0.0)))
    vs=float(p.get("breakaway_velocity", p.get("vs", 0.0)))
    rv=float(p.get("viscous_ratio", 0.0))
    rs=float(p.get("stribeck_ratio", 0.0))
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("SmoothStribeck smoothing_velocity must be finite and > 0")
    if not np.isfinite(vs) or vs <= 0:
        raise ValueError("SmoothStribeck breakaway_velocity must be finite and > 0")
    z=np.asarray(z,dtype=float)
    th=np.tanh(z/eps)
    return rv*z + th*(1.0 + rs*np.exp(-(np.abs(z)/vs)**2))


@dataclass
class SmoothStribeckProfile:
    target: str
    source: str
    smoothing_velocity: float
    breakaway_velocity: float
    validation_target_nrmse: float
    coefficient: float
    viscous_ratio: float
    stribeck_ratio: float
    expanded_coefficients: Dict[str, float]
    expected_signs: Dict[str, int]
    candidates_evaluated: int
    candidates_rejected_by_sign: int
    runner_up_validation_target_nrmse: Optional[float] = None
    selection_margin_relative: Optional[float] = None
    top_candidates: Optional[list[Dict[str,float]]] = None
    truth_channels_used: bool = False
    selection_metric: str = "target_derivative_nrmse"
    family: str = "SmoothStribeck"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _state_names(ds: PublicDataset) -> list[str]:
    return [v["name"] for v in ds.schema["states"]]


def _input_names(ds: PublicDataset) -> list[str]:
    return [v["name"] for v in ds.schema["inputs"]]


def _source_midpoint(ds: PublicDataset, name: str) -> np.ndarray:
    states=_state_names(ds); inputs=_input_names(ds)
    if name in states:
        j=states.index(name)
        return 0.5*(np.asarray(ds.X[:,:-1,j],dtype=float)+np.asarray(ds.X[:,1:,j],dtype=float))
    if name in inputs:
        return np.asarray(ds.U[:,:,inputs.index(name)],dtype=float)
    if name == "__constant__":
        return np.ones(ds.X.shape[:2][:-1] + (ds.X.shape[1]-1,),dtype=float)
    raise KeyError(name)


def _fit_scaled_ridge(A: np.ndarray, b: np.ndarray, ridge: float) -> np.ndarray:
    A=np.asarray(A,dtype=float); b=np.asarray(b,dtype=float)
    scale=np.sqrt(np.mean(A*A,axis=0)); scale=np.maximum(scale,1e-12)
    An=A/scale
    lhs=An.T@An+float(ridge)*np.eye(An.shape[1])
    cn=np.linalg.solve(lhs,An.T@b)
    return cn/scale


def _sign_ok(value: float, expected: int, floor: float=1e-12) -> bool:
    if expected == 0: return True
    if abs(float(value)) <= floor: return False
    return (value > 0) if expected > 0 else (value < 0)


def _fit_target_expanded(
    ds: PublicDataset,
    *,
    target: str,
    sources: Sequence[str],
    friction_source: str,
    smoothing_velocity: float,
    breakaway_velocity: float,
    ridge: float,
) -> tuple[Dict[str,float], np.ndarray]:
    states=_state_names(ds)
    if target not in states: raise KeyError(target)
    y=((np.asarray(ds.X[:,1:,states.index(target)],dtype=float)-np.asarray(ds.X[:,:-1,states.index(target)],dtype=float))/float(ds.dt)).reshape(-1)
    cols=[]; labels=[]
    for src in sources:
        z=_source_midpoint(ds,src)
        if src == friction_source:
            th=np.tanh(z/float(smoothing_velocity))
            st=np.exp(-(np.abs(z)/float(breakaway_velocity))**2)*th
            for suffix,col in (("viscous",z),("coulomb",th),("stribeck",st)):
                cols.append(col.reshape(-1)); labels.append(f"{src}:{suffix}")
        else:
            cols.append(z.reshape(-1)); labels.append(src)
    if not cols: return {},y
    A=np.stack(cols,axis=1)
    c=_fit_scaled_ridge(A,y,ridge)
    return {k:float(v) for k,v in zip(labels,c)},y


def _target_derivative_nrmse(model: CanonicalMechanismModel, ds: PublicDataset, target: str) -> float:
    states=_state_names(ds); ti=states.index(target)
    X=np.asarray(ds.X,dtype=float); U=np.asarray(ds.U,dtype=float)
    xm=.5*(X[:,:-1]+X[:,1:])
    yp=np.asarray(model.rhs_numpy(xm,U),dtype=float)[:,:,ti]
    yt=(X[:,1:,ti]-X[:,:-1,ti])/float(ds.dt)
    den=max(float(np.std(yt)),1e-6)
    return float(np.sqrt(np.mean((yt-yp)**2))/den)


def _fit_fixed_canonical_rows(
    ds: PublicDataset,
    *, graph: Mapping[str,Sequence[str]], operators: Mapping[str,Mapping[str,str]],
    params: Mapping[str,Mapping[str,Any]], ridge: float,
) -> Dict[str,Dict[str,float]]:
    states=_state_names(ds)
    X=np.asarray(ds.X,dtype=float); U=np.asarray(ds.U,dtype=float)
    xm=.5*(X[:,:-1]+X[:,1:])
    coeff: Dict[str,Dict[str,float]]={}
    probe=CanonicalMechanismModel(schema=ds.schema,dt=ds.dt,coefficients={},operators=operators,operator_parameters=params)
    for ti,target in enumerate(states):
        srcs=list(graph.get(target,[])); cols=[]
        for src in srcs:
            z=_source_midpoint(ds,src)
            op=operators.get(target,{}).get(src,"Identity")
            par=params.get(target,{}).get(src)
            cols.append(probe._transform(z,op,par).reshape(-1))
        if not cols:
            coeff[target]={}; continue
        A=np.stack(cols,axis=1)
        y=((X[:,1:,ti]-X[:,:-1,ti])/float(ds.dt)).reshape(-1)
        c=_fit_scaled_ridge(A,y,ridge)
        coeff[target]={s:float(v) for s,v in zip(srcs,c)}
    return coeff


def profile_smooth_stribeck(
    template: "RecoveryResult",
    train: PublicDataset,
    validation: PublicDataset,
    *,
    target: str,
    source: str,
    smoothing_grid: Sequence[float],
    breakaway_grid: Sequence[float],
    expected_signs: Optional[Mapping[str,int]]=None,
    ridge: float=1e-10,
) -> tuple["RecoveryResult", SmoothStribeckProfile]:
    """Profile a reusable smooth-friction motif on one existing graph edge.

    The graph topology is held fixed.  Only public state/input channels present
    in ``train`` and ``validation`` are used.  Candidate smoothing/breakaway
    scales are chosen by held-out target derivative NRMSE.  Optional sign
    constraints encode partial physical prior and are generic source-name
    constraints rather than device-specific truth.
    """
    from .api import RecoveryResult, RecoverySummary

    if _state_names(train) != _state_names(validation) or _input_names(train) != _input_names(validation):
        raise ValueError("train/validation schemas must match")
    if abs(float(train.dt)-float(validation.dt)) > max(1e-12,abs(float(train.dt))*1e-9):
        raise ValueError("train/validation dt must match")
    graph={t:list(v) for t,v in template.summary.graph.items()}
    if target not in graph or source not in graph[target]:
        raise ValueError(f"SmoothStribeck edge {source}->{target} must already exist in template graph")
    sgrid=sorted({float(x) for x in smoothing_grid if np.isfinite(x) and float(x)>0})
    vgrid=sorted({float(x) for x in breakaway_grid if np.isfinite(x) and float(x)>0})
    if not sgrid or not vgrid: raise ValueError("non-empty positive smoothing/breakaway grids are required")
    signs={str(k):int(v) for k,v in (expected_signs or {}).items()}
    if any(v not in (-1,0,1) for v in signs.values()): raise ValueError("expected_signs values must be -1, 0, or +1")

    base_ops={t:{s:template.summary.operators.get(t,{}).get(s,"Identity") for s in srcs} for t,srcs in graph.items()}
    base_params={t:{s:template.summary.operator_parameters.get(t,{}).get(s) for s in srcs} for t,srcs in graph.items()}
    best=None; evaluated=0; rejected=0; accepted_records=[]
    for eps in sgrid:
        for vs in vgrid:
            expanded,_=_fit_target_expanded(train,target=target,sources=graph[target],friction_source=source,
                                            smoothing_velocity=eps,breakaway_velocity=vs,ridge=ridge)
            c_c=float(expanded.get(f"{source}:coulomb",0.0))
            if abs(c_c) <= 1e-12 or not np.isfinite(c_c):
                rejected+=1; continue
            candidate_coeff={}
            for src in graph[target]:
                if src==source: candidate_coeff[src]=c_c
                else: candidate_coeff[src]=float(expanded[src])
            candidate_coeff[source]=c_c
            sign_fail=False
            for src,sgn in signs.items():
                if src not in candidate_coeff or not _sign_ok(candidate_coeff[src],sgn):
                    sign_fail=True; break
            if sign_fail:
                rejected+=1; continue
            rv=float(expanded[f"{source}:viscous"])/c_c
            rs=float(expanded[f"{source}:stribeck"])/c_c
            if not np.all(np.isfinite([rv,rs])):
                rejected+=1; continue
            ops={t:dict(row) for t,row in base_ops.items()}; pars={t:dict(row) for t,row in base_params.items()}
            ops[target][source]="SmoothStribeck"
            pars[target][source]={"smoothing_velocity":eps,"breakaway_velocity":vs,
                                  "viscous_ratio":rv,"stribeck_ratio":rs}
            # Fit every row in one common canonical coordinate system, while
            # preserving the profiled friction ratios for the target edge.
            coeff=_fit_fixed_canonical_rows(train,graph=graph,operators=ops,params=pars,ridge=ridge)
            # Re-fit of the compressed SmoothStribeck column can shift c_c. Keep
            # the expanded basis's c_c so the exact profiled shape is preserved.
            coeff[target][source]=c_c
            for src in graph[target]:
                if src!=source: coeff[target][src]=candidate_coeff[src]
            model=CanonicalMechanismModel(schema=train.schema,dt=train.dt,coefficients=coeff,operators=ops,operator_parameters=pars)
            err=_target_derivative_nrmse(model,validation,target)
            evaluated+=1
            accepted_records.append({"validation_target_nrmse":float(err),"smoothing_velocity":float(eps),"breakaway_velocity":float(vs)})
            key=(float(err),float(eps),float(vs))
            if best is None or key < best[0]:
                best=(key,model,coeff,ops,pars,expanded,rv,rs)
    if best is None:
        raise RuntimeError("no SmoothStribeck candidate satisfied the supplied constraints")
    key,model,coeff,ops,pars,expanded,rv,rs=best
    eps=float(key[1]); vs=float(key[2]); err=float(key[0])
    summary=RecoverySummary(graph=graph,operators=ops,coefficients=coeff,val_nrmse=err,
                            seed=int(template.summary.seed),constitutive_search=True,
                            backend="constitutive-profile",
                            operator_parameters=pars,
                            backend_metadata={
                                "solver":"fixed-topology-smooth-stribeck-profile+scaled-ridge",
                                "template_backend":template.summary.backend,
                                "target":target,"source":source,
                                "selection_metric":"target_derivative_nrmse",
                                "truth_channels_used":False,
                            })
    result=RecoveryResult(model=model,summary=summary,schema=train.schema,dt=float(train.dt))
    accepted_records=sorted(accepted_records,key=lambda r:(r["validation_target_nrmse"],r["smoothing_velocity"],r["breakaway_velocity"]))
    runner=float(accepted_records[1]["validation_target_nrmse"]) if len(accepted_records)>1 else None
    margin=None if runner is None else float((runner-err)/max(abs(err),1e-12))
    profile=SmoothStribeckProfile(target=target,source=source,smoothing_velocity=eps,breakaway_velocity=vs,
                                  validation_target_nrmse=err,coefficient=float(coeff[target][source]),
                                  viscous_ratio=float(rv),stribeck_ratio=float(rs),
                                  expanded_coefficients={k:float(v) for k,v in expanded.items()},
                                  expected_signs=signs,candidates_evaluated=evaluated,
                                  candidates_rejected_by_sign=rejected,
                                  runner_up_validation_target_nrmse=runner,selection_margin_relative=margin,
                                  top_candidates=accepted_records[:5])
    return result,profile
