#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
import numpy as np, torch
from .contracts import PublicDataset

@dataclass
class CompletionCertificate:
    status: str
    prior_conflicts: List[str]
    coverage_ratio: float
    identifiable: bool
    rank: int
    n_columns: int
    condition_number: float
    evidence: Dict[str, Any]
    def to_dict(self): return asdict(self)


@dataclass
class CoverageCalibration:
    """Independent measurement/numerical residual floor from replicate acquisitions.

    Replicate groups must contain repeated measurements of the same underlying
    physical trajectory/condition.  The floor is estimated from pairwise
    residual differences, so deterministic model bias shared by the replicates
    cancels instead of being absorbed into the calibration floor.
    """
    method: str
    quantile: float
    target_floors: Dict[str, float]
    target_pair_median: Dict[str, float]
    target_pair_max: Dict[str, float]
    target_scales: Dict[str, float]
    replicate_groups: int
    replicate_trajectories: int
    pair_count: int
    assumptions: Dict[str, Any]
    def to_dict(self): return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CoverageCalibration":
        return cls(
            method=str(d["method"]), quantile=float(d["quantile"]),
            target_floors={str(k):float(v) for k,v in d["target_floors"].items()},
            target_pair_median={str(k):float(v) for k,v in d.get("target_pair_median",{}).items()},
            target_pair_max={str(k):float(v) for k,v in d.get("target_pair_max",{}).items()},
            target_scales={str(k):float(v) for k,v in d.get("target_scales",{}).items()},
            replicate_groups=int(d.get("replicate_groups",0)),
            replicate_trajectories=int(d.get("replicate_trajectories",0)),
            pair_count=int(d.get("pair_count",0)), assumptions=dict(d.get("assumptions",{})),
        )


def equation_residual_tensor(model, public: PublicDataset, device='cpu') -> tuple[np.ndarray, np.ndarray]:
    """Return residual and observed derivative tensors with shape [N,T-1,D]."""
    X=np.asarray(public.X,dtype=np.float64); U=np.asarray(public.U,dtype=np.float64)
    truth=(X[:,1:]-X[:,:-1])/float(public.dt)
    if hasattr(model,'rhs_numpy'):
        xm=.5*(X[:,:-1]+X[:,1:])
        pred=np.asarray(model.rhs_numpy(xm,U),dtype=np.float64)
    else:
        xt=torch.as_tensor(X[:,:-1],dtype=torch.float32,device=device)
        ut=torch.as_tensor(U,dtype=torch.float32,device=device)
        with torch.no_grad(): pred=np.asarray(model.rhs(xt,ut).detach().cpu().numpy(),dtype=np.float64)
    return truth-pred, truth


def target_residual_nrmse(model, public: PublicDataset, device='cpu', *, scales: Optional[Mapping[str,float]]=None) -> Dict[str,float]:
    residual, truth=equation_residual_tensor(model,public,device=device)
    names=[v["name"] for v in public.schema["states"]]
    if scales is None:
        den=np.maximum(np.std(truth,axis=(0,1)),1e-6)
    else:
        missing=[n for n in names if n not in scales]
        if missing: raise ValueError(f"coverage scales missing target(s): {missing}")
        den=np.asarray([max(float(scales[n]),1e-12) for n in names],dtype=float)
    rms=np.sqrt(np.mean(residual*residual,axis=(0,1)))
    return {n:float(rms[i]/den[i]) for i,n in enumerate(names)}


def calibrate_replicate_coverage(model, replicate_groups: Sequence[PublicDataset], *,
                                 quantile: float=0.95, min_floor: float=1e-6,
                                 device: str='cpu') -> CoverageCalibration:
    """Calibrate per-target residual floors from repeated measurement realizations.

    Each PublicDataset is one physical-condition group and must contain at least
    two replicate trajectories.  Inputs that are *not* measurement drivers must
    be identical across replicates. Measurement-driver inputs are allowed to
    differ because their sensor noise is part of the measurement chain being
    calibrated.
    """
    groups=list(replicate_groups)
    if not groups: raise ValueError("at least one replicate group is required")
    if not (0.5 <= float(quantile) <= 1.0): raise ValueError("quantile must be in [0.5,1.0]")
    names=[v["name"] for v in groups[0].schema["states"]]
    residuals=[]; truths=[]; total_n=0; pair_count=0
    for gi,ds in enumerate(groups):
        if int(ds.X.shape[0]) < 2: raise ValueError(f"replicate group {gi} needs at least two trajectories")
        if [v["name"] for v in ds.schema["states"]] != names: raise ValueError("replicate groups must have identical state schema")
        if abs(float(ds.dt)-float(groups[0].dt)) > max(1e-12,abs(float(groups[0].dt))*1e-9): raise ValueError("replicate groups must have identical dt")
        # Exact exogenous commands/contexts must be shared. Public measurement
        # drivers promoted into U are intentionally allowed to differ by noise.
        input_meta=list(ds.schema.get("inputs",[]))
        fixed_idx=[i for i,m in enumerate(input_meta) if not bool(m.get("promoted_from_state",False))]
        if fixed_idx:
            base=np.asarray(ds.U[0,:,fixed_idx],dtype=float)
            for ri in range(1,ds.U.shape[0]):
                if not np.allclose(np.asarray(ds.U[ri,:,fixed_idx],dtype=float),base,rtol=1e-10,atol=1e-12):
                    raise ValueError(f"replicate group {gi} changes exogenous input/context across repeats")
        r,y=equation_residual_tensor(model,ds,device=device)
        residuals.append(r); truths.append(y); total_n += int(ds.X.shape[0])
        pair_count += int(ds.X.shape[0])*(int(ds.X.shape[0])-1)//2
    Y=np.concatenate(truths,axis=0)
    scale=np.maximum(np.std(Y,axis=(0,1)),1e-6)
    pair_values=[[] for _ in names]
    for r in residuals:
        n=r.shape[0]
        for i in range(n):
            for j in range(i+1,n):
                d=(r[i]-r[j])/np.sqrt(2.0)
                vals=np.sqrt(np.mean(d*d,axis=0))/scale
                for k,v in enumerate(vals): pair_values[k].append(float(v))
    floors={}; med={}; mx={}; scales={}
    for k,n in enumerate(names):
        vals=np.asarray(pair_values[k],dtype=float)
        if vals.size==0: raise ValueError(f"no replicate pairs available for target {n!r}")
        floors[n]=float(max(np.quantile(vals,float(quantile)),float(min_floor)))
        med[n]=float(np.median(vals)); mx[n]=float(np.max(vals)); scales[n]=float(scale[k])
    return CoverageCalibration(
        method="paired_residual_difference",
        quantile=float(quantile), target_floors=floors,
        target_pair_median=med, target_pair_max=mx, target_scales=scales,
        replicate_groups=len(groups), replicate_trajectories=total_n, pair_count=pair_count,
        assumptions={
            "same_underlying_physical_trajectory_within_group": True,
            "deterministic_model_bias_cancelled_by_pairwise_difference": True,
            "measurement_driver_noise_included": True,
            "truth_channels_required": False,
        },
    )

def equation_residual_nrmse(model, public: PublicDataset, device='cpu') -> float:
    by_target=target_residual_nrmse(model,public,device=device)
    return float(np.mean(list(by_target.values())))

def source_identifiability(public: PublicDataset, columns: List[tuple[str,int]]) -> Dict[str,float|int|bool]:
    # columns entries: ('state'|'input', index). This is a direct observation-design audit.
    mats=[]
    for kind,idx in columns:
        arr=public.X[:,:-1,idx] if kind=='state' else public.U[:,:,idx]
        mats.append(np.asarray(arr,dtype=np.float64).reshape(-1))
    A=np.stack(mats,axis=1)
    A=A-A.mean(axis=0,keepdims=True)
    scale=A.std(axis=0,keepdims=True); scale[scale<1e-12]=1.0
    A=A/scale
    s=np.linalg.svd(A,compute_uv=False)
    tol=max(A.shape)*np.finfo(float).eps*(s[0] if len(s) else 1.0)
    rank=int(np.sum(s>tol)); cond=float(np.inf if len(s)==0 or s[-1]<=tol else s[0]/s[-1])
    return {'identifiable': bool(rank==A.shape[1]), 'rank':rank,'n_columns':int(A.shape[1]),'condition_number':cond,
            'min_singular':float(s[-1] if len(s) else 0.0),'max_singular':float(s[0] if len(s) else 0.0)}

def prior_support_by_coefficient(model, declared_present: Dict[str,List[str]], relative_floor: float=0.03) -> List[str]:
    coeff=model.effective_coefficients(); conflicts=[]
    for target,names in declared_present.items():
        row=coeff.get(target,{})
        mx=max([abs(float(v)) for v in row.values()]+[1e-12])
        for name in names:
            v=abs(float(row.get(name,0.0)))
            if v < relative_floor*mx:
                conflicts.append(f'{name}->{target}')
    return conflicts

def decide(*, prior_conflicts:List[str], coverage_ratio:float, identifiable:bool, rank:int, n_columns:int,
           condition_number:float, coverage_threshold:float=8.0, evidence:Optional[Dict[str,float]]=None) -> CompletionCertificate:
    if prior_conflicts: status='PRIOR_CONFLICT'
    elif coverage_ratio >= coverage_threshold: status='GRAMMAR_UNCOVERED'
    elif not identifiable: status='UNIDENTIFIABLE'
    else: status='COMPLETE'
    return CompletionCertificate(status,prior_conflicts,float(coverage_ratio),bool(identifiable),int(rank),int(n_columns),float(condition_number),evidence or {})
