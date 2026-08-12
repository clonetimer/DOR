#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mechanism and raw representations for M4."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, Mapping
import numpy as np

from .contracts import PublicDataset


@dataclass
class NominalMechanism:
    graph: Dict[str,List[str]]
    coefficients: Dict[str,Dict[str,float]]
    derivative_scale: Dict[str,float]
    state_names: List[str]
    input_names: List[str]
    feature_names: List[str]
    operators: Dict[str,Dict[str,str]] = field(default_factory=dict)
    operator_parameters: Dict[str,Dict[str,Any]] = field(default_factory=dict)


def _names(ds: PublicDataset):
    states=[x["name"] for x in ds.schema["states"]]
    inputs=[x["name"] for x in ds.schema["inputs"]]
    return states,inputs


def _series(ds: PublicDataset, traj: int, source: str, states: List[str], inputs: List[str]):
    if source in states:
        # Trapezoidal/midpoint state is the numerically consistent regressor for
        # a continuous-time mechanism observed at finite sampling interval.
        # Using only x_t created a known discretization residual floor in the
        # faster motor/MSD systems even when the mechanism was exactly correct.
        j=states.index(source)
        return 0.5*(ds.X[traj,:-1,j].astype(np.float64)+ds.X[traj,1:,j].astype(np.float64))
    # Inputs are generated/held piecewise constant over [t,t+dt).
    return ds.U[traj,:,inputs.index(source)].astype(np.float64)


def _transform_series(z: np.ndarray, op: str, param: Any):
    if op == "Identity": return z
    if op == "Saturation":
        if param is None: raise ValueError("Saturation parameter missing")
        return np.clip(z,-float(param),float(param))
    if op == "DeadZone":
        if param is None: raise ValueError("DeadZone parameter missing")
        c=float(param); return np.sign(z)*np.maximum(np.abs(z)-c,0.0)
    if op == "SmoothStribeck":
        from .friction import smooth_stribeck_transform
        if not isinstance(param,Mapping): raise ValueError("SmoothStribeck parameter mapping missing")
        return smooth_stribeck_transform(z,param)
    raise ValueError(f"unsupported PHM constitutive operator {op!r}")

def _design(ds: PublicDataset, traj: int, target: str, sources: List[str], states: List[str], inputs: List[str],
            operators: Optional[Mapping[str,Mapping[str,str]]]=None,
            operator_parameters: Optional[Mapping[str,Mapping[str,Any]]]=None):
    orow=(operators or {}).get(target,{})
    prow=(operator_parameters or {}).get(target,{})
    cols=[]
    for src in sources:
        z=_series(ds,traj,src,states,inputs)
        cols.append(_transform_series(z,orow.get(src,"Identity"),prow.get(src)))
    Phi=np.stack(cols,axis=1) if cols else np.zeros((ds.X.shape[1]-1,0),dtype=float)
    yi=np.diff(ds.X[traj,:,states.index(target)].astype(np.float64))/ds.dt
    return Phi,yi


def _ridge(Phi,y,lam=1e-8):
    A=Phi.T@Phi + lam*np.eye(Phi.shape[1])
    return np.linalg.solve(A,Phi.T@y)


def fit_nominal(ds: PublicDataset, graph: Dict[str,List[str]], *, operators=None, operator_parameters=None) -> NominalMechanism:
    states,inputs=_names(ds)
    coeff={}; scales={}; feat=[]
    for target,sources in graph.items():
        P=[]; Y=[]
        for k in range(len(ds.X)):
            p,y=_design(ds,k,target,sources,states,inputs,operators,operator_parameters); P.append(p);Y.append(y)
        P=np.concatenate(P,axis=0); Y=np.concatenate(Y,axis=0)
        th=_ridge(P,Y)
        coeff[target]={s:float(v) for s,v in zip(sources,th)}
        scales[target]=float(np.std(Y)+1e-9)
        for s in sources: feat.append(f"dtheta:{target}<-{s}")
        feat += [f"r0_mean:{target}",f"r0_rms:{target}",f"rfit_mean:{target}",f"rfit_rms:{target}"]
    return NominalMechanism(graph,coeff,scales,states,inputs,feat,operators or {},operator_parameters or {})


def mechanism_features(ds: PublicDataset, nominal: NominalMechanism) -> np.ndarray:
    rows=[]
    for k in range(len(ds.X)):
        f=[]
        for target,sources in nominal.graph.items():
            Phi,y=_design(ds,k,target,sources,nominal.state_names,nominal.input_names,nominal.operators,nominal.operator_parameters)
            theta0=np.array([nominal.coefficients[target][s] for s in sources])
            th=_ridge(Phi,y)
            den=np.maximum(np.abs(theta0),1e-5)
            f.extend(((th-theta0)/den).tolist())
            sc=nominal.derivative_scale[target]
            r0=(y-Phi@theta0)/sc; rf=(y-Phi@th)/sc
            f += [float(np.mean(r0)),float(np.sqrt(np.mean(r0*r0))),
                  float(np.mean(rf)),float(np.sqrt(np.mean(rf*rf)))]
        rows.append(f)
    return np.asarray(rows,np.float64)


def raw_trajectory_features(ds: PublicDataset, points: int=32) -> np.ndarray:
    # Direct baseline gets the raw trajectory shape, not fault-specific engineered statistics.
    T=ds.X.shape[1]
    idx=np.linspace(0,T-1,points).round().astype(int)
    # U has T-1 points; use nearest valid input index and append it to sampled states.
    uidx=np.minimum(idx,ds.U.shape[1]-1)
    rows=[]
    for k in range(len(ds.X)):
        z=np.concatenate([ds.X[k,idx],ds.U[k,uidx]],axis=1)
        rows.append(z.reshape(-1))
    return np.asarray(rows,np.float64)


def post_refit_residual_rms(ds: PublicDataset, nominal: NominalMechanism) -> np.ndarray:
    vals=[]
    for k in range(len(ds.X)):
        rr=[]
        for target,sources in nominal.graph.items():
            P,y=_design(ds,k,target,sources,nominal.state_names,nominal.input_names)
            th=_ridge(P,y); sc=nominal.derivative_scale[target]
            r=(y-P@th)/sc; rr.append(np.mean(r*r))
        vals.append(float(np.sqrt(np.mean(rr))))
    return np.asarray(vals,float)

@dataclass
class PHMHealthCalibration:
    """Normal-window calibration for mechanism health evidence.

    ``normal_abs_floor`` is an empirical false-positive envelope over the
    supplied normal windows. Features that cannot be estimated on the supplied
    windows (for example an unexcited input coefficient) are explicitly listed
    in ``blocked_features`` rather than silently imputed.
    """
    feature_names: List[str]
    center: Dict[str,float]
    normal_abs_floor: Dict[str,float]
    robust_scale: Dict[str,float]
    eligible_features: List[str]
    blocked_features: Dict[str,str]
    trajectory_count: int

    def to_dict(self) -> Dict[str,Any]:
        return {
            "feature_names":list(self.feature_names),
            "center":dict(self.center),
            "normal_abs_floor":dict(self.normal_abs_floor),
            "robust_scale":dict(self.robust_scale),
            "eligible_features":list(self.eligible_features),
            "blocked_features":dict(self.blocked_features),
            "trajectory_count":int(self.trajectory_count),
        }


def anchored_parameter_drift(ds: PublicDataset, nominal: NominalMechanism, *,
                             energy_floor: float=1e-20) -> tuple[np.ndarray,List[str]]:
    """Estimate one-edge-at-a-time coefficient drift around a frozen baseline.

    For edge ``j`` the other baseline coefficients remain fixed and only the
    partial residual projected onto column ``j`` is estimated. This avoids a
    common PHM failure mode where a fully joint local refit redistributes one
    physical parameter change over several correlated mechanism coefficients.

    If a source has no usable excitation in the window, its drift is returned
    as NaN. Callers must preserve that as an identifiability result; it must not
    be zero-filled and interpreted as "no drift".
    """
    names=[]
    for target,sources in nominal.graph.items():
        names.extend([f"adtheta:{target}<-{s}" for s in sources])
    rows=[]
    for k in range(len(ds.X)):
        f=[]
        for target,sources in nominal.graph.items():
            Phi,y=_design(ds,k,target,sources,nominal.state_names,nominal.input_names,
                          nominal.operators,nominal.operator_parameters)
            theta0=np.asarray([nominal.coefficients[target][s] for s in sources],dtype=float)
            residual=np.asarray(y-Phi@theta0,dtype=float)
            for j in range(len(sources)):
                x=np.asarray(Phi[:,j],dtype=float)
                den=float(x@x)
                f.append(float(x@residual/den) if np.isfinite(den) and den>energy_floor else np.nan)
        rows.append(f)
    return np.asarray(rows,dtype=np.float64),names


def mechanism_health_evidence(ds: PublicDataset, nominal: NominalMechanism, *,
                              energy_floor: float=1e-20) -> tuple[np.ndarray,List[str]]:
    """Return PHM evidence designed for a frozen certified normal mechanism.

    Evidence consists of anchored per-edge coefficient drift plus nominal and
    post-refit residual RMS for each target equation. Operator transforms are
    inherited from ``nominal``; therefore constitutive models such as
    SmoothStribeck remain active in both the baseline and local refit paths.
    """
    ad,ad_names=anchored_parameter_drift(ds,nominal,energy_floor=energy_floor)
    residual_names=[]; residual_rows=[]
    for target in nominal.graph:
        residual_names += [f"r0_rms:{target}",f"rfit_rms:{target}"]
    for k in range(len(ds.X)):
        r=[]
        for target,sources in nominal.graph.items():
            Phi,y=_design(ds,k,target,sources,nominal.state_names,nominal.input_names,
                          nominal.operators,nominal.operator_parameters)
            theta0=np.asarray([nominal.coefficients[target][s] for s in sources],dtype=float)
            th=_ridge(Phi,y)
            sc=max(float(nominal.derivative_scale[target]),1e-12)
            r0=(y-Phi@theta0)/sc; rf=(y-Phi@th)/sc
            r += [float(np.sqrt(np.mean(r0*r0))),float(np.sqrt(np.mean(rf*rf)))]
        residual_rows.append(r)
    # Interleave by target/edge order for readability rather than appending all
    # residuals after all edge drifts.
    out_names=[]; cols=[]; ad_col=0; rr_col=0
    for target,sources in nominal.graph.items():
        for source in sources:
            out_names.append(f"adtheta:{target}<-{source}"); cols.append(ad[:,ad_col]); ad_col+=1
        out_names += [f"r0_rms:{target}",f"rfit_rms:{target}"]
        rr=np.asarray(residual_rows,dtype=float)
        cols += [rr[:,rr_col],rr[:,rr_col+1]]; rr_col+=2
    return np.stack(cols,axis=1).astype(np.float64),out_names


def calibrate_phm_health(normal: PublicDataset, nominal: NominalMechanism, *,
                         energy_floor: float=1e-20, min_robust_scale: float=1e-12) -> PHMHealthCalibration:
    """Calibrate normal false-positive envelopes and feature eligibility."""
    feat,names=mechanism_health_evidence(normal,nominal,energy_floor=energy_floor)
    center={}; floor={}; scale={}; eligible=[]; blocked={}
    for j,name in enumerate(names):
        z=np.asarray(feat[:,j],dtype=float); finite=np.isfinite(z)
        if not np.all(finite) or int(finite.sum()) < 2:
            blocked[name]="WINDOW_UNIDENTIFIABLE_OR_NONFINITE"
            continue
        q=z[finite]; med=float(np.median(q)); mad=float(np.median(np.abs(q-med)))
        rs=max(1.4826*mad,float(np.std(q,ddof=1)) if len(q)>1 else 0.0,float(min_robust_scale))
        center[name]=med; floor[name]=float(np.max(np.abs(q-med))); scale[name]=rs; eligible.append(name)
    return PHMHealthCalibration(names,center,floor,scale,eligible,blocked,int(len(normal.X)))


def score_phm_health(data: PublicDataset, nominal: NominalMechanism,
                     calibration: PHMHealthCalibration, *, energy_floor: float=1e-20) -> tuple[np.ndarray,List[str]]:
    """Return signed normal-calibrated health scores for eligible evidence."""
    feat,names=mechanism_health_evidence(data,nominal,energy_floor=energy_floor)
    idx={n:i for i,n in enumerate(names)}; cols=[]
    for name in calibration.eligible_features:
        z=feat[:,idx[name]]
        cols.append((z-float(calibration.center[name]))/max(float(calibration.robust_scale[name]),1e-12))
    if not cols:
        return np.zeros((len(data.X),0),dtype=np.float64),[]
    return np.stack(cols,axis=1).astype(np.float64),list(calibration.eligible_features)

def slice_time_window(ds: PublicDataset, start_s: float, end_s: float) -> PublicDataset:
    """Return a copy of a fixed-rate PublicDataset restricted to [start_s,end_s]."""
    if not np.isfinite(start_s) or not np.isfinite(end_s) or start_s < 0 or end_s <= start_s:
        raise ValueError("require finite 0 <= start_s < end_s")
    i0=int(round(float(start_s)/float(ds.dt)))
    i1=int(round(float(end_s)/float(ds.dt)))
    i0=max(0,min(i0,ds.X.shape[1]-2)); i1=max(i0+1,min(i1,ds.X.shape[1]-1))
    return PublicDataset(X=np.asarray(ds.X[:,i0:i1+1]).copy(),U=np.asarray(ds.U[:,i0:i1]).copy(),
                         trajectory_id=np.asarray(ds.trajectory_id).copy(),schema=ds.schema,dt=float(ds.dt),
                         regime=ds.regime,variant=f"{ds.variant}-window-{start_s:g}-{end_s:g}")
