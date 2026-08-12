#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent M4 PHM fault generator.

This module deliberately imports no ISGN model/grammar code.  It reuses only the
public system schemas / input-trajectory utilities of the frozen independent
simulator, then integrates faulted physics with its own NumPy RK4 routine.

M4 contains no new normal-system primitive.  Parameter faults alter coefficients
already present in the normal graph.  Relation-breaking faults add an unmeasured
term that is intentionally outside the frozen normal graph.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

from .simulation import (
    PublicDataset, SYSTEM_SCHEMAS, PARAMS, _input_trajectory, _initial_state,
)


@dataclass
class FaultTruth:
    system: str
    fault_class: str       # normal | param | relation
    severity: float
    target_parameter: str | None
    true_relative_drift: float


def fault_metadata(system: str, fault_class: str, severity: float) -> FaultTruth:
    if fault_class == "normal":
        return FaultTruth(system, fault_class, 0.0, None, 0.0)
    if fault_class == "param":
        if system == "thermal": return FaultTruth(system, fault_class, severity, "h", +severity)
        if system == "tank":    return FaultTruth(system, fault_class, severity, "A", +severity)
        if system == "motor":   return FaultTruth(system, fault_class, severity, "B", +severity)
        if system == "msd":     return FaultTruth(system, fault_class, severity, "k", -0.5*severity)
    if fault_class == "relation":
        return FaultTruth(system, fault_class, severity, None, 0.0)
    raise ValueError((system, fault_class, severity))


def _fault_params(system: str, fault_class: str, severity: float) -> Dict[str, float]:
    p = dict(PARAMS[system])
    if fault_class != "param":
        return p
    s = float(severity)
    if system == "thermal": p["h"] *= (1.0 + s)
    elif system == "tank":  p["A"] *= (1.0 + s)
    elif system == "motor": p["B"] *= (1.0 + s)
    elif system == "msd":   p["k"] *= (1.0 - 0.5*s)
    else: raise KeyError(system)
    return p


def _rhs_fault(system: str, x: np.ndarray, u: np.ndarray, p: Dict[str,float], fault_class: str, severity: float) -> np.ndarray:
    s = float(severity)
    if system == "thermal":
        T=x[0]; P,Ta=u
        d=(P-p["h"]*(T-Ta))/p["C"]
        if fault_class == "relation":
            # Unmodelled radiation-like heat loss. Not added to the ISGN Grammar.
            d -= (0.020*s)*(T-Ta)**3 / p["C"]
        return np.array([d], dtype=np.float64)
    if system == "tank":
        h=x[0]; qin,qout=u
        d=(qin-qout)/p["A"]
        if fault_class == "relation":
            # Unmeasured nonlinear leak.
            d -= (0.40*s)*np.sqrt(max(h,1e-8))/p["A"]
        return np.array([d], dtype=np.float64)
    if system == "motor":
        i,w=x; V,TL=u
        di=(V-p["R"]*i-p["Ke"]*w)/p["L"]
        dw=(p["Kt"]*i-p["B"]*w-TL)/p["J"]
        if fault_class == "relation":
            # Unmeasured nonlinear load/drag, local to the mechanical equation.
            dw -= (0.080*s)*w*abs(w)/p["J"]
        return np.array([di,dw], dtype=np.float64)
    if system == "msd":
        xpos,v=x; F=u[0]
        dx=v
        dv=(F-p["c"]*v-p["k"]*xpos)/p["m"]
        if fault_class == "relation":
            # Unmodelled cubic stiffness term.
            dv -= (8.0*s)*(xpos**3)/p["m"]
        return np.array([dx,dv], dtype=np.float64)
    raise KeyError(system)


def _rk4(system, x, u, h, p, fault_class, severity):
    k1=_rhs_fault(system,x,u,p,fault_class,severity)
    k2=_rhs_fault(system,x+0.5*h*k1,u,p,fault_class,severity)
    k3=_rhs_fault(system,x+0.5*h*k2,u,p,fault_class,severity)
    k4=_rhs_fault(system,x+h*k3,u,p,fault_class,severity)
    return x+h*(k1+2*k2+2*k3+k4)/6.0


def simulate_fault(
    system: str,
    fault_class: str,
    severity: float,
    n: int,
    T: int = 161,
    dt: float = 0.05,
    seed: int = 0,
    regime: str = "iid",
    substeps: int = 5,
) -> Tuple[PublicDataset, list[FaultTruth]]:
    if fault_class not in ("normal","param","relation"):
        raise ValueError(fault_class)
    rng=np.random.RandomState(seed)
    D=len(SYSTEM_SCHEMAS[system]["states"]); Ud=len(SYSTEM_SCHEMAS[system]["inputs"])
    X=np.zeros((n,T,D),np.float32); U=np.zeros((n,T-1,Ud),np.float32)
    t=np.arange(T-1,dtype=np.float64)*dt
    p=_fault_params(system,fault_class,severity)
    truth=fault_metadata(system,fault_class,severity)
    truths=[]
    for k in range(n):
        uk=_input_trajectory(system,t,rng,regime,"clean")
        x=_initial_state(system,rng,uk[0],regime,"clean")
        X[k,0]=x.astype(np.float32); U[k]=uk.astype(np.float32)
        h=dt/substeps
        for j in range(T-1):
            for _ in range(substeps):
                x=_rk4(system,x,uk[j],h,p,fault_class,severity)
            X[k,j+1]=x.astype(np.float32)
        truths.append(truth)
    public=PublicDataset(X=X,U=U,trajectory_id=np.arange(n,dtype=np.int64),
                         schema=SYSTEM_SCHEMAS[system],dt=dt,regime=regime,
                         variant=f"m4_{fault_class}")
    return public, truths


def concat_public(parts: list[PublicDataset]) -> PublicDataset:
    if not parts: raise ValueError("empty parts")
    base=parts[0]
    X=np.concatenate([p.X for p in parts],axis=0)
    U=np.concatenate([p.U for p in parts],axis=0)
    return PublicDataset(X=X,U=U,trajectory_id=np.arange(len(X),dtype=np.int64),
                         schema=base.schema,dt=base.dt,regime=base.regime,variant="m4_concat")
