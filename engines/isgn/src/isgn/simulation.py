#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent C1 four-system simulator.

Hard rule: this file does not import ISGN operators, grammar or model code.
Ground truth is integrated by a NumPy RK4 substep solver. Training only receives PublicDataset;
PrivateTruth is kept separately for evaluation.

Variants:
  clean         : fully observed linear state equations used by C1-A.
  constitutive  : C1-B2a; the *public command* passes through an unobserved actuator
                  Saturation or DeadZone before entering the same physical state equation.
  misspecified  : C1-X minimal test; thermal keeps actuator Saturation but also contains an
                  unmodelled cubic heat-loss term not present in the current grammar.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import math
import numpy as np


from .contracts import PublicDataset

@dataclass
class PrivateTruth:
    system: str
    parameters: Dict[str, float]
    graph: Dict[str, List[str]]
    coefficients: Dict[str, Dict[str, float]]
    modes: Dict[str, float]
    transforms: Dict[str, Dict[str, Dict[str, float | str]]] = field(default_factory=dict)
    variant: str = "clean"


SYSTEM_SCHEMAS: Dict[str, Dict] = {
    "thermal": {
        "states": [{"name": "T", "dim": "temperature"}],
        "inputs": [{"name": "P", "dim": "power"}, {"name": "Ta", "dim": "temperature", "role": "context"}],
    },
    "tank": {
        "states": [{"name": "h", "dim": "level"}],
        "inputs": [{"name": "qin", "dim": "volume_flow"}, {"name": "qout", "dim": "volume_flow"}],
    },
    "motor": {
        "states": [{"name": "i", "dim": "current"}, {"name": "w", "dim": "speed"}],
        "inputs": [{"name": "V", "dim": "voltage"}, {"name": "TL", "dim": "torque"}],
    },
    "msd": {
        "states": [{"name": "x", "dim": "displacement"}, {"name": "v", "dim": "velocity"}],
        "inputs": [{"name": "F", "dim": "force"}],
    },
}

PARAMS: Dict[str, Dict[str, float]] = {
    "thermal": dict(C=1.0, h=0.3),
    "tank": dict(A=1.2),
    "motor": dict(L=0.5, R=1.0, Ke=0.2, Kt=0.6, J=0.5, B=0.4),
    "msd": dict(m=1.0, c=0.6, k=4.0),
}

# Truth is deliberately outside the public schema.  Same primitive is reused across domains.
CONSTITUTIVE_PARAMS: Dict[str, Dict[str, float | str]] = {
    "thermal": {"op": "Saturation", "param": 2.2, "source": "P", "target": "T"},
    "tank": {"op": "DeadZone", "param": 0.30, "source": "qin", "target": "h"},
    "motor": {"op": "Saturation", "param": 3.0, "source": "V", "target": "i"},
    "msd": {"op": "DeadZone", "param": 0.25, "source": "F", "target": "v"},
}


def _multisine(t: np.ndarray, rng: np.random.RandomState, freqs, amps, bias=0.0):
    y = np.full_like(t, float(bias), dtype=np.float64)
    for f, a in zip(freqs, amps):
        y += a * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    return y


def _regime_scale(regime: str) -> Tuple[float, float, float]:
    if regime == "iid":
        return 1.0, 1.0, 1.0
    if regime == "near":
        return 1.35, 1.25, 1.4
    if regime == "far":
        return 1.9, 1.7, 2.0
    raise ValueError(regime)


def _input_trajectory(system: str, t: np.ndarray, rng: np.random.RandomState, regime: str, variant: str) -> np.ndarray:
    amp, freq, context = _regime_scale(regime)
    if system == "thermal":
        P = _multisine(t, rng, [0.025*freq, 0.08*freq, 0.22*freq], [0.9*amp, 0.45*amp, 0.2*amp], bias=rng.uniform(1.0, 2.5))
        Ta0 = rng.uniform(18.0, 22.0) if regime == "iid" else (rng.uniform(14.0, 18.0) if regime == "near" else rng.uniform(8.0, 14.0))
        Ta = np.full_like(t, Ta0 * (1.0 if regime == "iid" else context / max(context, 1.0)))
        return np.stack([P, Ta], -1)
    if system == "tank":
        if variant == "constitutive":
            # Command deliberately crosses the dead-zone threshold; qout remains an observed physical flow.
            qin = _multisine(t, rng, [0.04*freq, 0.13*freq, 0.31*freq], [0.32*amp, 0.15*amp, 0.08*amp], bias=rng.uniform(0.25, 0.55))
            qin = np.clip(qin, 0.0, None)
            qout = _multisine(t, rng, [0.055*freq, 0.17*freq], [0.05*amp, 0.025*amp], bias=rng.uniform(0.08, 0.16))
            qout = np.clip(qout, 0.02, None)
        else:
            qin = _multisine(t, rng, [0.04*freq, 0.13*freq, 0.31*freq], [0.25*amp, 0.13*amp, 0.08*amp], bias=rng.uniform(0.85, 1.25))
            qout = _multisine(t, rng, [0.055*freq, 0.17*freq], [0.20*amp, 0.10*amp], bias=rng.uniform(0.75, 1.15))
            qin = np.clip(qin, 0.15, None); qout = np.clip(qout, 0.15, None)
        return np.stack([qin, qout], -1)
    if system == "motor":
        V = _multisine(t, rng, [0.06*freq, 0.18*freq, 0.45*freq], [1.2*amp, 0.6*amp, 0.25*amp], bias=rng.uniform(2.0, 3.6))
        TL = _multisine(t, rng, [0.035*freq, 0.12*freq], [0.10*amp, 0.05*amp], bias=rng.uniform(0.12, 0.38))
        return np.stack([V, TL], -1)
    if system == "msd":
        F = _multisine(t, rng, [0.09*freq, 0.30*freq, 0.58*freq], [0.55*amp, 0.45*amp, 0.20*amp], bias=rng.uniform(-0.20, 0.20))
        return F[:, None]
    raise KeyError(system)


def _deadzone(x: float | np.ndarray, c: float):
    x = np.asarray(x)
    return np.sign(x) * np.maximum(np.abs(x) - c, 0.0)


def _physical_input(system: str, u_public: np.ndarray, variant: str) -> np.ndarray:
    if variant == "clean":
        return np.asarray(u_public, dtype=np.float64)
    if variant not in ("constitutive", "misspecified"):
        raise ValueError(variant)
    u = np.asarray(u_public, dtype=np.float64).copy()
    info = CONSTITUTIVE_PARAMS[system]
    c = float(info["param"])
    if system == "thermal":
        u[0] = np.clip(u[0], -c, c)
    elif system == "tank":
        u[0] = float(_deadzone(u[0], c))
    elif system == "motor":
        u[0] = np.clip(u[0], -c, c)
    elif system == "msd":
        u[0] = float(_deadzone(u[0], c))
    return u


def _rhs(system: str, x: np.ndarray, u_public: np.ndarray, variant: str = "clean") -> np.ndarray:
    p = PARAMS[system]
    u = _physical_input(system, u_public, variant)
    if system == "thermal":
        T = x[0]; P, Ta = u
        cubic = 0.004 * (T - Ta) ** 3 if variant == "misspecified" else 0.0
        return np.array([(P - p["h"] * (T - Ta) - cubic) / p["C"]], dtype=np.float64)
    if system == "tank":
        qin, qout = u
        return np.array([(qin - qout) / p["A"]], dtype=np.float64)
    if system == "motor":
        i, w = x; V, TL = u
        di = (V - p["R"]*i - p["Ke"]*w) / p["L"]
        dw = (p["Kt"]*i - p["B"]*w - TL) / p["J"]
        return np.array([di, dw], dtype=np.float64)
    if system == "msd":
        xpos, v = x; F = u[0]
        return np.array([v, (F - p["c"]*v - p["k"]*xpos) / p["m"]], dtype=np.float64)
    raise KeyError(system)


def _rk4_substep(system: str, x: np.ndarray, u: np.ndarray, h: float, variant: str) -> np.ndarray:
    k1 = _rhs(system, x, u, variant)
    k2 = _rhs(system, x + 0.5*h*k1, u, variant)
    k3 = _rhs(system, x + 0.5*h*k2, u, variant)
    k4 = _rhs(system, x + h*k3, u, variant)
    return x + h*(k1 + 2*k2 + 2*k3 + k4)/6.0


def _initial_state(system: str, rng: np.random.RandomState, u0: np.ndarray, regime: str, variant: str) -> np.ndarray:
    spread = {"iid": 1.0, "near": 1.5, "far": 2.2}[regime]
    if system == "thermal":
        return np.array([u0[1] + rng.uniform(3.0, 8.0)*spread])
    if system == "tank":
        if variant == "constitutive":
            return np.array([rng.uniform(1.5, 2.5)*spread])
        return np.array([rng.uniform(0.5, 1.0)*spread])
    if system == "motor":
        return np.array([rng.uniform(-0.2, 0.2)*spread, rng.uniform(0.0, 1.5)*spread])
    if system == "msd":
        return np.array([rng.uniform(-0.25, 0.25)*spread, rng.uniform(-0.1, 0.1)*spread])
    raise KeyError(system)


def private_truth(system: str, variant: str = "clean") -> PrivateTruth:
    p = dict(PARAMS[system])
    if system == "thermal":
        coeff = {"T": {"T": -p["h"]/p["C"], "P": 1.0/p["C"], "Ta": p["h"]/p["C"]}}
        graph = {"T": ["T", "P", "Ta"]}
        modes = {"tau": p["C"]/p["h"], "pole": -p["h"]/p["C"]}
    elif system == "tank":
        coeff = {"h": {"qin": 1.0/p["A"], "qout": -1.0/p["A"]}}
        graph = {"h": ["qin", "qout"]}; modes = {}
    elif system == "motor":
        coeff = {
            "i": {"i": -p["R"]/p["L"], "w": -p["Ke"]/p["L"], "V": 1.0/p["L"]},
            "w": {"i": p["Kt"]/p["J"], "w": -p["B"]/p["J"], "TL": -1.0/p["J"]},
        }
        graph = {"i": ["i", "w", "V"], "w": ["i", "w", "TL"]}
        A = np.array([[-p["R"]/p["L"], -p["Ke"]/p["L"]], [p["Kt"]/p["J"], -p["B"]/p["J"]]])
        ev = np.linalg.eigvals(A); modes = {"pole_real_max": float(np.max(ev.real))}
    else:
        coeff = {"x": {"v": 1.0}, "v": {"x": -p["k"]/p["m"], "v": -p["c"]/p["m"], "F": 1.0/p["m"]}}
        graph = {"x": ["v"], "v": ["x", "v", "F"]}
        wn = math.sqrt(p["k"]/p["m"]); zeta = p["c"]/(2*math.sqrt(p["k"]*p["m"]))
        modes = {"wn": wn, "zeta": zeta}

    transforms: Dict[str, Dict[str, Dict[str, float | str]]] = {}
    if variant in ("constitutive", "misspecified"):
        info = CONSTITUTIVE_PARAMS[system]
        transforms = {
            str(info["target"]): {
                str(info["source"]): {"op": str(info["op"]), "param": float(info["param"])}
            }
        }
    if variant == "misspecified" and system == "thermal":
        p["gamma_cubic_unmodelled"] = 0.004
    return PrivateTruth(system=system, parameters=p, graph=graph, coefficients=coeff, modes=modes,
                        transforms=transforms, variant=variant)


def simulate(
    system: str,
    n: int,
    T: int = 201,
    dt: float = 0.05,
    seed: int = 0,
    regime: str = "iid",
    substeps: int = 5,
    variant: str = "clean",
) -> Tuple[PublicDataset, PrivateTruth]:
    rng = np.random.RandomState(seed)
    D = len(SYSTEM_SCHEMAS[system]["states"]); Ud = len(SYSTEM_SCHEMAS[system]["inputs"])
    X = np.zeros((n, T, D), dtype=np.float32); U = np.zeros((n, T-1, Ud), dtype=np.float32)
    t = np.arange(T-1, dtype=np.float64) * dt
    for k in range(n):
        uk = _input_trajectory(system, t, rng, regime, variant)
        x = _initial_state(system, rng, uk[0], regime, variant)
        X[k, 0] = x.astype(np.float32); U[k] = uk.astype(np.float32)
        h = dt / substeps
        for j in range(T-1):
            for _ in range(substeps):
                x = _rk4_substep(system, x, uk[j], h, variant)
            X[k, j+1] = x.astype(np.float32)
    public = PublicDataset(X=X, U=U, trajectory_id=np.arange(n, dtype=np.int64),
                           schema=SYSTEM_SCHEMAS[system], dt=dt, regime=regime, variant=variant)
    return public, private_truth(system, variant=variant)
