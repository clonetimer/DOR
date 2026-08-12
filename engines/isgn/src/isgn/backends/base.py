from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Mapping, Optional, Protocol
import numpy as np

from ..contracts import PublicDataset


@dataclass
class DiscoveryResult:
    graph: Dict[str, List[str]]
    coefficients: Dict[str, Dict[str, float]]
    operators: Dict[str, Dict[str, str]]
    operator_parameters: Dict[str, Dict[str, Any]]
    validation_error: float
    backend: str
    metadata: Dict[str, Any]
    model: Any


class DiscoveryBackend(Protocol):
    name: str
    @classmethod
    def available(cls) -> bool: ...
    def recover(self, train: PublicDataset, validation: PublicDataset, **kwargs) -> DiscoveryResult: ...
    def complete(self, train: PublicDataset, validation: PublicDataset,
                 *, known_present=None, known_absent=None, **kwargs) -> DiscoveryResult: ...


class CanonicalMechanismModel:
    """Solver-independent deployment model for sparse state equations.

    It represents the common v1 operational form
        d x_target / dt = sum_s coefficient[target,s] * phi_s(source_s)
    where phi is Identity/Saturation/DeadZone.  Discovery backends may be
    PySINDy, the native grammar solver, or another backend; certification and
    PHM operate on this canonical representation rather than a solver object.
    """
    def __init__(self, *, schema: Dict[str, Any], dt: float,
                 coefficients: Mapping[str, Mapping[str, float]],
                 operators: Optional[Mapping[str, Mapping[str, str]]] = None,
                 operator_parameters: Optional[Mapping[str, Mapping[str, Any]]] = None):
        self.schema = schema
        self.dt = float(dt)
        self._coefficients = {t: {s: float(v) for s, v in row.items()} for t, row in coefficients.items()}
        self._operators = {t: dict(row) for t, row in (operators or {}).items()}
        self._op_params = {t: dict(row) for t, row in (operator_parameters or {}).items()}
        self.allow_constant = False
        self.constitutive_search = any(op != "Identity" for row in self._operators.values() for op in row.values())

    @property
    def state_names(self):
        return [v["name"] for v in self.schema["states"]]

    @property
    def input_names(self):
        return [v["name"] for v in self.schema["inputs"]]

    def effective_coefficients(self) -> Dict[str, Dict[str, float]]:
        return {t: dict(row) for t, row in self._coefficients.items()}

    def _source(self, x: np.ndarray, u: np.ndarray, name: str) -> np.ndarray:
        if name in self.state_names:
            z = x[..., self.state_names.index(name)]
        elif name in self.input_names:
            z = u[..., self.input_names.index(name)]
        elif name == "__constant__":
            z = np.ones(x.shape[:-1], dtype=float)
        else:
            raise KeyError(name)
        return z

    def _transform(self, z: np.ndarray, op: str, param: Any) -> np.ndarray:
        if op == "Identity": return z
        if op == "Saturation":
            if param is None: raise ValueError("Saturation parameter missing")
            return np.clip(z, -float(param), float(param))
        if op == "DeadZone":
            if param is None: raise ValueError("DeadZone parameter missing")
            c=float(param); return np.sign(z)*np.maximum(np.abs(z)-c,0.0)
        if op == "SmoothStribeck":
            from ..friction import smooth_stribeck_transform
            if not isinstance(param, Mapping):
                raise ValueError("SmoothStribeck parameter mapping missing")
            return smooth_stribeck_transform(z,param)
        raise ValueError(f"unsupported canonical operator {op!r}")

    def rhs_numpy(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x=np.asarray(x,dtype=float); u=np.asarray(u,dtype=float)
        out=np.zeros_like(x,dtype=float)
        for ti,t in enumerate(self.state_names):
            row=self._coefficients.get(t,{})
            acc=np.zeros(x.shape[:-1],dtype=float)
            for s,c in row.items():
                op=self._operators.get(t,{}).get(s,"Identity")
                param=self._op_params.get(t,{}).get(s)
                acc += float(c)*self._transform(self._source(x,u,s),op,param)
            out[...,ti]=acc
        return out

    def one_step_numpy(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        h=self.dt
        k1=self.rhs_numpy(x,u)
        k2=self.rhs_numpy(x+0.5*h*k1,u)
        k3=self.rhs_numpy(x+0.5*h*k2,u)
        k4=self.rhs_numpy(x+h*k3,u)
        return x+h*(k1+2*k2+2*k3+k4)/6.0

    def rollout_numpy(self, x0: np.ndarray, u: np.ndarray) -> np.ndarray:
        x=np.asarray(x0,dtype=float).copy(); u=np.asarray(u,dtype=float)
        out=[x.copy()]
        for k in range(u.shape[-2]):
            x=self.one_step_numpy(x,u[...,k,:])
            out.append(x.copy())
        return np.stack(out,axis=-2)

    def to_dict(self) -> Dict[str, Any]:
        return {"schema":self.schema,"dt":self.dt,"coefficients":self._coefficients,
                "operators":self._operators,"operator_parameters":self._op_params}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CanonicalMechanismModel":
        return cls(schema=d["schema"],dt=d["dt"],coefficients=d["coefficients"],
                   operators=d.get("operators"),operator_parameters=d.get("operator_parameters"))


def rollout_nrmse_canonical(model: CanonicalMechanismModel, data: PublicDataset, max_steps: Optional[int]=200) -> float:
    steps=data.U.shape[1] if max_steps is None else min(int(max_steps),data.U.shape[1])
    pred=model.rollout_numpy(data.X[:,0].astype(float),data.U[:,:steps].astype(float))
    truth=data.X[:,:steps+1].astype(float)
    rmse=np.sqrt(np.mean((pred-truth)**2,axis=(0,1)))
    sd=np.std(truth,axis=(0,1)); sd=np.maximum(sd,1e-6)
    return float(np.mean(rmse/sd))
