from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Protocol, Sequence
import numpy as np


@dataclass
class PublicDataset:
    """Device-agnostic fixed-rate observation contract used by ISGN core.

    X has shape [trajectory, time, state], U has shape [trajectory, time-1, input].
    The schema carries semantic roles, dimensions and observation metadata; core
    algorithms must not infer device identity from variable names.

    ``sample_time_s`` is optional traceable observation time in seconds with shape
    [trajectory, time].  It may be absolute or trajectory-local; only differences
    are interpreted by the Observation Gate.  The scalar ``dt`` remains the
    nominal fixed-rate interval used by existing solvers.
    """
    X: np.ndarray
    U: np.ndarray
    trajectory_id: np.ndarray
    schema: Dict[str, Any]
    dt: float
    regime: str
    variant: str = "user"
    sample_time_s: np.ndarray | None = None

    def validate(self) -> None:
        X=np.asarray(self.X); U=np.asarray(self.U); ids=np.asarray(self.trajectory_id)
        if X.ndim != 3 or U.ndim != 3:
            raise ValueError("X and U must be rank-3 arrays")
        if X.shape[0] != U.shape[0] or X.shape[0] != len(ids):
            raise ValueError("trajectory count mismatch between X/U/trajectory_id")
        if U.shape[1] != X.shape[1]-1:
            raise ValueError("U must have exactly one fewer time sample than X")
        if not np.isfinite(float(self.dt)) or float(self.dt) <= 0:
            raise ValueError("dt must be finite and positive")
        states=list(self.schema.get("states",[])); inputs=list(self.schema.get("inputs",[]))
        if len(states) != X.shape[2] or len(inputs) != U.shape[2]:
            raise ValueError("schema dimensions do not match X/U")
        if self.sample_time_s is not None:
            time=np.asarray(self.sample_time_s,dtype=np.float64)
            if time.shape != X.shape[:2]:
                raise ValueError("sample_time_s must have shape [trajectory, time] matching X")
            if not np.isfinite(time).all():
                raise ValueError("sample_time_s must contain only finite seconds")


@dataclass
class AlgebraicDataset:
    """Device-agnostic sample table for constitutive/algebraic mechanism scopes."""
    values: np.ndarray          # [sample, variable]
    sample_id: np.ndarray       # [sample]
    schema: Dict[str, Any]      # {variables:[{name,dim,...}]}
    regime: str = "user"
    variant: str = "algebraic"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        A=np.asarray(self.values); ids=np.asarray(self.sample_id)
        if A.ndim != 2: raise ValueError("values must be rank-2")
        if len(ids) != len(A): raise ValueError("sample_id length mismatch")
        vars=list(self.schema.get("variables",[]))
        if len(vars) != A.shape[1]: raise ValueError("schema variable count does not match values")

    @property
    def variable_names(self) -> list[str]: return [str(v["name"]) for v in self.schema.get("variables",[])]


@dataclass(frozen=True)
class ObservationContract:
    """Traceable, device-neutral description of how a PublicDataset was observed."""
    source: str
    acquisition_id: str
    state_roles: Mapping[str, str] = field(default_factory=dict)
    input_roles: Mapping[str, str] = field(default_factory=dict)
    sensor_metadata: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    production_mechanism_eligible: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParameterPolicy:
    """Per-parameter PHM eligibility, independent of any concrete device adapter."""
    eligible_delta_theta: Sequence[str] = ()
    baseline_locked: Sequence[str] = ()
    window_unidentifiable: Sequence[str] = ()
    residual_ready: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


class DatasetAdapter(Protocol):
    name: str
    def load(self, source: str, **kwargs: Any) -> PublicDataset | AlgebraicDataset: ...
