from __future__ import annotations

"""Minimal solver-aware Universal Differential Equation runtime.

This implements a transparent fixed-step RK4 path for qualified continuous-
time dynamics.  It does not claim adaptive/stiff-solver or adjoint support.
"""

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import torch
from torch import nn

from .symbolic_compiler import CompiledSymbolicModule


@dataclass(frozen=True)
class UDERuntimeContract:
    solver: str = "fixed_step_rk4"
    differentiable: bool = True
    adaptive_step: bool = False
    stiff_solver: bool = False
    adjoint: bool = False
    known_path: str = "declared_symbolic_or_module"
    unknown_path: str = "additive_neural_residual"
    physical_truth_claim: str = "NOT_ESTABLISHED_BY_COMPILATION_OR_FIT"

    def to_dict(self) -> dict:
        return asdict(self)


class CompiledVectorField(nn.Module):
    """Stack scalar symbolic equations into dx/dt with state/control names."""

    def __init__(
        self,
        equations: Sequence[CompiledSymbolicModule],
        state_names: Sequence[str],
        control_names: Sequence[str] = (),
        *,
        time_name: str = "t",
    ) -> None:
        super().__init__()
        if not equations or len(equations) != len(state_names):
            raise ValueError("one symbolic equation is required per state")
        self.equations = nn.ModuleList(equations)
        self.state_names = tuple(state_names)
        self.control_names = tuple(control_names)
        self.time_name = str(time_name)
        expected = set(self.state_names) | set(self.control_names) | {self.time_name}
        for equation in equations:
            if set(equation.input_names) != expected:
                raise ValueError(
                    f"equation inputs {equation.input_names} do not match vector-field inputs {sorted(expected)}"
                )

    def forward(self, t: torch.Tensor, state: torch.Tensor, control: torch.Tensor | None = None) -> torch.Tensor:
        if state.shape[-1] != len(self.state_names):
            raise ValueError("state last dimension does not match state_names")
        if self.control_names:
            if control is None or control.shape[-1] != len(self.control_names):
                raise ValueError("control last dimension does not match control_names")
        env = {name: state[..., i] for i, name in enumerate(self.state_names)}
        if self.control_names:
            env.update({name: control[..., i] for i, name in enumerate(self.control_names)})
        env[self.time_name] = t
        return torch.stack([equation(**env) for equation in self.equations], dim=-1)


class HybridUDE(nn.Module):
    """Known vector field plus a trainable correction on selected state rates."""

    runtime_contract = UDERuntimeContract()

    def __init__(self, known_field: nn.Module, correction: nn.Module | None = None, *, correction_indices: Sequence[int] | None = None) -> None:
        super().__init__()
        self.known_field = known_field
        self.correction = correction
        self.correction_indices = None if correction_indices is None else tuple(int(x) for x in correction_indices)

    def forward(self, t: torch.Tensor, state: torch.Tensor, control: torch.Tensor | None = None) -> torch.Tensor:
        known = self.known_field(t, state, control)
        if known.shape != state.shape:
            raise ValueError(f"known field shape {tuple(known.shape)} must equal state shape {tuple(state.shape)}")
        if self.correction is None:
            return known
        features = state if control is None else torch.cat((state, control), dim=-1)
        residual = self.correction(features)
        if self.correction_indices is None:
            if residual.shape != known.shape:
                raise ValueError("full UDE correction must match state-rate shape")
            return known + residual
        if residual.shape[-1] != len(self.correction_indices) or residual.shape[:-1] != known.shape[:-1]:
            raise ValueError("partial UDE correction shape does not match correction_indices")
        if any(i < 0 or i >= known.shape[-1] for i in self.correction_indices):
            raise ValueError("correction_indices contain an invalid state index")
        out = known.clone()
        out[..., list(self.correction_indices)] = out[..., list(self.correction_indices)] + residual
        return out


def integrate_ude(
    model: HybridUDE,
    initial_state: torch.Tensor,
    times: torch.Tensor,
    controls: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable fixed-step RK4 integration.

    ``initial_state`` is ``[..., D]``; ``times`` is ``[T]``; optional controls
    are ``[T, ..., U]`` and are held constant over each interval.  The output is
    ``[T, ..., D]``.
    """
    if times.ndim != 1 or len(times) < 2:
        raise ValueError("times must be a one-dimensional tensor with at least two points")
    if not bool(torch.all(torch.isfinite(times)).detach().cpu()) or not bool(torch.all(times[1:] > times[:-1]).detach().cpu()):
        raise ValueError("times must be finite and strictly increasing")
    if controls is not None and (controls.shape[0] != len(times) or controls.shape[1:-1] != initial_state.shape[:-1]):
        raise ValueError("controls must have shape [T, ..., U] aligned with initial_state")
    state = initial_state
    states = [state]
    for i in range(len(times) - 1):
        t, dt = times[i], times[i + 1] - times[i]
        control = None if controls is None else controls[i]
        k1 = model(t, state, control)
        k2 = model(t + dt / 2, state + dt * k1 / 2, control)
        k3 = model(t + dt / 2, state + dt * k2 / 2, control)
        k4 = model(t + dt, state + dt * k3, control)
        state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if not bool(torch.all(torch.isfinite(state)).detach().cpu()):
            raise RuntimeError(f"UDE integration produced non-finite state at interval {i}")
        states.append(state)
    return torch.stack(states, dim=0)
