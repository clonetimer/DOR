#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical hierarchical ISGN core for C0/C1.

Hierarchy used by the formal validation chain:

    variables / roles / dimensions
        -> constitutive transform on a source path
        -> sparse signed/weighted equation term
        -> state balance / accumulation equation
        -> fixed differentiable RK4 integrator

Important design rule: equation semantics and constitutive operators are different layers.
`Identity / Saturation / DeadZone` may compete *within the constitutive layer* for an
input path, but they never compete with the state accumulation equation itself.

For C1-A, constitutive search is disabled and every path is Identity.  C1-B2a enables
constitutive search on public variables with role="input" while using the exact same
`HierarchicalISGNGraph` class.  There is no trainable dense x_t -> x_{t+1} readout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


DimVec = Tuple[int, int, int, int, int]
TIME: DimVec = (0, 0, 1, 0, 0)

# Single source of truth for physical dimensions: reuse the canonical Operator Library.
from .operators import Dim as CanonicalDim
DIMENSIONS: Dict[str, DimVec] = {
    "dimensionless": tuple(CanonicalDim.get("none")),
    "temperature": tuple(CanonicalDim.get("temperature")),
    "pressure": tuple(CanonicalDim.get("pressure")),
    "power": tuple(CanonicalDim.get("power")),
    "level": tuple(CanonicalDim.get("level")),
    "volume_flow": tuple(CanonicalDim.get("volume_flow")),
    "current": tuple(CanonicalDim.get("current")),
    "speed": tuple(CanonicalDim.get("speed")),
    "voltage": tuple(CanonicalDim.get("voltage")),
    "torque": tuple(CanonicalDim.get("torque")),
    "displacement": tuple(CanonicalDim.get("displacement")),
    "velocity": tuple(CanonicalDim.get("velocity")),
    "force": tuple(CanonicalDim.get("force")),
}


def dim_sub(a: Sequence[int], b: Sequence[int]) -> DimVec:
    return tuple(int(x - y) for x, y in zip(a, b))  # type: ignore[return-value]


def derivative_dim(state_dim: Sequence[int]) -> DimVec:
    return dim_sub(state_dim, TIME)


def _inv_softplus(y: float) -> float:
    y = max(float(y), 1e-6)
    return math.log(math.expm1(y))


@dataclass(frozen=True)
class VariableSpec:
    name: str
    role: str                 # "state" | "input" | "context"
    dim: DimVec


@dataclass(frozen=True)
class TermSpec:
    target_state: str
    source_name: str
    source_role: str
    source_dim: DimVec
    target_derivative_dim: DimVec
    coefficient_dim: DimVec
    kind: str = "source_term"


class SparseGate(nn.Module):
    """Deterministic hard-concrete-style gate with an L0-like expected-active penalty."""
    def __init__(self, init_log_alpha: float = 0.0, gamma: float = -0.1, zeta: float = 1.1):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(float(init_log_alpha)))
        self.gamma = float(gamma)
        self.zeta = float(zeta)
        self.register_buffer("_frozen", torch.tensor(-1.0))  # -1 means not frozen

    def value(self) -> torch.Tensor:
        if float(self._frozen) >= 0.0:
            return self._frozen
        s = torch.sigmoid(self.log_alpha)
        return torch.clamp(s * (self.zeta - self.gamma) + self.gamma, 0.0, 1.0)

    def expected_active(self) -> torch.Tensor:
        if float(self._frozen) >= 0.0:
            return self._frozen
        return torch.sigmoid(self.log_alpha)

    def freeze(self, active: bool) -> None:
        self._frozen.fill_(1.0 if active else 0.0)
        self.log_alpha.requires_grad_(False)


class ConstitutiveTransformBank(nn.Module):
    """Typed constitutive transform selection for one source path.

    C1-B2a deliberately keeps the vocabulary small:
      Identity(x)             = x
      Saturation(x; b)        = clamp(x, -b, b), b > 0
      DeadZone(x; c)          = sign(x) max(|x|-c, 0), c > 0

    The source dimension is preserved by all three transformations; the equation coefficient
    still supplies whatever dimension is needed to close d(state)/dt.
    """
    VOCAB = ("Identity", "Saturation", "DeadZone")

    def __init__(self, candidates: Sequence[str], seed: int = 0):
        super().__init__()
        cand = tuple(candidates)
        if not cand or any(c not in self.VOCAB for c in cand):
            raise ValueError(f"invalid constitutive candidates: {cand}")
        self.candidates = cand
        gen = torch.Generator().manual_seed(int(seed))
        self.logits = nn.Parameter(torch.randn(len(cand), generator=gen) * 0.03)
        # Generic, non-truth-specific starts.  Parameters are physical positive scales.
        self.raw_sat = nn.Parameter(torch.tensor(_inv_softplus(1.5)))
        self.raw_deadzone = nn.Parameter(torch.tensor(_inv_softplus(0.20)))
        self.temperature = 1.0
        self.register_buffer("_frozen_index", torch.tensor(-1, dtype=torch.long))

    def _values(self, x: torch.Tensor) -> List[torch.Tensor]:
        b = F.softplus(self.raw_sat) + 1e-4
        c = F.softplus(self.raw_deadzone) + 1e-4
        sat = torch.maximum(torch.minimum(x, b), -b)
        dz = torch.sign(x) * F.relu(torch.abs(x) - c)
        mapping = {"Identity": x, "Saturation": sat, "DeadZone": dz}
        return [mapping[n] for n in self.candidates]

    def probs(self) -> torch.Tensor:
        if int(self._frozen_index) >= 0:
            p = torch.zeros(len(self.candidates), dtype=self.logits.dtype, device=self.logits.device)
            p[int(self._frozen_index)] = 1.0
            return p
        return F.softmax(self.logits / max(float(self.temperature), 1e-3), dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vals = self._values(x)
        if int(self._frozen_index) >= 0:
            return vals[int(self._frozen_index)]
        stack = torch.stack(vals, dim=0)
        p = self.probs()
        shape = [len(self.candidates)] + [1] * x.ndim
        return (p.view(*shape) * stack).sum(dim=0)

    def entropy(self) -> torch.Tensor:
        if len(self.candidates) <= 1 or int(self._frozen_index) >= 0:
            return self.logits.sum() * 0.0
        p = self.probs().clamp_min(1e-8)
        return -(p * p.log()).sum()

    def complexity(self) -> torch.Tensor:
        """Prefer Identity unless data requires a nonlinear constitutive primitive."""
        if len(self.candidates) <= 1 or int(self._frozen_index) >= 0 or "Identity" not in self.candidates:
            return self.logits.sum() * 0.0
        p = self.probs()
        return 1.0 - p[self.candidates.index("Identity")]

    def selected(self) -> str:
        if int(self._frozen_index) >= 0:
            return self.candidates[int(self._frozen_index)]
        return self.candidates[int(torch.argmax(self.logits).detach())]

    def freeze(self, name: str) -> None:
        if name not in self.candidates:
            raise ValueError(f"{name} not in {self.candidates}")
        idx = self.candidates.index(name)
        self._frozen_index.fill_(idx)
        self.logits.requires_grad_(False)
        # Only the selected nonlinear operator retains its physical parameter.
        self.raw_sat.requires_grad_(name == "Saturation")
        self.raw_deadzone.requires_grad_(name == "DeadZone")

    def set_for_test(self, name: str, param: Optional[float] = None) -> None:
        self.freeze(name)
        if name == "Saturation" and param is not None:
            self.raw_sat.data.fill_(_inv_softplus(float(param)))
        if name == "DeadZone" and param is not None:
            self.raw_deadzone.data.fill_(_inv_softplus(float(param)))

    def physical_parameter(self) -> Optional[float]:
        name = self.selected()
        if name == "Saturation":
            return float((F.softplus(self.raw_sat) + 1e-4).detach())
        if name == "DeadZone":
            return float((F.softplus(self.raw_deadzone) + 1e-4).detach())
        return None

    def trainable_physical_parameters(self) -> List[nn.Parameter]:
        out: List[nn.Parameter] = []
        if self.raw_sat.requires_grad:
            out.append(self.raw_sat)
        if self.raw_deadzone.requires_grad:
            out.append(self.raw_deadzone)
        return out


class LinearMechanismTerm(nn.Module):
    def __init__(self, spec: TermSpec, coef_init: float, gate_init: float, transform_candidates: Sequence[str], seed: int):
        super().__init__()
        self.spec = spec
        self.coefficient = nn.Parameter(torch.tensor(float(coef_init)))
        self.gate = SparseGate(gate_init)
        self.transform = ConstitutiveTransformBank(transform_candidates, seed=seed)

    def effective_coefficient(self) -> torch.Tensor:
        return self.gate.value() * self.coefficient

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        return self.effective_coefficient() * self.transform(source)


class EquationNode(nn.Module):
    """One state-derivative equation: dx_i/dt = sum_j z_ij c_ij phi_ij(source_j)."""
    def __init__(self, target: VariableSpec, sources: Sequence[VariableSpec], seed: int = 0, constitutive_search: bool = False):
        super().__init__()
        if target.role != "state":
            raise ValueError("EquationNode target must be a state variable")
        self.target = target
        td = derivative_dim(target.dim)
        gen = torch.Generator().manual_seed(int(seed))
        terms: List[LinearMechanismTerm] = []
        specs: List[TermSpec] = []
        for j, src in enumerate(sources):
            coeff_dim = dim_sub(td, src.dim)
            spec = TermSpec(
                target_state=target.name,
                source_name=src.name,
                source_role=src.role,
                source_dim=src.dim,
                target_derivative_dim=td,
                coefficient_dim=coeff_dim,
            )
            c0 = float(torch.randn((), generator=gen) * 0.08)
            g0 = float(torch.randn((), generator=gen) * 0.08)
            # Only actuator/input paths search constitutive transforms. State/context paths stay Identity.
            tc = ("Identity", "Saturation", "DeadZone") if (constitutive_search and src.role == "input") else ("Identity",)
            terms.append(LinearMechanismTerm(spec, c0, g0, tc, seed=seed + 97 * (j + 1)))
            specs.append(spec)
        self.terms = nn.ModuleList(terms)
        self.specs = tuple(specs)

    def forward(self, source_values: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(source_values) != len(self.terms):
            raise ValueError("source_values length mismatch")
        out = torch.zeros_like(source_values[0])
        for term, value in zip(self.terms, source_values):
            out = out + term(value)
        return out

    def l0_penalty(self) -> torch.Tensor:
        return torch.stack([t.gate.expected_active() for t in self.terms]).sum()

    def operator_entropy(self) -> torch.Tensor:
        vals = [t.transform.entropy() for t in self.terms]
        return torch.stack(vals).sum() if vals else torch.tensor(0.0)

    def constitutive_complexity(self) -> torch.Tensor:
        vals = [t.transform.complexity() for t in self.terms]
        return torch.stack(vals).sum() if vals else torch.tensor(0.0)

    def effective_coefficients(self) -> Dict[str, float]:
        return {t.spec.source_name: float(t.effective_coefficient().detach()) for t in self.terms}


class HierarchicalISGNGraph(nn.Module):
    """Single canonical graph used by C0 and all C1 experiments.

    Public prior: variable names, roles, dimensions and a generic grammar vocabulary.
    Unknown: topology, signs, coefficients and (when enabled) constitutive operator choice.
    """
    def __init__(
        self,
        state_specs: Sequence[VariableSpec],
        input_specs: Sequence[VariableSpec],
        dt: float,
        seed: int = 0,
        allow_constant: bool = True,
        constitutive_search: bool = False,
    ):
        super().__init__()
        self.state_specs = tuple(state_specs)
        self.input_specs = tuple(input_specs)
        self.dt = float(dt)
        self.allow_constant = bool(allow_constant)
        self.constitutive_search = bool(constitutive_search)
        self.state_index = {v.name: i for i, v in enumerate(self.state_specs)}
        self.input_index = {v.name: i for i, v in enumerate(self.input_specs)}
        if len(self.state_index) != len(self.state_specs) or len(self.input_index) != len(self.input_specs):
            raise ValueError("variable names must be unique")

        sources: List[VariableSpec] = list(self.state_specs) + list(self.input_specs)
        if self.allow_constant:
            sources.append(VariableSpec("__constant__", "context", DIMENSIONS["dimensionless"]))
        self.source_specs = tuple(sources)
        self.equations = nn.ModuleList([
            EquationNode(target, self.source_specs, seed=seed + 1009 * i, constitutive_search=self.constitutive_search)
            for i, target in enumerate(self.state_specs)
        ])

    @classmethod
    def from_schema(
        cls,
        schema: Dict,
        dt: float,
        seed: int = 0,
        allow_constant: bool = True,
        constitutive_search: bool = False,
    ):
        states = [VariableSpec(v["name"], "state", DIMENSIONS[v["dim"]]) for v in schema["states"]]
        inputs = [VariableSpec(v["name"], v.get("role", "input"), DIMENSIONS[v["dim"]]) for v in schema["inputs"]]
        return cls(states, inputs, dt=dt, seed=seed, allow_constant=allow_constant, constitutive_search=constitutive_search)

    def set_operator_temperature(self, temperature: float) -> None:
        for eq in self.equations:
            for term in eq.terms:
                term.transform.temperature = float(temperature)

    def _source_values(self, x: torch.Tensor, u: torch.Tensor) -> List[torch.Tensor]:
        vals: List[torch.Tensor] = [x[..., i] for i in range(x.shape[-1])]
        vals.extend(u[..., i] for i in range(u.shape[-1]))
        if self.allow_constant:
            vals.append(torch.ones_like(x[..., 0]))
        return vals

    def rhs(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        vals = self._source_values(x, u)
        dx = [eq(vals) for eq in self.equations]
        return torch.stack(dx, dim=-1)

    def one_step(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        dt = self.dt
        k1 = self.rhs(x, u)
        k2 = self.rhs(x + 0.5 * dt * k1, u)
        k3 = self.rhs(x + 0.5 * dt * k2, u)
        k4 = self.rhs(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def rollout(self, x0: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        x = x0
        xs = [x]
        for t in range(u.shape[1]):
            x = self.one_step(x, u[:, t])
            xs.append(x)
        return torch.stack(xs, dim=1)

    def l0_penalty(self) -> torch.Tensor:
        return torch.stack([eq.l0_penalty() for eq in self.equations]).sum()

    def operator_entropy(self) -> torch.Tensor:
        return torch.stack([eq.operator_entropy() for eq in self.equations]).sum()

    def constitutive_complexity(self) -> torch.Tensor:
        return torch.stack([eq.constitutive_complexity() for eq in self.equations]).sum()

    def effective_coefficients(self) -> Dict[str, Dict[str, float]]:
        return {eq.target.name: eq.effective_coefficients() for eq in self.equations}

    def selected_transforms(self, active_graph: Optional[Dict[str, Iterable[str]]] = None) -> Dict[str, Dict[str, str]]:
        out: Dict[str, Dict[str, str]] = {}
        for eq in self.equations:
            active = None if active_graph is None else set(active_graph.get(eq.target.name, []))
            row: Dict[str, str] = {}
            for term in eq.terms:
                if active is None or term.spec.source_name in active:
                    row[term.spec.source_name] = term.transform.selected()
            out[eq.target.name] = row
        return out

    def constitutive_parameters(self, active_graph: Optional[Dict[str, Iterable[str]]] = None) -> Dict[str, Dict[str, Optional[float]]]:
        out: Dict[str, Dict[str, Optional[float]]] = {}
        for eq in self.equations:
            active = None if active_graph is None else set(active_graph.get(eq.target.name, []))
            row: Dict[str, Optional[float]] = {}
            for term in eq.terms:
                if active is None or term.spec.source_name in active:
                    row[term.spec.source_name] = term.transform.physical_parameter()
            out[eq.target.name] = row
        return out

    def active_graph(self, coef_threshold: float = 0.03, gate_threshold: float = 0.25) -> Dict[str, List[str]]:
        graph: Dict[str, List[str]] = {}
        for eq in self.equations:
            active: List[str] = []
            for term in eq.terms:
                c = abs(float(term.effective_coefficient().detach()))
                g = float(term.gate.value().detach())
                if c >= coef_threshold and g >= gate_threshold:
                    active.append(term.spec.source_name)
            graph[eq.target.name] = active
        return graph

    def apply_topology_prior(
        self,
        known_present: Optional[Dict[str, Iterable[str]]] = None,
        known_absent: Optional[Dict[str, Iterable[str]]] = None,
    ) -> None:
        """Apply a partial mechanism-graph prior before structure search.

        `known_present` edges remain active but their coefficients are still learned.
        `known_absent` edges are removed from the searchable hypothesis space.
        All other dimensionally legal edges remain learnable. Conflicting declarations raise.
        """
        kp = {k: set(v) for k, v in (known_present or {}).items()}
        ka = {k: set(v) for k, v in (known_absent or {}).items()}
        for target in set(kp) | set(ka):
            conflict = kp.get(target, set()) & ka.get(target, set())
            if conflict:
                raise ValueError(f"conflicting topology prior for {target}: {sorted(conflict)}")
        valid = {eq.target.name: {t.spec.source_name: t for t in eq.terms} for eq in self.equations}
        for target, names in list(kp.items()) + list(ka.items()):
            if target not in valid:
                raise KeyError(f"unknown target in topology prior: {target}")
            missing = set(names) - set(valid[target])
            if missing:
                raise KeyError(f"unknown source(s) for {target}: {sorted(missing)}")
        for eq in self.equations:
            present = kp.get(eq.target.name, set())
            absent = ka.get(eq.target.name, set())
            for term in eq.terms:
                name = term.spec.source_name
                if name in present:
                    term.gate.freeze(True)
                elif name in absent:
                    term.gate.freeze(False)
                    term.coefficient.requires_grad_(False)
                    term.coefficient.data.zero_()
                    term.transform.freeze("Identity" if "Identity" in term.transform.candidates else term.transform.candidates[0])

    def freeze_topology(self, active_graph: Dict[str, Iterable[str]]) -> None:
        for eq in self.equations:
            active = set(active_graph.get(eq.target.name, []))
            for term in eq.terms:
                is_active = term.spec.source_name in active
                term.gate.freeze(is_active)
                if not is_active:
                    term.coefficient.requires_grad_(False)
                    term.coefficient.data.zero_()
                    term.transform.freeze("Identity" if "Identity" in term.transform.candidates else term.transform.candidates[0])

    def freeze_constitutive(self, selected: Dict[str, Dict[str, str]]) -> None:
        for eq in self.equations:
            row = selected.get(eq.target.name, {})
            for term in eq.terms:
                if term.spec.source_name in row:
                    term.transform.freeze(row[term.spec.source_name])

    def trainable_physical_parameters(self) -> List[nn.Parameter]:
        out: List[nn.Parameter] = []
        for eq in self.equations:
            for term in eq.terms:
                if term.coefficient.requires_grad:
                    out.append(term.coefficient)
                    out.extend(term.transform.trainable_physical_parameters())
        # Unique parameters only.
        seen=set(); uniq=[]
        for p in out:
            if id(p) not in seen:
                seen.add(id(p)); uniq.append(p)
        return uniq

    def set_coefficients_for_test(self, coefficients: Dict[str, Dict[str, float]]) -> None:
        """C1-A test-only oracle injection (all constitutive transforms Identity)."""
        self.set_structure_for_test(coefficients, transforms={})

    def set_structure_for_test(
        self,
        coefficients: Dict[str, Dict[str, float]],
        transforms: Dict[str, Dict[str, Dict[str, float | str]]],
    ) -> None:
        """Test-only oracle injection for C0/B2a alignment. Never used by training."""
        for eq in self.equations:
            vals = coefficients.get(eq.target.name, {})
            trow = transforms.get(eq.target.name, {})
            for term in eq.terms:
                name = term.spec.source_name
                if name in vals:
                    term.coefficient.data.fill_(float(vals[name]))
                    term.gate.freeze(True)
                    info = trow.get(name, {"op": "Identity"})
                    op = str(info.get("op", "Identity"))
                    param = info.get("param", None)
                    term.transform.set_for_test(op, None if param is None else float(param))
                else:
                    term.coefficient.data.zero_()
                    term.gate.freeze(False)
                    term.transform.freeze("Identity" if "Identity" in term.transform.candidates else term.transform.candidates[0])
