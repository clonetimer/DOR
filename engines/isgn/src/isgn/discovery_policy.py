from __future__ import annotations

"""Evidence-aware routing across mechanism-discovery families.

The policy separates sparse derivative regression, trajectory residual fitting,
and symbolic relation search.  It selects a route, not a winning physical law.
"""

from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class DiscoveryRouteDecision:
    status: str
    route: str
    reason: str
    prerequisites: tuple[str, ...]
    alternatives: tuple[str, ...]
    claim_boundary: str = "A routed candidate still requires held-out mechanism validation and does not prove a unique true physical law."

    def to_dict(self) -> dict:
        return asdict(self)


def recommend_discovery_route(
    *,
    problem_kind: str,
    derivative_status: str = "UNKNOWN",
    variable_count: int,
    partial_physics: bool = False,
    sparse_library_available: bool = False,
    grammar_uncovered: bool = False,
    backend_availability: Mapping[str, bool] | None = None,
) -> DiscoveryRouteDecision:
    kind = str(problem_kind).upper()
    derivative = str(derivative_status).upper()
    if kind not in {"DYNAMICS", "EXPLICIT_RELATION", "IMPLICIT_RELATION"}:
        raise ValueError("problem_kind must be DYNAMICS, EXPLICIT_RELATION or IMPLICIT_RELATION")
    if derivative not in {"PASS", "WARN", "FAIL", "UNKNOWN"}:
        raise ValueError("derivative_status must be PASS, WARN, FAIL or UNKNOWN")
    if variable_count < 1:
        raise ValueError("variable_count must be positive")
    available = dict(backend_availability or {})

    if kind == "DYNAMICS":
        if derivative == "PASS" and sparse_library_available:
            backend = "PYSINDY" if available.get("pysindy", False) else "NATIVE_SPARSE"
            return DiscoveryRouteDecision(
                "READY", backend,
                "qualified derivatives and a declared candidate library support sparse state-equation identification",
                ("independent trajectory validation", "library coverage audit", "coefficient stability across splits"),
                ("TRAJECTORY_RESIDUAL", "SYMBOLIC_DISTILLATION"),
            )
        if partial_physics:
            return DiscoveryRouteDecision(
                "READY", "TRAJECTORY_RESIDUAL",
                "known partial dynamics are available but direct derivative evidence is not fully qualified",
                ("solver-compatible known dynamics", "state observability", "multi-start credit-assignment audit"),
                ("PYSINDY_AFTER_ACQUISITION_REPAIR",),
            )
        return DiscoveryRouteDecision(
            "BLOCKED", "OBSERVATION_OR_PRIOR_REQUIRED",
            "unqualified derivatives and absent partial physics do not support reliable dynamic-law discovery",
            ("improve sampling/noise", "declare partial physics or a justified candidate library"), (),
        )

    reducer = variable_count >= 5
    prerequisites = ["train/validation relation split", "unit/domain contract", "complexity-aware selection"]
    if reducer:
        prerequisites.insert(0, "physics/dimensional reduction before combinatorial search")
    if kind == "IMPLICIT_RELATION":
        return DiscoveryRouteDecision(
            "EXTERNAL_RUNTIME_REQUIRED", "SYMMATIKA_IMPLICIT_UPSTREAM",
            "implicit relation discovery is supported by upstream SymMatika but not by the current ISGN bridge",
            tuple(prerequisites + ["implicit-derivative metric validation"]),
            ("MANUAL_INVARIANT_LIBRARY",),
        )
    if grammar_uncovered:
        status = "READY" if available.get("symmatika", False) else "EXTERNAL_RUNTIME_REQUIRED"
        route = "SYMMATIKA_EXPLICIT" if not reducer else "PHYSICS_REDUCER_THEN_SYMMATIKA"
        return DiscoveryRouteDecision(
            status, route,
            "the finite native grammar is uncovered; structure-aware symbolic search is a qualified proposal generator",
            tuple(prerequisites), ("AI_FEYNMAN_STYLE_REDUCTION", "EQL_CANDIDATE"),
        )
    return DiscoveryRouteDecision(
        "READY", "NATIVE_ALGEBRAIC",
        "the relation remains inside the qualified native algebraic grammar",
        tuple(prerequisites), ("SYMMATIKA_AS_INDEPENDENT_CONSENSUS",),
    )
