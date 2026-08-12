"""Observation-scope planning for mechanism discovery.

This module sits *after* the raw Observation Gate and before a discovery backend.
It does not discover mechanisms and it does not change Gate thresholds.  Its job
is to turn evidence from multiple acquisition conditions (for example different
export rates and simulator-truth audits) into a traceable state-by-state route:

- direct discovery is supported;
- direct discovery is rate-limited;
- a nonlinear backend is acceptable even though a linear diagnostic probe is poor;
- filtering/weak-form/observer work is required first;
- the sensor resolution is insufficient;
- evidence is still insufficient.

The distinction is important for solver-agnostic ISGN: a failed derivative probe
must not be confused with "the physical mechanism does not exist".
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .observation_gate import ObservationGateConfig, ObservationGateReport, StateObservationAudit, assess_observation
from .contracts import PublicDataset


@dataclass(frozen=True)
class ObservationCandidate:
    name: str
    dataset: PublicDataset
    production_eligible: bool = True


@dataclass
class StateDiscoveryRoute:
    state: str
    eligibility: str  # READY / CONDITIONAL / BLOCKED
    route: str
    preferred_candidate: Optional[str]
    preferred_dt_s: Optional[float]
    max_direct_export_dt_s: Optional[float]
    production_statuses: Dict[str, str]
    truth_statuses: Dict[str, str]
    reason: str
    recommendation: str
    evidence: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ObservationScopeReport:
    status: str  # FULL_GRAPH_READY / PARTIAL_SCOPE_READY / OBSERVATION_BLOCKED
    full_graph_ready: bool
    candidates: Dict[str, Dict[str, Any]]
    state_routes: List[StateDiscoveryRoute]
    ready_states: List[str]
    conditional_states: List[str]
    blocked_states: List[str]
    candidate_scopes: Dict[str, Dict[str, List[str]]]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "full_graph_ready": self.full_graph_ready,
            "candidates": dict(self.candidates),
            "state_routes": [x.to_dict() for x in self.state_routes],
            "ready_states": list(self.ready_states),
            "conditional_states": list(self.conditional_states),
            "blocked_states": list(self.blocked_states),
            "candidate_scopes": dict(self.candidate_scopes),
            "recommendations": list(self.recommendations),
        }


def project_state_scope(ds: PublicDataset, states: Sequence[str], *, variant_suffix: str = "scope") -> PublicDataset:
    """Project a PublicDataset onto an explicit state scope, preserving inputs.

    This is a scope declaration, not imputation/state estimation.  The selected
    X columns remain byte-for-byte values from the source dataset.
    """
    names = [str(x.get("name")) for x in ds.schema.get("states", [])]
    requested = [str(x) for x in states]
    if not requested:
        raise ValueError("state scope cannot be empty")
    if len(set(requested)) != len(requested):
        raise ValueError("state scope contains duplicate names")
    missing = [x for x in requested if x not in names]
    if missing:
        raise ValueError(f"state scope contains unknown states: {missing}")
    idx = [names.index(x) for x in requested]
    schema = {
        "states": [dict(ds.schema["states"][i]) for i in idx],
        "inputs": [dict(x) for x in ds.schema.get("inputs", [])],
    }
    if "bridge" in ds.schema:
        schema["bridge"] = dict(ds.schema["bridge"])
    return PublicDataset(
        X=np.asarray(ds.X)[:, :, idx].copy(),
        U=np.asarray(ds.U).copy(),
        trajectory_id=np.asarray(ds.trajectory_id).copy(),
        schema=schema,
        dt=float(ds.dt),
        regime=str(ds.regime),
        variant=f"{ds.variant}-{variant_suffix}",
    )


def _state_map(report: ObservationGateReport) -> Dict[str, StateObservationAudit]:
    return {x.state: x for x in report.states}


def _sensor_resolution_failed(s: StateObservationAudit, cfg: ObservationGateConfig) -> bool:
    return s.variation_to_sensor_floor is not None and s.variation_to_sensor_floor < cfg.fail_measurement_variation_ratio


def _warning_route(s: StateObservationAudit, cfg: ObservationGateConfig) -> Tuple[str, str, str]:
    noisy = s.derivative_noise_ratio > cfg.warn_derivative_noise_ratio
    bandwidth = s.f95_nyquist_ratio > cfg.warn_bandwidth_nyquist_ratio
    if noisy or bandwidth:
        return (
            "FILTERED_OR_WEAK_FRONTEND",
            "direct derivative evidence is marginal even though the state is not hard-blocked",
            "validate an update-aware filtered/weak-form frontend before interpreting this state's physical coefficients",
        )
    if s.derivative_probe_nrmse > cfg.warn_probe_nrmse:
        return (
            "NONLINEAR_BACKEND_OK",
            "the observation itself is numerically usable, but the local linear diagnostic probe is poor",
            "retain the state and use a nonlinear/grammar-aware backend; do not treat linear-probe error as an observability failure",
        )
    return (
        "CONDITIONAL_DIRECT",
        "the state is conditionally usable under the current acquisition contract",
        "retain explicit observation caution and verify the state during backend consensus/certification",
    )


def compare_observation_candidates(
    candidates: Sequence[ObservationCandidate],
    *,
    config: Optional[ObservationGateConfig] = None,
) -> ObservationScopeReport:
    """Compare production and audit-only acquisition candidates state-by-state."""
    cfg = config or ObservationGateConfig()
    if not candidates:
        raise ValueError("at least one observation candidate is required")
    names = [c.name for c in candidates]
    if len(set(names)) != len(names):
        raise ValueError("observation candidate names must be unique")

    reports: Dict[str, ObservationGateReport] = {c.name: assess_observation(c.dataset, config=cfg) for c in candidates}
    production = [c for c in candidates if c.production_eligible]
    truth = [c for c in candidates if not c.production_eligible]
    if not production:
        raise ValueError("at least one production-eligible observation candidate is required")

    state_union: List[str] = []
    for c in production:
        for s in c.dataset.schema.get("states", []):
            n = str(s.get("name"))
            if n and n not in state_union:
                state_union.append(n)

    report_maps = {name: _state_map(rep) for name, rep in reports.items()}
    routes: List[StateDiscoveryRoute] = []

    for state in state_union:
        prod_rows = [(c, report_maps[c.name].get(state)) for c in production if report_maps[c.name].get(state) is not None]
        truth_rows = [(c, report_maps[c.name].get(state)) for c in truth if report_maps[c.name].get(state) is not None]
        prod_status = {c.name: s.status for c, s in prod_rows}
        truth_status = {c.name: s.status for c, s in truth_rows}
        evidence: Dict[str, Any] = {
            "production": {
                c.name: {
                    "dt_s": float(c.dataset.dt),
                    "status": s.status,
                    "derivative_noise_ratio": s.derivative_noise_ratio,
                    "derivative_probe_nrmse": s.derivative_probe_nrmse,
                    "f95_nyquist_ratio": s.f95_nyquist_ratio,
                    "variation_to_sensor_floor": s.variation_to_sensor_floor,
                    "sensor_period_to_export_dt": s.sensor_period_to_export_dt,
                }
                for c, s in prod_rows
            },
            "audit_only": {
                c.name: {
                    "dt_s": float(c.dataset.dt),
                    "status": s.status,
                    "derivative_noise_ratio": s.derivative_noise_ratio,
                    "derivative_probe_nrmse": s.derivative_probe_nrmse,
                    "f95_nyquist_ratio": s.f95_nyquist_ratio,
                }
                for c, s in truth_rows
            },
        }

        # Measurement resolution is a hard production boundary.  If all
        # production candidates have declared sensors and all are below the
        # minimum variation/noise margin, collecting the same signal longer at
        # the same operating condition cannot be called a direct mechanism input.
        resolution_good = [(c, s) for c, s in prod_rows if not _sensor_resolution_failed(s, cfg)]
        resolution_bad = [(c, s) for c, s in prod_rows if _sensor_resolution_failed(s, cfg)]
        if prod_rows and not resolution_good and resolution_bad:
            best = max(resolution_bad, key=lambda x: float(x[1].variation_to_sensor_floor or 0.0))
            routes.append(StateDiscoveryRoute(
                state, "BLOCKED", "SENSOR_RESOLUTION_OR_EXCITATION_CHANGE_REQUIRED", None, None, None,
                prod_status, truth_status,
                "observed variation is not distinguishable from the declared sensor noise/quantization floor",
                "increase physical variation/observation duration or improve the sensor before including this state in a certified mechanism graph",
                evidence,
            ))
            continue

        direct = [(c, s) for c, s in resolution_good if s.status == "PASS"]
        if direct:
            # Prefer an acquisition rate close to the sensor's native period and
            # then the lower derivative-noise ratio.  This avoids blindly picking
            # the highest export rate when it merely amplifies measurement noise.
            def direct_key(item):
                c, s = item
                ratio = s.sensor_period_to_export_dt
                alignment = abs(np.log(max(float(ratio), 1e-12))) if ratio is not None and ratio > 0 else 0.0
                return (alignment, s.derivative_noise_ratio, float(c.dataset.dt))
            best_c, best_s = min(direct, key=direct_key)
            pass_dts = sorted(float(c.dataset.dt) for c, s in direct)
            failed_at_coarser = any(s.status == "FAIL" and float(c.dataset.dt) > max(pass_dts) for c, s in resolution_good)
            if failed_at_coarser:
                route = "DIRECT_WITH_EXPORT_RATE_REQUIREMENT"
                reason = "the state is directly usable at a finer acquisition/export grid but fails at a coarser grid"
                rec = f"use export dt <= {max(pass_dts):.9g} s for this state or validate a weak/filtered alternative"
                max_dt = max(pass_dts)
            else:
                route = "DIRECT"
                reason = "at least one production acquisition condition supports direct derivative-based observation"
                rec = "retain the preferred acquisition contract during mechanism discovery"
                max_dt = max(pass_dts)
            routes.append(StateDiscoveryRoute(
                state, "READY", route, best_c.name, float(best_c.dataset.dt), max_dt,
                prod_status, truth_status, reason, rec, evidence,
            ))
            continue

        conditional = [(c, s) for c, s in resolution_good if s.status == "WARN"]
        if conditional:
            def warn_key(item):
                c, s = item
                noisy = max(0.0, s.derivative_noise_ratio - cfg.warn_derivative_noise_ratio)
                bw = max(0.0, s.f95_nyquist_ratio - cfg.warn_bandwidth_nyquist_ratio)
                probe = max(0.0, s.derivative_probe_nrmse - cfg.warn_probe_nrmse)
                return (noisy + bw + 0.25 * probe, float(c.dataset.dt))
            best_c, best_s = min(conditional, key=warn_key)
            route, reason, rec = _warning_route(best_s, cfg)
            eligibility = "READY" if route == "NONLINEAR_BACKEND_OK" else "CONDITIONAL"
            routes.append(StateDiscoveryRoute(
                state, eligibility, route, best_c.name, float(best_c.dataset.dt), None,
                prod_status, truth_status, reason, rec, evidence,
            ))
            continue

        # No production condition is directly/conditionally usable.  Truth-audit
        # evidence can distinguish an observation-chain problem from a state that
        # is unresolved even in the simulator's latent state.
        truth_usable = [(c, s) for c, s in truth_rows if s.status in ("PASS", "WARN")]
        if truth_usable:
            best_t, best_s = min(truth_usable, key=lambda x: (0 if x[1].status == "PASS" else 1, x[1].derivative_noise_ratio))
            routes.append(StateDiscoveryRoute(
                state, "BLOCKED", "FILTERED_OR_OBSERVER_FRONTEND_REQUIRED", None, None, None,
                prod_status, truth_status,
                "public measurements are hard-blocked while audit-only simulator truth remains usable",
                "validate filtering/delay compensation/state estimation against held-out truth before exposing this state to a mechanism-discovery backend",
                evidence,
            ))
        else:
            routes.append(StateDiscoveryRoute(
                state, "BLOCKED", "UNRESOLVED_OBSERVATION_LIMIT", None, None, None,
                prod_status, truth_status,
                "no tested production acquisition condition supports the state and audit-only evidence does not isolate the limitation to the sensor chain",
                "change acquisition/excitation or validate a weak-form/observer formulation before mechanism discovery",
                evidence,
            ))

    ready = [r.state for r in routes if r.eligibility == "READY"]
    conditional = [r.state for r in routes if r.eligibility == "CONDITIONAL"]
    blocked = [r.state for r in routes if r.eligibility == "BLOCKED"]
    full_ready = not blocked and all(r.eligibility == "READY" for r in routes)
    if full_ready:
        status = "FULL_GRAPH_READY"
    elif ready or conditional:
        status = "PARTIAL_SCOPE_READY"
    else:
        status = "OBSERVATION_BLOCKED"

    candidate_scopes: Dict[str, Dict[str, List[str]]] = {}
    for c in production:
        m = report_maps[c.name]
        candidate_scopes[c.name] = {
            "direct": [s for s, a in m.items() if a.status == "PASS" and not _sensor_resolution_failed(a, cfg)],
            "conditional": [s for s, a in m.items() if a.status == "WARN" and not _sensor_resolution_failed(a, cfg)],
            "blocked": [s for s, a in m.items() if a.status == "FAIL" or _sensor_resolution_failed(a, cfg)],
        }

    recs: List[str] = []
    for r in routes:
        if r.recommendation and r.recommendation not in recs:
            recs.append(r.recommendation)

    candidate_payload = {
        c.name: {
            "production_eligible": bool(c.production_eligible),
            "dt_s": float(c.dataset.dt),
            "gate": reports[c.name].to_dict(),
        }
        for c in candidates
    }
    return ObservationScopeReport(status, full_ready, candidate_payload, routes, ready, conditional, blocked, candidate_scopes, recs)
