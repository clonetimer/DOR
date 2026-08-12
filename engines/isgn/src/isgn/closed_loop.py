from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence


@dataclass
class InformationGap:
    gap_id: str
    gap_type: str
    target_variables: List[str]
    reason: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    priority: int = 50

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClosedLoopAction:
    action_id: str
    gap_id: str
    gap_type: str
    target_variables: List[str]
    purpose: str
    action_type: str
    requested_changes: Dict[str, Any]
    success_criteria: Dict[str, Any]
    priority: int
    dependencies: List[str] = field(default_factory=list)
    production_safety: Dict[str, Any] = field(default_factory=lambda: {
        "truth_channels_allowed": False,
        "fault_labels_allowed_for_normal_mechanism_discovery": False,
        "device_specific_primitive_allowed_without_reuse_evidence": False,
    })

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClosedLoopPlan:
    system_id: str
    status: str
    gaps: List[InformationGap]
    actions: List[ClosedLoopAction]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": "isgn-closed-loop-plan-v1",
            "system_id": self.system_id,
            "status": self.status,
            "gaps": [g.to_dict() for g in self.gaps],
            "actions": [a.to_dict() for a in self.actions],
            "metadata": dict(self.metadata),
        }


@dataclass
class ActionEvaluation:
    action_id: str
    gap_id: str
    status: str
    reason: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClosedLoopEvaluation:
    system_id: str
    status: str
    actions: List[ActionEvaluation]
    resolved_actions: List[str]
    unresolved_actions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": "isgn-closed-loop-evaluation-v1",
            "system_id": self.system_id,
            "status": self.status,
            "actions": [a.to_dict() for a in self.actions],
            "resolved_actions": list(self.resolved_actions),
            "unresolved_actions": list(self.unresolved_actions),
            "metadata": dict(self.metadata),
        }


def _upper(value: Any) -> str:
    return str(value or "").upper()


def _observation_report(payload: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    # accept a direct ObservationGateReport dict, a wrapper, or an observation
    # closure candidate whose ``gate`` field is the production gate.
    if "states" in payload and "blocked_states" in payload:
        return payload
    if isinstance(payload.get("gate"), Mapping):
        return payload["gate"]
    if isinstance(payload.get("observation"), Mapping):
        return payload["observation"]
    return payload


def _state_map(observation: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {str(x.get("state")): x for x in observation.get("states", []) if isinstance(x, Mapping) and x.get("state")}


def _state_route_map(observation: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {str(x.get("state")): x for x in observation.get("state_routes", []) if isinstance(x, Mapping) and x.get("state")}


def _consensus_report(payload: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    if isinstance(payload.get("consensus"), Mapping):
        return payload["consensus"]
    return payload


def _primary_graph(payload: Optional[Mapping[str, Any]]) -> Mapping[str, Sequence[str]]:
    if not isinstance(payload, Mapping):
        return {}
    primary = payload.get("primary")
    if isinstance(primary, Mapping) and isinstance(primary.get("graph"), Mapping):
        return primary["graph"]
    graph = payload.get("graph")
    return graph if isinstance(graph, Mapping) else {}


def _completion_certificate(payload: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    if isinstance(payload.get("certificate"), Mapping):
        return payload["certificate"]
    if isinstance(payload.get("consensus_certificate"), Mapping):
        return payload["consensus_certificate"]
    return payload


def _blocked_observation_gaps(observation: Optional[Mapping[str, Any]]) -> List[InformationGap]:
    obs = _observation_report(observation)
    if not obs:
        return []
    blocked = list(map(str, obs.get("blocked_states", []) or []))
    smap = _state_map(obs)
    rmap = _state_route_map(obs)
    dt = float((obs.get("summary") or {}).get("dt", 0.0) or 0.0)
    cfg = obs.get("config") or {}
    fail_noise = float(cfg.get("fail_derivative_noise_ratio", 2.0) or 2.0)
    fail_variation = float(cfg.get("fail_measurement_variation_ratio", 2.0) or 2.0)
    gaps: List[InformationGap] = []
    for state in blocked:
        row = smap.get(state, {})
        route = rmap.get(state, {})
        # ObservationScopeReport is already a state-level synthesis of multiple
        # production acquisition candidates.  Prefer its production-only route
        # over reconstructing the decision from one raw Gate report.
        if route:
            prod = (route.get("evidence") or {}).get("production", {})
            best_name = route.get("preferred_candidate")
            best = prod.get(best_name, {}) if best_name in prod else (next(iter(prod.values())) if prod else {})
            row = {**best, **row}
        variation = row.get("variation_to_sensor_floor")
        noise_ratio = row.get("derivative_noise_ratio")
        sensor_period = row.get("sensor_sample_period_s")
        route_name = str(route.get("route", ""))
        evidence = {
            "state_status": row.get("status"),
            "observation_route": route_name or None,
            "route_reason": route.get("reason"),
            "route_recommendation": route.get("recommendation"),
            "preferred_candidate": route.get("preferred_candidate"),
            "preferred_dt_s": route.get("preferred_dt_s"),
            "max_direct_export_dt_s": route.get("max_direct_export_dt_s"),
            "export_dt_s": dt or row.get("dt_s") or None,
            "sensor_sample_period_s": sensor_period,
            "variation_to_sensor_floor": variation,
            "derivative_noise_ratio": noise_ratio,
            "fail_measurement_variation_ratio": fail_variation,
            "fail_derivative_noise_ratio": fail_noise,
        }
        if route_name == "SENSOR_RESOLUTION_OR_EXCITATION_CHANGE_REQUIRED" or (variation is not None and float(variation) < fail_variation):
            reason = "measured state variation is below the declared sensing-resolution margin"
            priority = 100
        elif route_name in {"FILTERED_OR_OBSERVER_FRONTEND_REQUIRED", "UNRESOLVED_OBSERVATION_LIMIT"}:
            reason = str(route.get("reason") or "production observation requires a different identification/sensing front-end")
            priority = 96
        elif noise_ratio is not None and float(noise_ratio) >= fail_noise:
            reason = "direct derivative evidence is too noisy at the current production acquisition/front-end"
            priority = 95
        else:
            reason = "state is blocked by the production Observation Gate"
            priority = 90
        gaps.append(InformationGap(f"observation:{state}", "OBSERVATION_BLOCKED", [state], reason, evidence, priority))
    return gaps


def _gate_check_gaps(observation: Optional[Mapping[str, Any]]) -> List[InformationGap]:
    obs = _observation_report(observation)
    if not obs:
        return []
    checks = {str(c.get("name")): c for c in obs.get("checks", []) if isinstance(c, Mapping)}
    gaps: List[InformationGap] = []
    data = checks.get("data_integrity", {})
    if _upper(data.get("status")) == "FAIL":
        gaps.append(InformationGap(
            "observation:data-integrity", "DATA_INTEGRITY_INSUFFICIENT", [],
            str(data.get("reason") or "trajectory/data contract is insufficient for discovery"),
            {"metrics": dict(data.get("metrics", {})), "recommendation": data.get("recommendation")}, 110))
    pe = checks.get("persistent_excitation", {})
    if _upper(pe.get("status")) == "FAIL":
        metrics = dict(pe.get("metrics", {})); targets=list(map(str, metrics.get("dynamic_inputs", []) or []))
        gaps.append(InformationGap(
            "observation:persistent-excitation", "PERSISTENT_EXCITATION_INSUFFICIENT", targets,
            str(pe.get("reason") or "command history lacks sufficient independent temporal excitation"),
            {"metrics": metrics, "recommendation": pe.get("recommendation")}, 98))
    si = checks.get("source_identifiability", {})
    if _upper(si.get("status")) == "FAIL":
        metrics = dict(si.get("metrics", {})); targets=list(map(str, metrics.get("inactive_sources", []) or []))
        if not targets:
            targets=list(map(str, metrics.get("active_sources", []) or []))
        gaps.append(InformationGap(
            "observation:source-identifiability", "SOURCE_IDENTIFIABILITY_INSUFFICIENT", targets,
            str(si.get("reason") or "candidate sources are not independently identifiable"),
            {"metrics": metrics, "recommendation": si.get("recommendation")}, 97))
    return gaps


def _synthesis_discovery_gaps(synthesis: Optional[Mapping[str, Any]]) -> List[InformationGap]:
    if not isinstance(synthesis, Mapping):
        return []
    atlas = synthesis.get("atlas") if isinstance(synthesis.get("atlas"), Mapping) else synthesis
    scopes = atlas.get("scopes", {}) if isinstance(atlas, Mapping) else {}
    gaps: List[InformationGap] = []
    for sid, scope in scopes.items():
        if not isinstance(scope, Mapping):
            continue
        md = scope.get("metadata") or {}
        missing = list(map(str, md.get("discovery_required_targets", []) or []))
        if not missing:
            continue
        pool = list(map(str, md.get("candidate_driver_pool", []) or []))
        gaps.append(InformationGap(
            f"discovery:{sid}", "DISCOVERY_REQUIRED", missing,
            "scope is physically plausible but one or more target dynamics have not yet been stably discovered",
            {"scope_id": str(sid), "candidate_driver_pool": pool, "known_graph": dict(scope.get("graph", {}))}, 85,
        ))
    return gaps


def _consensus_gaps(consensus: Optional[Mapping[str, Any]]) -> List[InformationGap]:
    rep = _consensus_report(consensus)
    if not rep or _upper(rep.get("status")) != "BACKEND_DISAGREEMENT":
        return []
    disagreement = rep.get("disagreement_edges") or {}
    variables: set[str] = set()
    for edges in disagreement.values() if isinstance(disagreement, Mapping) else []:
        for edge in edges or []:
            text = str(edge)
            if "->" in text:
                source, target = text.split("->", 1)
                variables.update([source, target])
    return [InformationGap(
        "consensus:backend-disagreement", "BACKEND_DISAGREEMENT", sorted(variables),
        "independent discovery backends disagree beyond the configured structural tolerance",
        {"graph_jaccard": rep.get("graph_jaccard"), "operator_agreement": rep.get("operator_agreement"),
         "disagreement_edges": disagreement, "backends": list(rep.get("backends", []) or [])}, 92,
    )]


def _completion_gaps(completion: Optional[Mapping[str, Any]]) -> List[InformationGap]:
    cert = _completion_certificate(completion)
    if not cert:
        return []
    status = _upper(cert.get("status"))
    target = cert.get("coverage_target")
    targets = [str(target)] if target else []
    if status == "GRAMMAR_UNCOVERED":
        return [InformationGap(
            "completion:grammar-uncovered", "GRAMMAR_UNCOVERED", targets,
            "the certified residual exceeds the independently calibrated coverage floor for at least one target",
            {"coverage_target": target, "coverage_ratio": cert.get("coverage_ratio"),
             "per_target_coverage": cert.get("per_target_coverage"), "backend": cert.get("backend")}, 88,
        )]
    if status == "UNIDENTIFIABLE":
        return [InformationGap(
            "completion:unidentifiable", "UNIDENTIFIABLE", targets,
            "candidate mechanisms exist but the current data do not uniquely identify the active relation/parameters",
            {"rank": cert.get("rank"), "n_columns": cert.get("n_columns"), "condition_number": cert.get("condition_number")}, 94,
        )]
    if status == "PRIOR_CONFLICT":
        variables = sorted({str(x) for item in cert.get("prior_conflicts", []) for x in (item if isinstance(item, (list, tuple)) else [item])})
        return [InformationGap(
            "completion:prior-conflict", "PRIOR_CONFLICT", variables,
            "declared mechanism prior conflicts with production data",
            {"prior_conflicts": cert.get("prior_conflicts", [])}, 99,
        )]
    return []


def _action_for_gap(gap: InformationGap) -> ClosedLoopAction:
    safety = {
        "truth_channels_allowed": False,
        "fault_labels_allowed_for_normal_mechanism_discovery": False,
        "device_specific_primitive_allowed_without_reuse_evidence": False,
    }
    if gap.gap_type == "DATA_INTEGRITY_INSUFFICIENT":
        return ClosedLoopAction(
            f"act:{gap.gap_id}", gap.gap_id, gap.gap_type, gap.target_variables,
            "repair the production data contract before any mechanism search is attempted",
            "REPAIR_OR_EXTEND_DATA_ACQUISITION_CONTRACT",
            {"collect_longer_trajectory_if_horizon_is_short": True, "remove_or_reacquire_nonfinite_segments": True,
             "preserve_trajectory_boundaries_and_dt": True},
            {"gate_check": "data_integrity", "required_status": "PASS"}, gap.priority, production_safety=safety)

    if gap.gap_type == "PERSISTENT_EXCITATION_INSUFFICIENT":
        return ClosedLoopAction(
            f"act:{gap.gap_id}", gap.gap_id, gap.gap_type, gap.target_variables,
            "increase command-history rank across the target time scales while respecting equipment safety limits",
            "DESIGN_PERSISTENT_EXCITATION",
            {"dynamic_inputs": list(gap.target_variables), "use_multi_level_or_multisine_or_chirp_within_safe_limits": True,
             "vary_multiple_commands_independently_when_present": True, "preserve_normal_operating_domain": True},
            {"gate_check": "persistent_excitation", "required_status": "PASS"}, gap.priority, production_safety=safety)

    if gap.gap_type == "SOURCE_IDENTIFIABILITY_INSUFFICIENT":
        return ClosedLoopAction(
            f"act:{gap.gap_id}", gap.gap_id, gap.gap_type, gap.target_variables,
            "separate collinear or inactive candidate sources before attributing distinct mechanisms",
            "DECORRELATE_OR_ACTIVATE_CANDIDATE_SOURCES",
            {"focus_sources": list(gap.target_variables), "vary_controllable_sources_independently_if_safe": True,
             "add_independent_operating_regimes_for_context_variables": True,
             "remove_a_source_from_claimed_scope_if_it_cannot_vary": True},
            {"gate_check": "source_identifiability", "required_status": "PASS"}, gap.priority, production_safety=safety)

    if gap.gap_type == "OBSERVATION_BLOCKED":
        ev = gap.evidence
        variation = ev.get("variation_to_sensor_floor")
        fail_variation = float(ev.get("fail_measurement_variation_ratio", 2.0) or 2.0)
        noise = ev.get("derivative_noise_ratio")
        fail_noise = float(ev.get("fail_derivative_noise_ratio", 2.0) or 2.0)
        dt = ev.get("export_dt_s")
        sp = ev.get("sensor_sample_period_s")
        route_name = str(ev.get("observation_route") or "")
        if route_name == "SENSOR_RESOLUTION_OR_EXCITATION_CHANGE_REQUIRED" or (variation is not None and float(variation) < fail_variation):
            action_type = "IMPROVE_SENSOR_OR_PHYSICAL_VARIATION"
            purpose = "make the production measurement distinguish physical variation from the declared sensing floor"
            changes = {"increase_observation_duration": True, "increase_physical_excitation_if_safe": True,
                       "improve_sensor_noise_or_quantization_if_available": True}
            criteria = {"state_removed_from_blocked_states": True, "minimum_variation_to_sensor_floor": fail_variation}
        elif route_name == "DIRECT_WITH_EXPORT_RATE_REQUIREMENT":
            action_type = "INCREASE_EXPORT_RATE_TO_AVAILABLE_SENSOR_RATE"
            target_dt = ev.get("max_direct_export_dt_s") or ev.get("preferred_dt_s") or sp
            purpose = "avoid discarding sensor bandwidth before derivative-based discovery"
            changes = {"maximum_export_dt_s": float(target_dt), "preserve_sensor_contract": True, "no_truth_substitution": True}
            criteria = {"state_removed_from_blocked_states": True, "maximum_derivative_noise_ratio": fail_noise,
                        "maximum_export_dt_s": float(target_dt)}
        elif noise is not None and float(noise) >= fail_noise and dt and sp and float(sp) < float(dt) * (1.0 - 1e-6):
            action_type = "INCREASE_EXPORT_RATE_TO_AVAILABLE_SENSOR_RATE"
            purpose = "avoid discarding sensor bandwidth before derivative-based discovery"
            changes = {"maximum_export_dt_s": float(sp), "preserve_sensor_contract": True, "no_truth_substitution": True}
            criteria = {"state_removed_from_blocked_states": True, "maximum_derivative_noise_ratio": fail_noise,
                        "maximum_export_dt_s": float(sp)}
        else:
            action_type = "CHANGE_IDENTIFICATION_FRONTEND_OR_SENSING"
            purpose = "make the target usable without weakening the Observation Gate"
            changes = {"try_filtered_or_weak_form_frontend": True, "observer_if_physical_state_is_latent": True,
                       "higher_rate_or_lower_noise_sensor_if_available": True, "do_not_relax_gate_thresholds": True}
            criteria = {"state_removed_from_blocked_states": True, "maximum_derivative_noise_ratio": fail_noise}
        return ClosedLoopAction(f"act:{gap.gap_id}", gap.gap_id, gap.gap_type, gap.target_variables,
                                purpose, action_type, changes, criteria, gap.priority, production_safety=safety)

    if gap.gap_type == "DISCOVERY_REQUIRED":
        pool = list(gap.evidence.get("candidate_driver_pool", []) or [])
        return ClosedLoopAction(
            f"act:{gap.gap_id}", gap.gap_id, gap.gap_type, gap.target_variables,
            "collect normal operating evidence that separates candidate mechanism drivers and then require independent-backend agreement",
            "DESIGN_DISCRIMINATING_EXCITATION_AND_RUN_CONSENSUS",
            {"candidate_driver_pool": pool, "vary_inputs_across_multiple_safe_regimes": True,
             "prefer_independent_driver_variation": True, "hold_out_entire_trajectories_for_validation": True,
             "run_at_least_two_independent_backends": True},
            {"consensus_status": "CONSENSUS", "minimum_graph_jaccard": 0.8,
             "stable_relation_required_for_targets": list(gap.target_variables)},
            gap.priority, production_safety=safety,
        )

    if gap.gap_type == "BACKEND_DISAGREEMENT":
        return ClosedLoopAction(
            f"act:{gap.gap_id}", gap.gap_id, gap.gap_type, gap.target_variables,
            "acquire normal evidence that discriminates the competing mechanism structures instead of selecting the lowest validation error",
            "ACQUIRE_DISAMBIGUATING_EXCITATION_OR_OBSERVATION",
            {"focus_variables": list(gap.target_variables), "disagreement_edges": gap.evidence.get("disagreement_edges", {}),
             "increase_independent_excitation_rank": True, "repeat_across_independent_operating_trajectories": True,
             "do_not_auto_select_best_error_backend": True},
            {"consensus_status": "CONSENSUS", "minimum_graph_jaccard": 0.8, "minimum_operator_agreement": 0.8},
            gap.priority, production_safety=safety,
        )

    if gap.gap_type == "GRAMMAR_UNCOVERED":
        return ClosedLoopAction(
            f"act:{gap.gap_id}", gap.gap_id, gap.gap_type, gap.target_variables,
            "identify a reusable constitutive/mechanism family that closes the certified residual without changing the already-supported topology opportunistically",
            "RUN_REUSABLE_CONSTITUTIVE_LIBRARY_SEARCH",
            {"coverage_target": gap.evidence.get("coverage_target"), "preserve_existing_stable_topology_where_possible": True,
             "require_cross_trajectory_validation": True, "reuse_evidence_required_before_core_grammar_extension": True,
             "do_not_use_truth_force_or_hidden_state_as_production_input": True},
            {"completion_status": "COMPLETE", "use_same_independent_coverage_calibration": True,
             "coverage_ratio_must_fall_below_existing_threshold": True},
            gap.priority, production_safety=safety,
        )

    if gap.gap_type == "UNIDENTIFIABLE":
        return ClosedLoopAction(
            f"act:{gap.gap_id}", gap.gap_id, gap.gap_type, gap.target_variables,
            "increase source independence so the candidate mechanisms/parameters become identifiable",
            "INCREASE_INDEPENDENT_EXCITATION",
            {"target_variables": list(gap.target_variables), "vary_collinear_sources_independently_if_safe": True,
             "extend_operating_regime_coverage": True},
            {"identifiable": True, "completion_status_not": "UNIDENTIFIABLE"}, gap.priority, production_safety=safety,
        )

    return ClosedLoopAction(
        f"act:{gap.gap_id}", gap.gap_id, gap.gap_type, gap.target_variables,
        "resolve the mechanism-prior conflict before further model discovery",
        "REVIEW_OR_REVISE_DECLARED_MECHANISM_PRIOR",
        {"inspect_conflicting_prior_edges": gap.evidence.get("prior_conflicts", []), "require_domain_evidence_for_revision": True},
        {"completion_status_not": "PRIOR_CONFLICT"}, gap.priority, production_safety=safety,
    )


def plan_closed_loop_atlas(*, system_id: str, observation: Optional[Mapping[str, Any]] = None,
                           synthesis: Optional[Mapping[str, Any]] = None,
                           consensus: Optional[Mapping[str, Any]] = None,
                           completion: Optional[Mapping[str, Any]] = None,
                           metadata: Optional[Mapping[str, Any]] = None,
                           max_actions: Optional[int] = None) -> ClosedLoopPlan:
    """Convert current Atlas evidence gaps into safe next-information actions.

    The planner never uses simulator truth as a production acquisition proposal,
    never relaxes existing gates, and does not auto-select a disagreeing backend.
    """
    gaps = (_gate_check_gaps(observation) + _blocked_observation_gaps(observation) + _consensus_gaps(consensus) +
            _completion_gaps(completion) + _synthesis_discovery_gaps(synthesis))
    # deterministic ordering and duplicate protection
    uniq: Dict[str, InformationGap] = {}
    for gap in gaps:
        cur = uniq.get(gap.gap_id)
        if cur is None or gap.priority > cur.priority:
            uniq[gap.gap_id] = gap
    ordered = sorted(uniq.values(), key=lambda g: (-g.priority, g.gap_id))
    if max_actions is not None:
        ordered = ordered[:max(0, int(max_actions))]
    actions = [_action_for_gap(g) for g in ordered]
    return ClosedLoopPlan(
        str(system_id), "ACTION_PLAN_REQUIRED" if actions else "NO_ACTION_NEEDED", ordered, actions,
        metadata={"planner": "closed-loop-atlas-v1", "production_truth_allowed": False,
                  "n_gaps": len(ordered), "n_actions": len(actions), **dict(metadata or {})},
    )


def _after_state(after: Mapping[str, Any], state: str) -> tuple[bool, Mapping[str, Any], Mapping[str, Any]]:
    obs = _observation_report(after.get("observation") if isinstance(after, Mapping) else None)
    smap = _state_map(obs)
    return state not in set(map(str, obs.get("blocked_states", []) or [])), smap.get(state, {}), obs


def _eval_action(action: Mapping[str, Any], after: Mapping[str, Any]) -> ActionEvaluation:
    aid = str(action["action_id"]); gid = str(action["gap_id"]); gap_type = str(action["gap_type"])
    targets = list(map(str, action.get("target_variables", []) or []))
    criteria = action.get("success_criteria") or {}

    if gap_type in {"DATA_INTEGRITY_INSUFFICIENT", "PERSISTENT_EXCITATION_INSUFFICIENT", "SOURCE_IDENTIFIABILITY_INSUFFICIENT"}:
        if not isinstance(after.get("observation"), Mapping):
            return ActionEvaluation(aid, gid, "UNRESOLVED", "no post-action production Observation evidence was supplied")
        obs=_observation_report(after.get("observation")); checks={str(c.get("name")):c for c in obs.get("checks",[]) if isinstance(c,Mapping)}
        check_name={"DATA_INTEGRITY_INSUFFICIENT":"data_integrity","PERSISTENT_EXCITATION_INSUFFICIENT":"persistent_excitation",
                    "SOURCE_IDENTIFIABILITY_INSUFFICIENT":"source_identifiability"}[gap_type]
        c=checks.get(check_name,{})
        ok=_upper(c.get("status"))=="PASS"
        return ActionEvaluation(aid,gid,"RESOLVED" if ok else "UNRESOLVED",
                                f"{check_name} now passes" if ok else f"{check_name} is still not PASS",
                                {"check":check_name,"status":c.get("status"),"metrics":dict(c.get("metrics",{}))})

    if gap_type == "OBSERVATION_BLOCKED":
        if not isinstance(after.get("observation"), Mapping):
            return ActionEvaluation(aid, gid, "UNRESOLVED", "no post-action production Observation evidence was supplied")
        rows=[]; ok=True
        for state in targets:
            unblocked, row, obs = _after_state(after, state)
            route = _state_route_map(obs).get(state, {})
            if route:
                unblocked = str(route.get("eligibility", "")).upper() != "BLOCKED"
                prod=(route.get("evidence") or {}).get("production", {})
                best_name=route.get("preferred_candidate")
                best=prod.get(best_name,{}) if best_name in prod else (next(iter(prod.values())) if prod else {})
                row={**best, **row}
            if not row and not route:
                ok=False
            max_noise=criteria.get("maximum_derivative_noise_ratio")
            if max_noise is not None and row.get("derivative_noise_ratio") is not None:
                unblocked = unblocked and float(row["derivative_noise_ratio"]) < float(max_noise)
            min_var=criteria.get("minimum_variation_to_sensor_floor")
            if min_var is not None and row.get("variation_to_sensor_floor") is not None:
                unblocked = unblocked and float(row["variation_to_sensor_floor"]) >= float(min_var)
            max_dt=criteria.get("maximum_export_dt_s")
            dt=float((obs.get("summary") or {}).get("dt", 0.0) or 0.0)
            if max_dt is not None and dt:
                unblocked = unblocked and dt <= float(max_dt) * (1.0 + 1e-6)
            ok = ok and unblocked
            rows.append({"state": state, "state_status": row.get("status"), "derivative_noise_ratio": row.get("derivative_noise_ratio"),
                         "variation_to_sensor_floor": row.get("variation_to_sensor_floor"), "unblocked": bool(unblocked)})
        return ActionEvaluation(aid, gid, "RESOLVED" if ok else "UNRESOLVED",
                                "production Observation evidence satisfies the requested criterion" if ok else "target remains blocked or outside the requested production criterion",
                                {"states": rows})

    if gap_type in {"DISCOVERY_REQUIRED", "BACKEND_DISAGREEMENT"}:
        cons = after.get("consensus") if isinstance(after.get("consensus"), Mapping) else None
        if cons is None:
            return ActionEvaluation(aid, gid, "UNRESOLVED", "no post-action consensus evidence was supplied")
        rep = _consensus_report(cons); graph = _primary_graph(cons)
        ok = _upper(rep.get("status")) == "CONSENSUS"
        min_j = float(criteria.get("minimum_graph_jaccard", 0.0) or 0.0)
        if rep.get("graph_jaccard") is not None:
            ok = ok and float(rep.get("graph_jaccard")) >= min_j
        if gap_type == "DISCOVERY_REQUIRED":
            ok = ok and all(t in graph for t in targets)
        min_op = criteria.get("minimum_operator_agreement")
        if min_op is not None and rep.get("operator_agreement") is not None:
            ok = ok and float(rep.get("operator_agreement")) >= float(min_op)
        return ActionEvaluation(aid, gid, "RESOLVED" if ok else "UNRESOLVED",
                                "independent backends now agree on the required structure" if ok else "backend structure is still absent or not in consensus",
                                {"consensus_status": rep.get("status"), "graph_jaccard": rep.get("graph_jaccard"),
                                 "operator_agreement": rep.get("operator_agreement"), "graph": dict(graph)})

    if gap_type in {"GRAMMAR_UNCOVERED", "UNIDENTIFIABLE", "PRIOR_CONFLICT"}:
        comp = after.get("completion") if isinstance(after.get("completion"), Mapping) else None
        if comp is None:
            return ActionEvaluation(aid, gid, "UNRESOLVED", "no post-action Completion Certificate was supplied")
        cert = _completion_certificate(comp); status = _upper(cert.get("status"))
        if gap_type == "GRAMMAR_UNCOVERED":
            ok = status == "COMPLETE"
        elif gap_type == "UNIDENTIFIABLE":
            ok = status != "UNIDENTIFIABLE" and bool(cert.get("identifiable", True))
        else:
            ok = status != "PRIOR_CONFLICT" and not cert.get("prior_conflicts")
        return ActionEvaluation(aid, gid, "RESOLVED" if ok else "UNRESOLVED",
                                "post-action Completion evidence cleared the original certification gap" if ok else "the original certification gap remains",
                                {"completion_status": cert.get("status"), "coverage_target": cert.get("coverage_target"),
                                 "coverage_ratio": cert.get("coverage_ratio"), "identifiable": cert.get("identifiable")})

    return ActionEvaluation(aid, gid, "UNRESOLVED", "unsupported gap type for automatic closure evaluation")


def evaluate_closed_loop_plan(plan: Mapping[str, Any] | ClosedLoopPlan, *, after: Mapping[str, Any],
                              metadata: Optional[Mapping[str, Any]] = None) -> ClosedLoopEvaluation:
    pd = plan.to_dict() if isinstance(plan, ClosedLoopPlan) else dict(plan)
    system_id = str(pd.get("system_id", after.get("system_id", "unknown")))
    evaluations = [_eval_action(a, after) for a in pd.get("actions", [])]
    resolved = [e.action_id for e in evaluations if e.status == "RESOLVED"]
    unresolved = [e.action_id for e in evaluations if e.status != "RESOLVED"]
    if evaluations and not unresolved:
        status = "CLOSED"
    elif resolved:
        status = "PARTIAL_PROGRESS"
    elif evaluations:
        status = "UNRESOLVED"
    else:
        status = "NO_ACTIONS"
    return ClosedLoopEvaluation(system_id, status, evaluations, resolved, unresolved,
                                metadata={"evaluator": "closed-loop-atlas-v1", **dict(metadata or {})})


def closed_loop_input_from_dict(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "system_id": str(payload["system_id"]),
        "observation": payload.get("observation"),
        "synthesis": payload.get("synthesis"),
        "consensus": payload.get("consensus"),
        "completion": payload.get("completion"),
        "metadata": dict(payload.get("metadata", {})),
    }

@dataclass
class ClosedLoopCycle:
    system_id: str
    status: str
    plan: ClosedLoopPlan
    evaluation: ClosedLoopEvaluation
    next_plan: ClosedLoopPlan
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": "isgn-closed-loop-cycle-v1",
            "system_id": self.system_id,
            "status": self.status,
            "plan": self.plan.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "next_plan": self.next_plan.to_dict(),
            "metadata": dict(self.metadata),
        }


def run_closed_loop_cycle(*, before: Mapping[str, Any], after: Mapping[str, Any],
                          max_actions: Optional[int] = None,
                          metadata: Optional[Mapping[str, Any]] = None) -> ClosedLoopCycle:
    """Plan one information-acquisition step, evaluate new evidence, and re-plan.

    ``after`` is deliberately not merged with ``before``.  Only explicitly
    supplied new production evidence can resolve a gap.  The next plan is based
    on the new evidence state, so unresolved or newly exposed gaps remain visible.
    """
    b = closed_loop_input_from_dict(before)
    plan = plan_closed_loop_atlas(**b, max_actions=max_actions)
    evaluation = evaluate_closed_loop_plan(plan, after=after)
    a = closed_loop_input_from_dict({"system_id": after.get("system_id", b["system_id"]), **dict(after)})
    next_plan = plan_closed_loop_atlas(**a, max_actions=max_actions)
    if evaluation.status == "CLOSED" and next_plan.status == "NO_ACTION_NEEDED":
        status = "CYCLE_CLOSED"
    elif evaluation.status in {"CLOSED", "PARTIAL_PROGRESS"}:
        status = "CYCLE_ADVANCED"
    else:
        status = "CYCLE_UNRESOLVED"
    return ClosedLoopCycle(str(b["system_id"]), status, plan, evaluation, next_plan,
                           metadata={"engine": "closed-loop-atlas-v1", **dict(metadata or {})})
