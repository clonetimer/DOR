from __future__ import annotations

"""DOR v0.6 propagation/root-cause development prototype.

This module deliberately requires more than localization + graph adjacency before
it can claim a root cause.  A root-cause claim requires:

* a v0.5 localized fault hypothesis with explicit, timestamped support;
* provenance-bound directed topology paths;
* downstream evidence whose timing is compatible with propagation;
* sufficient coverage of the anomalous evidence set;
* no unresolved equivalent minimal root set.

The module consumes no test/fault log.  Ground truth is reserved for post-hoc
validation and rule changes must not be selected on the held-out truth.
"""

from itertools import combinations
from pathlib import Path
from typing import Any
import json
import math
import pandas as pd

from .ontology import ontology_index, validate_fault_ontology, validate_mechanism_component_map
from .localization import _read_evidence

ROOT_CAUSE_CONTRACT = "DOR-ROOT-CAUSE-RESULT/0.6"
PROPAGATION_EDGE_RELATIONS = {"POWER_FLOW", "FLUID_FLOW", "MECHANICAL", "THERMAL", "CONTROL", "SIGNAL"}
DOWNSTREAM_STATES = {"ABNORMAL", "HIGH", "LOW", "NO_RESPONSE", "FAILED_OFF", "DEGRADED"}
LOCAL_FAULT_STATES = DOWNSTREAM_STATES | {"STUCK", "OFFSET", "CLASS", "OPEN", "CLOSED"}


def _as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _time_value(v: Any) -> float | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    try:
        return float(v)
    except Exception:
        pass
    try:
        ts = pd.Timestamp(v)
        if pd.isna(ts):
            return None
        return float(ts.timestamp())
    except Exception:
        return None


def _graph(ontology: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    g: dict[str, list[dict[str, Any]]] = {}
    for e in _as_list(ontology.get("topology_edges")):
        rel = str(e.get("relation", "")).upper()
        if rel not in PROPAGATION_EDGE_RELATIONS:
            continue
        g.setdefault(str(e.get("source")), []).append(e)
    return g


def _find_path(g: dict[str, list[dict[str, Any]]], source: str, target: str, max_depth: int = 8) -> list[dict[str, Any]] | None:
    if source == target:
        return []
    q: list[tuple[str, list[dict[str, Any]]]] = [(source, [])]
    seen = {source}
    while q:
        node, path = q.pop(0)
        if len(path) >= max_depth:
            continue
        for e in g.get(node, []):
            nxt = str(e.get("target"))
            npath = path + [e]
            if nxt == target:
                return npath
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, npath))
    return None


def _mechanism_components(mechanism_map: dict[str, Any] | None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if not mechanism_map:
        return out
    for b in _as_list(mechanism_map.get("bindings")):
        out.setdefault(str(b.get("mechanism_id")), []).extend(str(x) for x in _as_list(b.get("component_ids")))
    return out


def _evidence_components(row: dict[str, Any], idx: dict[str, Any], mech: dict[str, list[str]]) -> list[str]:
    st = str(row.get("scope_type", "")).upper()
    sid = str(row.get("scope_id", ""))
    state = str(row.get("state", "")).upper()
    if st == "COMPONENT":
        return [sid] if sid in idx["components"] else []
    if st == "MECHANISM":
        return list(dict.fromkeys(mech.get(sid, [])))
    if st == "COMMAND":
        cmd = idx["commands"].get(sid)
        return [str(cmd.get("target_component_id"))] if cmd else []
    if st == "SENSOR":
        sensor = idx["sensors"].get(sid)
        if not sensor:
            return []
        # A sensor-state fault signature is evidence about sensor health, not
        # about the observed physical component.
        if state in {"STUCK", "OFFSET", "MISSING", "OOD"}:
            hc = str(sensor.get("health_component_id", ""))
            return [hc] if hc in idx["components"] else []
        return [str(x) for x in _as_list(sensor.get("observes_components")) if str(x) in idx["components"]]
    return []


def _is_explainable_anomaly(row: dict[str, Any]) -> bool:
    if str(row.get("validity", "")).upper() != "VALID":
        return False
    if str(row.get("scope_type", "")).upper() == "COMMAND":
        return False
    return str(row.get("state", "")).upper() in LOCAL_FAULT_STATES


def _path_public(path: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source": str(e.get("source")),
            "target": str(e.get("target")),
            "relation": str(e.get("relation", "")).upper(),
            "provenance": str(e.get("provenance", "")),
        }
        for e in path
    ]


def infer_root_causes(
    ontology: dict[str, Any],
    localization: dict[str, Any],
    evidence_rows: list[dict[str, Any]],
    *,
    mechanism_map: dict[str, Any] | None = None,
    max_root_set_size: int = 3,
    coverage_required: float = 1.0,
    alternative_margin: float = 0.05,
    temporal_tolerance: float = 0.0,
    max_path_depth: int = 8,
) -> dict[str, Any]:
    ov = validate_fault_ontology(ontology)
    if ov.state != "PASS":
        raise ValueError("invalid ontology: " + "; ".join(ov.errors))
    if mechanism_map:
        mv = validate_mechanism_component_map(mechanism_map, ontology)
        if mv.state != "PASS":
            raise ValueError("invalid mechanism map: " + "; ".join(mv.errors))
    if localization.get("contract") != "DOR-LOCALIZATION-RESULT/0.5":
        raise ValueError("root-cause v0.6 requires DOR-LOCALIZATION-RESULT/0.5")

    loc_status = str(localization.get("status", ""))
    if loc_status in {"UNKNOWN", "ABSTAIN"}:
        return _empty_result(
            "UNKNOWN",
            f"localization status {loc_status} does not support root-cause reasoning",
            localization,
            evidence_rows,
            coverage_required,
            max_root_set_size,
        )

    idx = ontology_index(ontology)
    mech = _mechanism_components(mechanism_map)
    g = _graph(ontology)
    valid_rows = [r for r in evidence_rows if str(r.get("validity", "")).upper() == "VALID"]
    ev_by_id = {str(r.get("evidence_id")): r for r in valid_rows}

    anomalies: dict[str, dict[str, Any]] = {}
    for r in valid_rows:
        if not _is_explainable_anomaly(r):
            continue
        comps = _evidence_components(r, idx, mech)
        if not comps:
            continue
        anomalies[str(r.get("evidence_id"))] = {"row": r, "components": comps, "strength": float(r.get("strength", 0.0))}

    if not anomalies:
        return _empty_result(
            "UNKNOWN",
            "no VALID component-resolvable anomaly evidence is available for root-cause reasoning",
            localization,
            evidence_rows,
            coverage_required,
            max_root_set_size,
        )

    candidates: list[dict[str, Any]] = []
    for h in _as_list(localization.get("fault_hypotheses")):
        fid = str(h.get("fault_id"))
        target = str(h.get("target_component_id"))
        matched_ids = [str(x) for x in _as_list(h.get("matched_evidence"))]
        matched_rows = [ev_by_id[x] for x in matched_ids if x in ev_by_id]
        matched_times = [_time_value(r.get("timestamp")) for r in matched_rows]
        matched_times = [t for t in matched_times if t is not None]
        root_time = max(matched_times) if matched_times else None

        covered = set(x for x in matched_ids if x in anomalies)
        downstream: list[dict[str, Any]] = []
        temporal_contradictions: list[dict[str, Any]] = []
        unprovenanced_paths: list[dict[str, Any]] = []

        if root_time is not None:
            for eid, meta in anomalies.items():
                if eid in covered:
                    continue
                row = meta["row"]
                ev_t = _time_value(row.get("timestamp"))
                for comp in meta["components"]:
                    path = _find_path(g, target, comp, max_depth=max_path_depth)
                    if path is None or not path:
                        continue
                    pub = _path_public(path)
                    if any(not e["provenance"].strip() for e in pub):
                        unprovenanced_paths.append({"evidence_id": eid, "target_component": comp, "path": pub})
                        continue
                    if ev_t is None:
                        continue
                    if ev_t + temporal_tolerance < root_time:
                        temporal_contradictions.append({
                            "evidence_id": eid,
                            "evidence_time": ev_t,
                            "root_support_time": root_time,
                            "path": pub,
                        })
                        continue
                    covered.add(eid)
                    downstream.append({
                        "evidence_id": eid,
                        "component_id": comp,
                        "evidence_time": ev_t,
                        "root_support_time": root_time,
                        "lag": ev_t - root_time,
                        "path": pub,
                    })
                    break

        total_weight = sum(meta["strength"] for meta in anomalies.values()) or 1.0
        covered_weight = sum(anomalies[eid]["strength"] for eid in covered)
        coverage = covered_weight / total_weight
        # A v0.6 root-cause claim requires temporal evidence and at least one
        # non-local propagation explanation.  Otherwise the result remains at
        # the v0.5 localization level and is not renamed "root cause".
        # A candidate is allowed to participate when it has timestamped local
        # support and at least one provenance-bound, temporally compatible
        # downstream explanation.  Earlier or unprovenanced observations are
        # *not* silently assigned to this root; they remain uncovered and may
        # require another concurrent root or force UNKNOWN.
        eligible = bool(root_time is not None and downstream)
        candidates.append({
            "fault_id": fid,
            "target_component_id": target,
            "localization_score": float(h.get("score", 0.0)),
            "root_support_time": root_time,
            "covered_evidence": sorted(covered),
            "coverage_fraction": coverage,
            "downstream_explanations": downstream,
            "temporal_contradictions": temporal_contradictions,
            "unprovenanced_paths": unprovenanced_paths,
            "root_candidate_eligible": eligible,
            "exclusion_group": h.get("exclusion_group"),
        })

    eligible = [c for c in candidates if c["root_candidate_eligible"]]
    if not eligible:
        return _result(
            "UNKNOWN",
            "localized fault hypotheses exist, but none has timestamped, provenance-bound downstream propagation evidence",
            candidates,
            [],
            [],
            anomalies,
            localization,
            coverage_required,
            max_root_set_size,
            False,
            False,
        )

    anomaly_ids = set(anomalies)
    root_sets: list[dict[str, Any]] = []
    max_k = min(max_root_set_size, len(eligible))
    for k in range(1, max_k + 1):
        for combo in combinations(eligible, k):
            groups = [str(c.get("exclusion_group") or c["fault_id"]) for c in combo]
            if len(groups) != len(set(groups)):
                continue
            cov: set[str] = set()
            for c in combo:
                cov.update(c["covered_evidence"])
            total_weight = sum(anomalies[x]["strength"] for x in anomaly_ids) or 1.0
            covered_weight = sum(anomalies[x]["strength"] for x in cov if x in anomalies)
            frac = covered_weight / total_weight
            if frac + 1e-12 < coverage_required:
                continue
            score = sum(float(c["localization_score"]) for c in combo) + frac
            root_sets.append({
                "fault_ids": sorted(c["fault_id"] for c in combo),
                "component_ids": sorted({c["target_component_id"] for c in combo}),
                "coverage_fraction": frac,
                "score": score,
                "covered_evidence": sorted(cov),
            })
        if root_sets:
            break  # minimal-cardinality root sets only

    if not root_sets:
        unexplained = sorted(anomaly_ids - set().union(*(set(c["covered_evidence"]) for c in eligible))) if eligible else sorted(anomaly_ids)
        return _result(
            "UNKNOWN",
            "no admissible root-cause set explains the required fraction of valid anomaly evidence",
            candidates,
            [],
            unexplained,
            anomalies,
            localization,
            coverage_required,
            max_root_set_size,
            False,
            False,
        )

    root_sets.sort(key=lambda x: (-x["score"], x["fault_ids"]))
    best = root_sets[0]["score"]
    retained = [x for x in root_sets if best - x["score"] <= alternative_margin]
    if len(retained) > 1:
        return _result(
            "ALTERNATIVES",
            "multiple minimal root-cause sets remain observationally equivalent within the configured margin",
            candidates,
            retained,
            [],
            anomalies,
            localization,
            coverage_required,
            max_root_set_size,
            False,
            True,
        )

    return _result(
        "SUPPORTED",
        "a unique minimal root-cause set has timestamped local support and provenance-bound temporally compatible propagation coverage",
        candidates,
        retained,
        [],
        anomalies,
        localization,
        coverage_required,
        max_root_set_size,
        True,
        True,
    )


def _result(
    status: str,
    reason: str,
    candidates: list[dict[str, Any]],
    root_sets: list[dict[str, Any]],
    unexplained: list[str],
    anomalies: dict[str, Any],
    localization: dict[str, Any],
    coverage_required: float,
    max_root_set_size: int,
    root_cause_claimed: bool,
    propagation_inferred: bool,
) -> dict[str, Any]:
    return {
        "contract": ROOT_CAUSE_CONTRACT,
        "status": status,
        "reason": reason,
        "root_cause_set": root_sets[0]["fault_ids"] if status == "SUPPORTED" and root_sets else [],
        "candidate_root_sets": root_sets,
        "root_candidates": candidates,
        "unexplained_evidence": unexplained,
        "evidence_summary": {
            "explainable_anomaly_count": len(anomalies),
            "explainable_anomaly_ids": sorted(anomalies),
        },
        "localization_input": {
            "contract": localization.get("contract"),
            "status": localization.get("status"),
        },
        "policy": {
            "coverage_required": coverage_required,
            "max_root_set_size": max_root_set_size,
            "minimal_cardinality_preferred": True,
        },
        "safety": {
            "root_cause_claimed": root_cause_claimed,
            "propagation_inferred": propagation_inferred,
            "earliest_anomaly_is_not_root_cause_rule": True,
            "topology_only_is_not_sufficient": True,
            "timestamped_fault_specific_support_required": True,
            "provenance_bound_path_required": True,
            "test_truth_used_for_reasoning": False,
            "concurrent_root_causes_allowed": True,
        },
    }


def _empty_result(status: str, reason: str, localization: dict[str, Any], evidence_rows: list[dict[str, Any]], coverage_required: float, max_root_set_size: int) -> dict[str, Any]:
    return {
        "contract": ROOT_CAUSE_CONTRACT,
        "status": status,
        "reason": reason,
        "root_cause_set": [],
        "candidate_root_sets": [],
        "root_candidates": [],
        "unexplained_evidence": [],
        "evidence_summary": {"total_input_evidence": len(evidence_rows)},
        "localization_input": {"contract": localization.get("contract"), "status": localization.get("status")},
        "policy": {"coverage_required": coverage_required, "max_root_set_size": max_root_set_size, "minimal_cardinality_preferred": True},
        "safety": {
            "root_cause_claimed": False,
            "propagation_inferred": False,
            "earliest_anomaly_is_not_root_cause_rule": True,
            "topology_only_is_not_sufficient": True,
            "timestamped_fault_specific_support_required": True,
            "provenance_bound_path_required": True,
            "test_truth_used_for_reasoning": False,
            "concurrent_root_causes_allowed": True,
        },
    }


def infer_root_causes_from_files(
    ontology_path: str | Path,
    localization_path: str | Path,
    evidence_path: str | Path,
    *,
    mechanism_map_path: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    ontology = json.loads(Path(ontology_path).read_text(encoding="utf-8"))
    localization = json.loads(Path(localization_path).read_text(encoding="utf-8"))
    mechanism_map = json.loads(Path(mechanism_map_path).read_text(encoding="utf-8")) if mechanism_map_path else None
    evidence = _read_evidence(evidence_path)
    return infer_root_causes(ontology, localization, evidence, mechanism_map=mechanism_map, **kwargs)
