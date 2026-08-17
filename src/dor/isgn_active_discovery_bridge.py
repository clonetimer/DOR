from __future__ import annotations

"""Fail-closed DOR adapter for ISGN Active Discovery artifacts."""

from math import isfinite
from typing import Any, Mapping

ACTIVE_DISCOVERY_ADVISORY_CONTRACT = "DOR-ISGN-ACTIVE-DISCOVERY-ADVISORY/1.0"
_IDENTIFIABILITY = {
    "UNIQUE_SUPPORTED", "ALTERNATIVES", "UNIDENTIFIABLE",
    "HYPOTHESIS_CLASS_INADEQUATE",
}
_STABILITY = {"STABLE", "UNSTABLE", "INSUFFICIENT"}
_ALLOWED_MODES = {"ADVISORY_ONLY", "OFFLINE_REPLAY", "SIMULATION", "LIVE"}
_ADEQUACY = {"ADEQUATE", "HYPOTHESIS_CLASS_INADEQUATE"}


def _truth_quarantine_issues(payload: Mapping[str, Any]) -> list[str]:
    tq = payload.get("truth_quarantine")
    if not isinstance(tq, Mapping):
        return ["explicit truth_quarantine attestation is required"]
    required = ("truth_channels_allowed", "held_out_truth_loaded", "true_structure_loaded")
    missing = [name for name in required if name not in tq]
    issues = [f"truth_quarantine missing {name}" for name in missing]
    for name in required:
        if name in tq and tq[name] is not False:
            issues.append(f"truth_quarantine.{name} must be exactly false")
    if payload.get("truth_channels_allowed", False) is not False:
        issues.append("top-level truth_channels_allowed must be exactly false")
    return issues


def _posterior_rows(payload: Mapping[str, Any]) -> tuple[list[dict[str, float | str]], float, list[str]]:
    raw = payload.get("structure_posterior")
    if not isinstance(raw, Mapping) or not raw:
        return [], 0.0, ["structure_posterior must be a non-empty mapping"]
    rows: list[tuple[str, float]] = []
    issues: list[str] = []
    for key, value in raw.items():
        sid = str(key).strip()
        try:
            mass = float(value)
        except (TypeError, ValueError):
            issues.append(f"posterior mass for {sid or '<empty>'} is not numeric")
            continue
        if not sid:
            issues.append("structure_id must be non-empty")
        elif not isfinite(mass) or mass < 0.0 or mass > 1.0:
            issues.append(f"posterior mass for {sid} must be finite and in [0, 1]")
        else:
            rows.append((sid, mass))
    total = sum(mass for _, mass in rows)
    if not issues and abs(total - 1.0) > 1e-6:
        issues.append(f"structure posterior mass must sum to 1, got {total:.12g}")
    rows.sort(key=lambda item: (-item[1], item[0]))
    return [{"structure_id": sid, "posterior_mass": mass} for sid, mass in rows], total, issues


def _sanitize_experiment(value: Any) -> tuple[dict[str, Any] | None, list[str]]:
    if value is None:
        return None, []
    if not isinstance(value, Mapping):
        return None, ["next_experiment must be an object when supplied"]
    issues: list[str] = []
    experiment_id = str(value.get("experiment_id", "")).strip()
    mode = str(value.get("execution_mode", "ADVISORY_ONLY")).upper()
    if not experiment_id:
        issues.append("next_experiment.experiment_id is required")
    if mode not in _ALLOWED_MODES:
        issues.append(f"unsupported next_experiment.execution_mode {mode!r}")
    if mode == "LIVE":
        issues.append("DOR never forwards live-actuation recommendations")
    try:
        repeats = int(value.get("repeats", 1))
        if isinstance(value.get("repeats", 1), bool) or repeats < 1:
            raise ValueError
    except (TypeError, ValueError):
        repeats = 0
        issues.append("next_experiment.repeats must be a positive integer")
    raw_cost = value.get("cost", value.get("unit_cost"))
    cost = None
    if raw_cost is not None:
        try:
            cost = float(raw_cost)
            if not isfinite(cost) or cost <= 0:
                raise ValueError
        except (TypeError, ValueError):
            cost = None
            issues.append("next_experiment cost must be finite and positive")
    sanitized = {
        "experiment_id": experiment_id,
        "execution_mode": mode,
        "repeats": repeats,
        "cost": cost,
        "safety_eligible": value.get("safety_eligible") is True,
        "execution_allowed": False,
    }
    # Deliberately omit payload, command, actuator and safety-evidence contents.
    return sanitized, issues


def _sanitize_predictive_adequacy(value: Any) -> tuple[dict[str, Any] | None, list[str]]:
    if value is None:
        return None, []
    if not isinstance(value, Mapping):
        return None, ["predictive_adequacy must be an object when supplied"]
    issues: list[str] = []
    status = str(value.get("status", "")).upper()
    if status not in _ADEQUACY:
        issues.append(f"unsupported predictive-adequacy status {status!r}")
    if value.get("truth_channels_allowed") is not False:
        issues.append("predictive_adequacy.truth_channels_allowed must be exactly false")
    sanitized: dict[str, Any] = {"status": status, "truth_channels_allowed": False}
    for name in ("min_mahalanobis", "threshold"):
        if name not in value:
            continue
        try:
            number = float(value[name])
            if not isfinite(number) or number < 0:
                raise ValueError
            sanitized[name] = number
        except (TypeError, ValueError):
            issues.append(f"predictive_adequacy.{name} must be finite and non-negative")
    for name in ("nearest_structure", "nearest_particle"):
        if name in value:
            text = str(value[name]).strip()
            if not text:
                issues.append(f"predictive_adequacy.{name} must be non-empty")
            else:
                sanitized[name] = text
    return sanitized, issues


def build_active_discovery_advisory(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Convert Active Discovery output into planning-only, non-executable evidence.

    Missing quarantine attestations, malformed posterior mass, inconsistent uniqueness,
    or any live recommendation fail closed. Arbitrary experiment payloads are never
    forwarded across the DOR boundary.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("active-discovery payload must be a mapping")

    issues = _truth_quarantine_issues(payload)
    structures, posterior_sum, posterior_issues = _posterior_rows(payload)
    issues.extend(posterior_issues)

    ident_raw = payload.get("identifiability") or payload.get("identifiability_decision")
    ident = ident_raw if isinstance(ident_raw, Mapping) else {}
    ident_status = ident.get("status")
    if ident_status is not None:
        ident_status = str(ident_status).upper()
        if ident_status not in _IDENTIFIABILITY:
            issues.append(f"unsupported identifiability status {ident_status!r}")

    stability_raw = payload.get("evidence_stability")
    stability = stability_raw if isinstance(stability_raw, Mapping) else {}
    stability_status = stability.get("status")
    if stability_status is not None:
        stability_status = str(stability_status).upper()
        if stability_status not in _STABILITY:
            issues.append(f"unsupported evidence-stability status {stability_status!r}")

    coverage_complete = payload.get("hypothesis_space_coverage_complete")
    if ident_status == "UNIQUE_SUPPORTED":
        if stability_status != "STABLE":
            issues.append("UNIQUE_SUPPORTED requires STABLE evidence-stability attestation")
        if coverage_complete is not True:
            issues.append("UNIQUE_SUPPORTED requires explicit complete hypothesis-space coverage")

    next_value = payload.get("next_experiment") or payload.get("experiment_recommendation")
    next_experiment, experiment_issues = _sanitize_experiment(next_value)
    issues.extend(experiment_issues)
    predictive_adequacy, adequacy_issues = _sanitize_predictive_adequacy(payload.get("predictive_adequacy"))
    issues.extend(adequacy_issues)

    status = "BLOCKED" if issues else "ADVISORY_ONLY"
    reason = (
        "; ".join(issues)
        if issues else
        "ISGN Active Discovery is planning evidence only; certified mechanism identity still requires the existing ISGN certification/differential path"
    )
    remaining = ident.get("remaining_discriminating_experiments", [])
    if not isinstance(remaining, (list, tuple)) or any(not str(x).strip() for x in remaining):
        issues.append("remaining_discriminating_experiments must be an array of non-empty ids")
        status = "BLOCKED"
        reason = "; ".join(issues)

    return {
        "contract": ACTIVE_DISCOVERY_ADVISORY_CONTRACT,
        "status": status,
        "candidate_structures": structures,
        "posterior_mass_sum": posterior_sum,
        "next_experiment": next_experiment,
        "predictive_adequacy": predictive_adequacy,
        "identifiability_status": ident_status,
        "evidence_stability_status": stability_status,
        "hypothesis_space_coverage_complete": coverage_complete is True,
        "remaining_discriminating_experiments": [str(x) for x in remaining] if isinstance(remaining, (list, tuple)) else [],
        "admission_issues": issues,
        "safety": {
            "truth_channels_allowed": False,
            "truth_quarantine_verified": not _truth_quarantine_issues(payload),
            "diagnostic_claim_allowed": False,
            "fault_identification_allowed": False,
            "root_cause_claim_allowed": False,
            "mechanism_identity_certified": False,
            "proposal_is_evidence": False,
            "experiment_execution_allowed": False,
            "raw_experiment_payload_forwarded": False,
        },
        "reason": reason,
    }
