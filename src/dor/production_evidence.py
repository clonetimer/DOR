from __future__ import annotations

"""Production-evidence qualification for DOR deployments.

This module deliberately separates software correctness from deployment evidence.
A project can have a clean regression suite and still be blocked for production
because subgroup performance, distribution shift, operational alert burden, or
real fault-injection evidence is missing.
"""

from pathlib import Path
from typing import Any
import json

CONTRACT = "DOR-PRODUCTION-EVIDENCE-GATE/1.0"
EVIDENCE_CONTRACT = "DOR-PRODUCTION-EVIDENCE/1.0"
VALID_STATES = {"PASS", "BLOCKED", "NOT_RUN"}

PROFILE_GATES: dict[str, tuple[str, ...]] = {
    "ASSISTED_POC": (
        "SOFTWARE_REGRESSION",
        "RELEASE_HYGIENE",
        "REAL_TARGET_DATA",
        "ROLE_SEPARATION",
        "SUBGROUP_REPORTING",
    ),
    "ASSISTED_PRODUCTION": (
        "SOFTWARE_REGRESSION",
        "RELEASE_HYGIENE",
        "REAL_TARGET_DATA",
        "ROLE_SEPARATION",
        "SUBGROUP_REPORTING",
        "PROSPECTIVE_ACCEPTANCE_POLICY",
        "DISTRIBUTION_SHIFT",
        "MISSING_SENSOR",
        "NEW_SESSION_OR_ASSET",
        "DEPLOYMENT_OPERATIONS",
    ),
    "SAFETY_CRITICAL_AUTONOMY": (
        "SOFTWARE_REGRESSION",
        "RELEASE_HYGIENE",
        "REAL_TARGET_DATA",
        "ROLE_SEPARATION",
        "SUBGROUP_REPORTING",
        "PROSPECTIVE_ACCEPTANCE_POLICY",
        "DISTRIBUTION_SHIFT",
        "MISSING_SENSOR",
        "NEW_SESSION_OR_ASSET",
        "DEPLOYMENT_OPERATIONS",
        "HAZARD_ANALYSIS",
        "CLOSED_LOOP_SAFETY",
        "REAL_FAULT_INJECTION",
    ),
}


def _load(value: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return json.loads(Path(value).read_text(encoding="utf-8"))


def validate_production_evidence(evidence: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if evidence.get("contract") != EVIDENCE_CONTRACT:
        errors.append(f"contract must be {EVIDENCE_CONTRACT}")
    if not evidence.get("target_use"):
        errors.append("target_use is required")
    seen: set[str] = set()
    for i, gate in enumerate(evidence.get("gates", [])):
        gid = str(gate.get("gate", "")).strip()
        if not gid:
            errors.append(f"gate[{i}] missing gate")
            continue
        if gid in seen:
            errors.append(f"duplicate gate: {gid}")
        seen.add(gid)
        state = str(gate.get("status", "")).upper()
        if state not in VALID_STATES:
            errors.append(f"{gid}: status must be one of {sorted(VALID_STATES)}")
        if not gate.get("reason"):
            errors.append(f"{gid}: reason is required")
        if "evidence" not in gate or not isinstance(gate.get("evidence"), list):
            errors.append(f"{gid}: evidence must be a list")
    return errors


def evaluate_production_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    errors = validate_production_evidence(evidence)
    gates = {str(x.get("gate")): x for x in evidence.get("gates", []) if x.get("gate")}
    profiles: dict[str, Any] = {}

    for profile, required in PROFILE_GATES.items():
        details = []
        blockers = []
        for gate_id in required:
            src = gates.get(gate_id)
            if src is None:
                state = "BLOCKED"
                reason = "required deployment-evidence gate is missing"
                refs: list[Any] = []
            else:
                state = str(src.get("status", "BLOCKED")).upper()
                reason = str(src.get("reason", ""))
                refs = list(src.get("evidence", []))
            passed = state == "PASS"
            details.append({
                "gate": gate_id,
                "status": "PASS" if passed else "BLOCKED",
                "source_status": state,
                "reason": reason,
                "evidence": refs,
            })
            if not passed:
                blockers.append(gate_id)
        profiles[profile] = {
            "status": "SUPPORTED" if not blockers and not errors else "BLOCKED",
            "blocking_gates": blockers,
            "gates": details,
        }

    return {
        "contract": CONTRACT,
        "target_use": evidence.get("target_use"),
        "target_domain": evidence.get("target_domain"),
        "profiles": profiles,
        "errors": errors,
        "safety": {
            "software_tests_do_not_imply_production_readiness": True,
            "aggregate_metrics_do_not_waive_subgroup_evidence": True,
            "posthoc_thresholds_do_not_retroactively_qualify_test_results": True,
            "cross_asset_or_distribution_shift_evidence_must_be_explicit": True,
            "root_cause_contract_tests_do_not_equal_real_fault_injection_accuracy": True,
        },
    }


def evaluate_production_evidence_from_file(path: str | Path) -> dict[str, Any]:
    return evaluate_production_evidence(_load(path))
