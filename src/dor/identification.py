from __future__ import annotations

"""DOR v0.7 fault-identification orchestration.

Identification is deliberately separated from localization and root cause:

* localization asks *where / which fault hypothesis is supported?*
* root cause asks *which supported faults explain downstream effects?*
* identification asks *what state/parameter/structure change can actually be
  claimed for those supported faults?*

DOR never turns a residual, classifier score, or model disagreement directly
into a physical parameter-change claim.  Mechanism claims require an explicit
ISGN-derived identification evidence bundle with a COMPLETE baseline
certificate, valid abnormal-window evidence, and per-change identifiability.
Discrete/sensor state faults may be identified semantically from the validated
Fault Ontology without forcing ISGN to run.
"""

from pathlib import Path
from typing import Any, Iterable
import json
import hashlib

from .ontology import ontology_index, validate_fault_ontology

IDENTIFICATION_CONTRACT = "DOR-FAULT-IDENTIFICATION-RESULT/0.7"
IDENTIFICATION_REQUEST_CONTRACT = "DOR-FAULT-IDENTIFICATION-REQUEST/0.7"
ISGN_EVIDENCE_CONTRACT = "DOR-ISGN-IDENTIFICATION-EVIDENCE/0.7"
IDENTIFICATION_STATES = {"SUPPORTED", "ALTERNATIVES", "UNKNOWN", "ABSTAIN"}
GENERIC_MODES = {"state_fault", "abnormal", "unknown", "unspecified", "generic_fault"}


def _load(x: str | Path | dict[str, Any] | None) -> dict[str, Any] | None:
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    return json.loads(Path(x).read_text(encoding="utf-8"))


def _candidate_fault_ids(localization: dict[str, Any], root_cause: dict[str, Any] | None) -> tuple[list[str], str]:
    """Return candidate fault IDs and the diagnostic source used.

    A unique root set has priority.  For root-cause alternatives we retain the
    union rather than silently choosing one.  If root cause is UNKNOWN, fault
    identification may still proceed on localized hypotheses but must not call
    them roots.
    """
    if root_cause:
        st = str(root_cause.get("status", "UNKNOWN")).upper()
        if st == "SUPPORTED" and root_cause.get("root_cause_set"):
            return list(dict.fromkeys(map(str, root_cause["root_cause_set"]))), "ROOT_CAUSE_SET"
        if st == "ALTERNATIVES":
            ids: list[str] = []
            for item in root_cause.get("candidate_root_sets", []):
                for fid in item.get("fault_ids", []):
                    if str(fid) not in ids:
                        ids.append(str(fid))
            if ids:
                return ids, "ROOT_CAUSE_ALTERNATIVES"
    ids = [str(x.get("fault_id")) for x in localization.get("fault_hypotheses", []) if x.get("fault_id")]
    return list(dict.fromkeys(ids)), "LOCALIZATION"


def _sha256_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build_fault_identification_request(
    ontology: dict[str, Any],
    localization: dict[str, Any],
    *,
    root_cause: dict[str, Any] | None = None,
    case_id: str | None = None,
    window_start: float | None = None,
    window_end: float | None = None,
    window_evidence_validity: str = "VALID",
    artifact_bindings: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """Plan v0.7 identification routes before executing ISGN.

    Discrete/sensor hypotheses are routed to ontology-semantic identification.
    PARAMETRIC hypotheses require an authorized abnormal window and ISGN.
    The request does not contain fault truth or test labels.
    """
    ov = validate_fault_ontology(ontology)
    if ov.state != "PASS":
        return {"contract": IDENTIFICATION_REQUEST_CONTRACT, "status": "BLOCKED", "reason": "fault ontology is invalid", "case_id": case_id, "routes": [], "engine_admissions": {"ISGN": "BLOCKED"}}
    idx = ontology_index(ontology)
    fault_ids, source = _candidate_fault_ids(localization, root_cause)
    routes = []
    need_isgn = False
    for fid in fault_ids:
        f = idx["faults"].get(fid)
        if f is None:
            routes.append({"fault_id": fid, "route": "UNSUPPORTED", "reason": "fault absent from ontology"})
            continue
        cat = str(f.get("category", "COMPONENT")).upper()
        if cat == "PARAMETRIC":
            need_isgn = True
            routes.append({"fault_id": fid, "component_id": f.get("target_component_id"), "route": "ISGN_DIFFERENTIAL", "reason": "parameter/structure claims require certified mechanism differential evidence"})
        else:
            routes.append({"fault_id": fid, "component_id": f.get("target_component_id"), "route": "DISCRETE_SEMANTIC", "reason": "explicit discrete/sensor fault mode may be identified from validated ontology evidence"})
    validity = str(window_evidence_validity).upper()
    if need_isgn:
        valid_window = validity == "VALID" and window_start is not None and window_end is not None and float(window_end) > float(window_start)
        isgn_admission = "RUN" if valid_window else "BLOCKED"
        status = "PLANNED" if valid_window else "BLOCKED"
        reason = "identification routes prepared" if valid_window else "ISGN differential identification requires a finite authorized abnormal window with VALID evidence"
    else:
        isgn_admission = "SKIP"
        status = "PLANNED"
        reason = "all retained fault hypotheses use the discrete/sensor semantic identification route"
    return {
        "contract": IDENTIFICATION_REQUEST_CONTRACT,
        "status": status,
        "reason": reason,
        "case_id": case_id,
        "candidate_source": source,
        "candidate_fault_ids": fault_ids,
        "routes": routes,
        "window_authorization": {"start": window_start, "end": window_end, "evidence_validity": validity},
        "engine_admissions": {"ISGN": isgn_admission},
        "artifact_bindings": artifact_bindings or {},
        "safety": {"test_truth_allowed": False, "isgn_not_forced_for_discrete_faults": True, "parametric_claim_requires_valid_window": True},
    }


def build_identification_request_from_files(
    ontology_path: str | Path,
    localization_path: str | Path,
    *,
    root_cause_path: str | Path | None = None,
    case_id: str | None = None,
    window_start: float | None = None,
    window_end: float | None = None,
    window_evidence_validity: str = "VALID",
) -> dict[str, Any]:
    ontology = _load(ontology_path) or {}
    localization = _load(localization_path) or {}
    root_cause = _load(root_cause_path)
    bindings = {
        "ontology_sha256": _sha256_path(ontology_path),
        "localization_sha256": _sha256_path(localization_path),
        "root_cause_sha256": _sha256_path(root_cause_path),
    }
    return build_fault_identification_request(ontology, localization, root_cause=root_cause, case_id=case_id, window_start=window_start, window_end=window_end, window_evidence_validity=window_evidence_validity, artifact_bindings=bindings)


def validate_isgn_identification_evidence(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if payload.get("contract") != ISGN_EVIDENCE_CONTRACT:
        errors.append(f"contract must be {ISGN_EVIDENCE_CONTRACT}")
    if str(payload.get("baseline_certificate_status", "")).upper() != "COMPLETE":
        errors.append("baseline_certificate_status must be COMPLETE for physical mechanism identification")
    if str(payload.get("window_evidence_validity", "")).upper() != "VALID":
        errors.append("window_evidence_validity must be VALID")
    status = str(payload.get("status", "")).upper()
    if status not in {"SUPPORTED", "ALTERNATIVES", "UNIDENTIFIABLE", "UNSUPPORTED"}:
        errors.append("invalid ISGN identification status")
    for i, ch in enumerate(payload.get("changes", [])):
        if not str(ch.get("component_id", "")).strip():
            errors.append(f"change[{i}] missing component_id")
        if str(ch.get("change_type", "")).upper() not in {"PARAMETER_DRIFT", "STRUCTURE_CHANGE", "CONSTITUTIVE_CHANGE", "RESIDUAL_TERM"}:
            errors.append(f"change[{i}] unsupported change_type")
        if "identifiable" not in ch:
            errors.append(f"change[{i}] missing identifiable")
        if "supported" not in ch:
            errors.append(f"change[{i}] missing supported")
        if not str(ch.get("provenance", "")).strip():
            errors.append(f"change[{i}] missing provenance")
    return (not errors), errors


def _discrete_identification(fault: dict[str, Any], support_ids: list[str], *, evidence_rows_supplied: bool) -> dict[str, Any]:
    mode = str(fault.get("fault_mode", "unknown")).strip().lower()
    category = str(fault.get("category", "COMPONENT")).upper()
    component = str(fault.get("target_component_id", ""))
    if evidence_rows_supplied and not support_ids:
        return {"fault_id": fault["fault_id"], "component_id": component, "category": category, "status": "ABSTAIN", "identification_type": "DISCRETE_OR_SENSOR_STATE", "identified_change": {}, "basis": "NO_VALID_LOCAL_SUPPORT", "support_evidence": [], "reason": "the retained fault has no VALID matched local evidence available to the identification stage"}
    if mode in GENERIC_MODES:
        return {
            "fault_id": fault["fault_id"],
            "component_id": component,
            "category": category,
            "status": "UNKNOWN",
            "identification_type": "DISCRETE_OR_SENSOR_STATE",
            "identified_change": {"mode": mode, "exact_mode_identified": False},
            "basis": "ONTOLOGY_PLUS_VALIDATED_LOCAL_EVIDENCE",
            "support_evidence": support_ids,
            "reason": "component/state change is supported but the ontology/evidence does not identify a more specific physical mode",
        }
    return {
        "fault_id": fault["fault_id"],
        "component_id": component,
        "category": category,
        "status": "SUPPORTED",
        "identification_type": "SENSOR_BEHAVIOR" if category == "SENSOR" else "DISCRETE_STATE_CHANGE",
        "identified_change": {"mode": mode, "exact_mode_identified": True},
        "basis": "ONTOLOGY_PLUS_VALIDATED_LOCAL_EVIDENCE",
        "support_evidence": support_ids,
        "reason": "fault mode is an explicit ontology hypothesis with positive validated local support",
    }


def _mechanism_identifications(fault: dict[str, Any], report: dict[str, Any] | None) -> list[dict[str, Any]]:
    fid = str(fault["fault_id"]); component = str(fault.get("target_component_id", ""))
    if report is None:
        return [{
            "fault_id": fid, "component_id": component, "category": str(fault.get("category", "PARAMETRIC")).upper(),
            "status": "UNKNOWN", "identification_type": "MECHANISM_CHANGE",
            "identified_change": {}, "basis": "NO_QUALIFIED_ISGN_EVIDENCE",
            "support_evidence": [], "reason": "parametric/mechanism fault requires qualified ISGN identification evidence"
        }]
    ok, errors = validate_isgn_identification_evidence(report)
    if not ok:
        return [{
            "fault_id": fid, "component_id": component, "category": str(fault.get("category", "PARAMETRIC")).upper(),
            "status": "ABSTAIN", "identification_type": "MECHANISM_CHANGE",
            "identified_change": {}, "basis": "INVALID_ISGN_IDENTIFICATION_EVIDENCE",
            "support_evidence": [], "reason": "; ".join(errors)
        }]
    matches = [x for x in report.get("changes", []) if str(x.get("component_id")) == component]
    if not matches:
        return [{
            "fault_id": fid, "component_id": component, "category": str(fault.get("category", "PARAMETRIC")).upper(),
            "status": "UNKNOWN", "identification_type": "MECHANISM_CHANGE",
            "identified_change": {}, "basis": "ISGN_CERTIFIED_DIFFERENTIAL",
            "support_evidence": [], "reason": "ISGN evidence is qualified but contains no identified change for this component"
        }]
    eligible = [x for x in matches if bool(x.get("identifiable")) and bool(x.get("supported"))]
    if not eligible:
        return [{
            "fault_id": fid, "component_id": component, "category": str(fault.get("category", "PARAMETRIC")).upper(),
            "status": "UNKNOWN", "identification_type": "MECHANISM_CHANGE",
            "identified_change": {}, "basis": "ISGN_CERTIFIED_DIFFERENTIAL",
            "support_evidence": [str(x.get("evidence_id", "")) for x in matches if x.get("evidence_id")],
            "reason": "candidate mechanism changes are unidentifiable or do not exceed the ISGN normal empirical envelope in the authorized abnormal window"
        }]
    # Multiple identifiable changes are retained.  If ISGN itself says alternatives,
    # DOR must preserve that uncertainty rather than top-1 selecting a change.
    state = "ALTERNATIVES" if str(report.get("status", "")).upper() == "ALTERNATIVES" or len(eligible) > 1 else "SUPPORTED"
    return [{
        "fault_id": fid, "component_id": component, "category": str(fault.get("category", "PARAMETRIC")).upper(),
        "status": state, "identification_type": "MECHANISM_CHANGE",
        "identified_change": {"alternatives": eligible} if state == "ALTERNATIVES" else eligible[0],
        "basis": "ISGN_CERTIFIED_DIFFERENTIAL",
        "support_evidence": [str(x.get("evidence_id", "")) for x in eligible if x.get("evidence_id")],
        "reason": "identified from a COMPLETE nominal mechanism and VALID authorized abnormal-window evidence"
    }]


def identify_fault_changes(
    ontology: dict[str, Any],
    localization: dict[str, Any],
    evidence: Iterable[dict[str, Any]] | None = None,
    *,
    root_cause: dict[str, Any] | None = None,
    isgn_identification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ov = validate_fault_ontology(ontology)
    if ov.state != "PASS":
        return {"contract": IDENTIFICATION_CONTRACT, "status": "ABSTAIN", "reason": "fault ontology is invalid", "identifications": [], "safety": {"physical_change_claimed": False, "all_requested_changes_identified": False, "root_cause_implied": False, "localization_is_not_identification": True, "residual_is_not_parameter_drift": True, "model_disagreement_is_not_fault_mode": True, "isgn_required_for_parametric_or_structure_claims": True}}
    if str(localization.get("status", "UNKNOWN")).upper() in {"UNKNOWN", "ABSTAIN"}:
        return {"contract": IDENTIFICATION_CONTRACT, "status": "ABSTAIN", "reason": "fault identification requires at least one supported/alternative localization hypothesis", "identifications": [], "safety": {"physical_change_claimed": False, "all_requested_changes_identified": False, "root_cause_implied": False, "localization_is_not_identification": True, "residual_is_not_parameter_drift": True, "model_disagreement_is_not_fault_mode": True, "isgn_required_for_parametric_or_structure_claims": True}}
    idx = ontology_index(ontology)
    fault_ids, source = _candidate_fault_ids(localization, root_cause)
    ev_rows = list(evidence or [])
    valid_ids = {str(x.get("evidence_id")) for x in ev_rows if str(x.get("validity", "VALID")).upper() == "VALID" and x.get("evidence_id")}
    # map localization matched evidence, but only retain VALID evidence IDs when rows were supplied
    loc_support = {str(x.get("fault_id")): [str(e) for e in x.get("matched_evidence", []) if not ev_rows or str(e) in valid_ids] for x in localization.get("fault_hypotheses", [])}

    identifications: list[dict[str, Any]] = []
    unknown_fault_ids: list[str] = []
    for fid in fault_ids:
        fault = idx["faults"].get(fid)
        if fault is None:
            unknown_fault_ids.append(fid)
            continue
        category = str(fault.get("category", "COMPONENT")).upper()
        if category == "PARAMETRIC":
            identifications.extend(_mechanism_identifications(fault, isgn_identification))
        else:
            identifications.append(_discrete_identification(fault, loc_support.get(fid, []), evidence_rows_supplied=bool(ev_rows)))

    if unknown_fault_ids:
        identifications.append({
            "fault_id": None, "component_id": None, "category": "UNKNOWN", "status": "UNKNOWN",
            "identification_type": "UNKNOWN", "identified_change": {}, "basis": "ONTOLOGY_LOOKUP_FAILED",
            "support_evidence": [], "reason": "candidate fault IDs are absent from the validated ontology: " + ",".join(unknown_fault_ids)
        })

    states = {x["status"] for x in identifications}
    if not identifications:
        status = "UNKNOWN"; reason = "no candidate faults were available for identification"
    elif "ABSTAIN" in states:
        status = "ABSTAIN"; reason = "at least one requested physical mechanism identification lacks a valid admissible evidence chain"
    elif "ALTERNATIVES" in states:
        status = "ALTERNATIVES"; reason = "at least one fault has multiple retained identifiable mechanism changes"
    elif "UNKNOWN" in states:
        status = "UNKNOWN"; reason = "fault location/component may be supported but at least one exact state/parameter/structure change remains unidentified"
    else:
        status = "SUPPORTED"; reason = "all retained faults have an explicitly supported state or mechanism-change identification"

    return {
        "contract": IDENTIFICATION_CONTRACT,
        "status": status,
        "reason": reason,
        "candidate_source": source,
        "candidate_fault_ids": fault_ids,
        "identifications": identifications,
        "isgn_admission": {
            "required": any(str(idx["faults"].get(fid, {}).get("category", "")).upper() == "PARAMETRIC" for fid in fault_ids),
            "provided": isgn_identification is not None,
        },
        "safety": {
            "physical_change_claimed": any(x.get("status") in {"SUPPORTED", "ALTERNATIVES"} for x in identifications),
            "all_requested_changes_identified": status == "SUPPORTED",
            "root_cause_implied": False,
            "localization_is_not_identification": True,
            "residual_is_not_parameter_drift": True,
            "model_disagreement_is_not_fault_mode": True,
            "isgn_required_for_parametric_or_structure_claims": True,
        },
    }


def identify_from_files(ontology_path: str | Path, localization_path: str | Path, evidence_path: str | Path, *, root_cause_path: str | Path | None = None, isgn_identification_path: str | Path | None = None) -> dict[str, Any]:
    import csv
    ontology = _load(ontology_path) or {}
    localization = _load(localization_path) or {}
    root_cause = _load(root_cause_path)
    isgn = _load(isgn_identification_path)
    with Path(evidence_path).open(newline="", encoding="utf-8") as f:
        evidence = list(csv.DictReader(f))
    return identify_fault_changes(ontology, localization, evidence, root_cause=root_cause, isgn_identification=isgn)
