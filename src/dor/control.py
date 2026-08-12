from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .contracts import (
    DiagnosticPlan,
    EngineAdmission,
    PlanStep,
    SUPPORTED_RUNTIME_TASKS,
    TaskQualification,
)
from .reference import build_reference_applicability, normalize_context_levels

MANIFEST_CONTRACT = "DOR-DATASET-MANIFEST/0.4"


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def _as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _role_values(request: dict[str, Any], role: str) -> list[str]:
    data = request.get("data") or {}
    key = {
        "train": "train_values",
        "validation": "validation_values",
        "calibration": "calibration_values",
        "test": "test_values",
    }[role]
    return [str(x) for x in _as_list(data.get(key))]


def formalize_manifest(
    manifest: dict[str, Any],
    request: dict[str, Any],
    data_path: str | Path,
) -> dict[str, Any]:
    """Normalize older manifests into the DOR v0.4 dataset/context contract.

    DOR owns the canonical task/data/context/reference-applicability contract.
    M2N's request remains a model-design request and is embedded by hash, not
    treated as dataset truth.  v0.4 never invents a coarser context hierarchy.
    """
    data_path = Path(data_path)
    if manifest.get("contract") == MANIFEST_CONTRACT:
        out = json.loads(json.dumps(manifest))
    elif manifest.get("contract") == "DOR-DATASET-MANIFEST/0.3":
        out = json.loads(json.dumps(manifest))
        out["contract"] = MANIFEST_CONTRACT
        out.setdefault("provenance", {}).setdefault("upgraded_from_contract", "DOR-DATASET-MANIFEST/0.3")
    else:
        task = str(request.get("task") or manifest.get("task") or "").upper()
        req_data = request.get("data") or {}
        supervision = request.get("supervision") or {}
        context_cols = _as_list(manifest.get("context_columns"))
        if not context_cols:
            # Prototype-specific explicit hints only. We do not discover contexts from test labels.
            if "channel_role_plan" in manifest:
                context_cols = ["channel"]
            elif str(manifest.get("target", "")).lower() == "accumulator":
                context_cols = ["cooler", "pump"]
        group_cols = _as_list(manifest.get("nuisance_group_columns"))
        if "split_group" in _as_list(req_data.get("id_columns")):
            group_cols = ["split_group"]
        normal_label = supervision.get("normal_label", manifest.get("normal_class"))
        out = {
            "contract": MANIFEST_CONTRACT,
            "dataset_id": manifest.get("dataset_id") or manifest.get("dataset") or data_path.stem,
            "domain": manifest.get("domain") or ("satellite" if "OPSSAT" in str(manifest.get("dataset", "")).upper() else "industrial"),
            "task": task,
            "data": {
                "path": str(data_path),
                "sha256": manifest.get("derived_sha256") or manifest.get("derived_csv_sha256") or sha256_file(data_path),
                "row_count": manifest.get("n_rows"),
            },
            "identifiers": _as_list(req_data.get("id_columns")),
            "split": {
                "column": req_data.get("split_column", "split"),
                "train_values": _role_values(request, "train"),
                "validation_values": _role_values(request, "validation"),
                "calibration_values": _role_values(request, "calibration"),
                "test_values": _role_values(request, "test"),
            },
            "supervision": {
                "label_column": supervision.get("label_column") or manifest.get("target_column"),
                "normal_label": None if normal_label is None else str(normal_label),
                "class_frame": [str(x) for x in _as_list(manifest.get("class_frame"))],
            },
            "context": {
                "columns": [str(x) for x in context_cols],
                "policy": "EXACT",
            },
            "calibration": {
                "alpha": float((manifest.get("calibration") or {}).get("alpha", 0.10)),
                "min_group_size": int((manifest.get("calibration") or {}).get("min_group_size", 20)),
                "missing_context_policy": "ABSTAIN",
            },
            "leakage_guard": {
                "group_columns": [str(x) for x in group_cols],
                "description": manifest.get("leakage_guard", "data-role groups must not cross train/validation/calibration/test"),
            },
            "mechanism": {
                "status": "DEFERRED" if "DEFERRED" in str(manifest.get("isgn_route", "")).upper() else "NOT_PROVIDED",
                "reason": manifest.get("isgn_route", "No certified mechanism artifact declared in the dataset manifest."),
            },
            "provenance": {
                "source_contract": manifest.get("contract"),
                "source_checksums": manifest.get("source_md5") or {},
                "notes": [x for x in [manifest.get("leakage_guard"), manifest.get("split_policy")] if x],
            },
        }
    # DOR-authoritative calibration/reference policy must always be explicit in v0.4.
    out.setdefault("context", {}).setdefault("columns", [])
    out["context"].setdefault("policy", "EXACT")
    # Materialize a canonical hierarchy. Existing EXACT contracts remain one level.
    levels = normalize_context_levels(out)
    out["context"]["levels"] = [x.to_dict() for x in levels]
    out["context"]["columns"] = list(levels[0].columns)
    out["context"].setdefault("fallback_policy", "DECLARED_ANCESTOR_ONLY")
    out.setdefault("calibration", {}).setdefault("alpha", 0.10)
    out["calibration"].setdefault("min_group_size", 20)
    out["calibration"].setdefault("missing_context_policy", "ABSTAIN")
    out.setdefault("mechanism", {}).setdefault("status", "NOT_PROVIDED")
    out.setdefault("leakage_guard", {}).setdefault("group_columns", [])
    out.setdefault("provenance", {})
    out["data"]["path"] = str(data_path)
    out["data"]["sha256"] = out["data"].get("sha256") or sha256_file(data_path)
    return out


def _mask_from_values(s: pd.Series, values: Iterable[str]) -> pd.Series:
    vals = {str(x) for x in values}
    return s.astype(str).isin(vals)


def qualify_task(
    manifest: dict[str, Any],
    request: dict[str, Any],
    data_path: str | Path,
) -> TaskQualification:
    p = Path(data_path)
    reasons: list[str] = []
    warnings: list[str] = []
    next_actions: list[str] = []
    checks: dict[str, Any] = {}
    task = str(manifest.get("task") or request.get("task") or "").upper()

    if task not in SUPPORTED_RUNTIME_TASKS:
        reasons.append(f"DOR v0.4 runtime does not execute task {task!r}; supported tasks are {sorted(SUPPORTED_RUNTIME_TASKS)}")
        return TaskQualification(task, "UNSUPPORTED", tuple(reasons), tuple(warnings), tuple(next_actions), checks)
    if str(request.get("task", "")).upper() != task:
        reasons.append("DatasetManifest.task and M2N diagnostic request task disagree.")
    if not p.exists():
        reasons.append(f"data file not found: {p}")
        return TaskQualification(task, "UNSUPPORTED", tuple(reasons), tuple(warnings), tuple(next_actions), checks)

    expected = str((manifest.get("data") or {}).get("sha256") or "")
    got = sha256_file(p)
    checks["data_sha256"] = {"expected": expected or None, "actual": got, "match": (not expected or expected == got)}
    if expected and expected != got:
        reasons.append("data SHA-256 does not match DatasetManifest; DOR refuses to plan on an unbound data artifact.")

    df = pd.read_csv(p)
    checks["row_count"] = int(len(df))
    declared_rows = (manifest.get("data") or {}).get("row_count")
    if declared_rows is not None and int(declared_rows) != len(df):
        warnings.append(f"declared row_count={declared_rows} differs from observed {len(df)}")

    split = manifest.get("split") or {}
    split_col = split.get("column", "split")
    label_col = (manifest.get("supervision") or {}).get("label_column")
    required_cols = {str(split_col)}
    required_cols.update(map(str, manifest.get("identifiers") or []))
    for level in normalize_context_levels(manifest):
        required_cols.update(map(str, level.columns))
    if label_col:
        required_cols.add(str(label_col))
    missing = sorted(required_cols - set(df.columns))
    checks["required_columns"] = {"required": sorted(required_cols), "missing": missing}
    if missing:
        reasons.append(f"data is missing required DOR contract columns: {missing}")
        return TaskQualification(task, "UNSUPPORTED", tuple(reasons), tuple(warnings), tuple(next_actions), checks)

    role_masks = {
        "train": _mask_from_values(df[split_col], split.get("train_values") or []),
        "validation": _mask_from_values(df[split_col], split.get("validation_values") or []),
        "calibration": _mask_from_values(df[split_col], split.get("calibration_values") or []),
        "test": _mask_from_values(df[split_col], split.get("test_values") or []),
    }
    role_counts = {k: int(v.sum()) for k, v in role_masks.items()}
    checks["role_counts"] = role_counts
    for role in ("train", "validation", "calibration", "test"):
        if role_counts[role] == 0:
            reasons.append(f"required data role {role} is empty")

    covered = pd.Series(False, index=df.index)
    for mask in role_masks.values():
        covered |= mask
    checks["unassigned_rows"] = int((~covered).sum())
    if (~covered).any():
        warnings.append(f"{int((~covered).sum())} rows are outside the declared four data roles and will not participate in the planned chain")

    # Leakage guard: declared groups may not cross roles.
    group_cols = [str(x) for x in (manifest.get("leakage_guard") or {}).get("group_columns") or []]
    if group_cols:
        assigned = df.loc[covered, group_cols + [split_col]].copy()
        assigned["__role"] = None
        for role, mask in role_masks.items():
            assigned.loc[mask[covered].values, "__role"] = role
        if len(group_cols) == 1:
            keys = assigned[group_cols[0]].astype(str)
        else:
            keys = assigned[group_cols].astype(str).agg("|".join, axis=1)
        nroles = pd.DataFrame({"g": keys, "r": assigned["__role"]}).groupby("g")["r"].nunique()
        bad = nroles[nroles > 1]
        checks["leakage_guard"] = {"group_columns": group_cols, "cross_role_group_count": int(len(bad))}
        if len(bad):
            reasons.append(f"{len(bad)} declared leakage groups cross data roles")
    else:
        checks["leakage_guard"] = {"group_columns": [], "cross_role_group_count": None}
        warnings.append("no leakage_guard.group_columns declared; DOR cannot prove trajectory/group separation")

    supervision = manifest.get("supervision") or {}
    normal_label = supervision.get("normal_label")
    if task == "ANOMALY_DETECTION":
        if label_col is None or normal_label is None:
            reasons.append("ANOMALY_DETECTION requires supervision.label_column and supervision.normal_label for qualification/audit")
        else:
            y = df[label_col].astype(str)
            normal = str(normal_label)
            train_non_normal = int((y[role_masks["train"]] != normal).sum())
            cal_non_normal = int((y[role_masks["calibration"]] != normal).sum())
            val_anom = int((y[role_masks["validation"]] != normal).sum())
            checks["anomaly_supervision"] = {
                "train_non_normal": train_non_normal,
                "calibration_non_normal": cal_non_normal,
                "validation_anomaly_rows": val_anom,
            }
            if train_non_normal:
                reasons.append("normal-only anomaly training split contains labelled anomaly rows")
            if cal_non_normal:
                reasons.append("trusted-normal anomaly calibration split contains labelled anomaly rows")
            if val_anom == 0:
                warnings.append("validation contains no labelled anomaly; M2N may be unable to rank anomaly detectors")
    elif task == "FAULT_CLASSIFICATION":
        if label_col is None:
            reasons.append("FAULT_CLASSIFICATION requires supervision.label_column")
        else:
            y = df[label_col].astype(str)
            train_classes = sorted(y[role_masks["train"]].unique().tolist())
            val_classes = sorted(y[role_masks["validation"]].unique().tolist())
            checks["classification_supervision"] = {"train_classes": train_classes, "validation_classes": val_classes}
            if len(train_classes) < 2:
                reasons.append("classification training split has fewer than two classes")
            missing_val_classes = sorted(set(train_classes) - set(val_classes))
            if missing_val_classes:
                warnings.append(f"validation lacks training classes {missing_val_classes}; model-family adjudication may be underpowered")

    # DOR v0.4 reference applicability: finest declared context first, then only
    # explicitly admitted ancestors. Test labels/scores are excluded by construction.
    try:
        app = build_reference_applicability(manifest, p, id_column=(manifest.get("identifiers") or [None])[0])
        checks["reference_applicability"] = app["summary"]
        checks["context_levels"] = app["levels"]
        unsupported_n = int(app["summary"]["unsupported_rows"])
        hierarchy_n = int(app["summary"]["status_counts"].get("HIERARCHY_SUPPORTED", 0))
        power_n = int(app["summary"]["status_counts"].get("POWER_LIMITED", 0))
        context_unsup_n = int(app["summary"]["status_counts"].get("CONTEXT_UNSUPPORTED", 0))
        if hierarchy_n:
            warnings.append(f"{hierarchy_n} test rows require an explicitly declared coarser reference level; any validity claim is conditional on exchangeability within that ancestor group")
        if unsupported_n:
            warnings.append(f"{unsupported_n} test rows have no admitted calibration reference: POWER_LIMITED={power_n}, CONTEXT_UNSUPPORTED={context_unsup_n}; DOR must fail closed there")
            next_actions.append("collect independent calibration reference or revise the declared context hierarchy using domain justification; do not discover pooling from held-out test outcomes")
    except ValueError as e:
        reasons.append(f"invalid DOR v0.4 context hierarchy/reference policy: {e}")

    status = "UNSUPPORTED" if reasons else ("CONDITIONALLY_QUALIFIED" if warnings else "QUALIFIED")
    return TaskQualification(task, status, tuple(reasons), tuple(warnings), tuple(next_actions), checks)


def make_effective_m2n_request(request: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """Ensure DOR-declared context columns survive M2N score-pack export as IDs.

    This is orchestration metadata only; it does not let DOR alter candidate families,
    metrics, or labels after validation/test inspection.
    """
    out = json.loads(json.dumps(request))
    data = out.setdefault("data", {})
    ids = [str(x) for x in _as_list(data.get("id_columns"))]
    for level in normalize_context_levels(manifest):
        for c in level.columns:
            if str(c) not in ids:
                ids.append(str(c))
    data["id_columns"] = ids
    return out


def engine_admissions(manifest: dict[str, Any], q: TaskQualification) -> tuple[EngineAdmission, ...]:
    mech = manifest.get("mechanism") or {}
    mech_status = str(mech.get("status", "NOT_PROVIDED")).upper()
    if mech_status in {"NOT_PROVIDED", "DEFERRED", "UNCERTIFIED", "UNIDENTIFIABLE"}:
        isgn = EngineAdmission(
            "ISGN", "SKIP",
            (f"mechanism route is {mech_status}; full DOR chain does not require fabricated mechanism discovery", str(mech.get("reason", ""))),
            ("DatasetManifest",), ("explicit skip/admission provenance",),
        )
    elif mech_status in {"CERTIFIED", "COMPLETE"}:
        isgn = EngineAdmission(
            "ISGN", "SKIP",
            ("a certified mechanism artifact is already declared; consume it rather than rediscovering mechanism during diagnosis orchestration",),
            ("CertifiedMechanismBundle",), ("mechanism feature binding",),
        )
    else:
        isgn = EngineAdmission(
            "ISGN", "BLOCKED",
            ("mechanism discovery was requested but DOR v0.4 has no executable DiscoveryRequest bound in the manifest",),
            ("DatasetManifest",), ("CertifiedMechanismBundle",),
        )

    if q.status == "UNSUPPORTED":
        m2n = EngineAdmission("M2N", "BLOCKED", ("task qualification failed",), ("DatasetManifest",), ("DiagnosticModelBundle",))
        fec = EngineAdmission("FEC", "BLOCKED", ("upstream model/score pack unavailable",), ("DiagnosticModelBundle",), ("CalibratedEvidenceBundle",))
        dor = EngineAdmission("DOR", "BLOCKED", ("calibrated evidence unavailable",), ("CalibratedEvidenceBundle",), ("DiagnosisBundle",))
    elif isgn.status == "BLOCKED":
        m2n = EngineAdmission("M2N", "BLOCKED", ("the manifest requested an unresolved ISGN mechanism route; DOR will not silently bypass a declared mechanism dependency",), ("CertifiedMechanismBundle", "DatasetManifest"), ("DiagnosticModelBundle",))
        fec = EngineAdmission("FEC", "BLOCKED", ("upstream model/score pack unavailable because the declared mechanism route is unresolved",), ("DiagnosticModelBundle",), ("CalibratedEvidenceBundle",))
        dor = EngineAdmission("DOR", "BLOCKED", ("calibrated evidence unavailable",), ("CalibratedEvidenceBundle",), ("DiagnosisBundle",))
    else:
        m2n = EngineAdmission("M2N", "RUN", ("task/data roles satisfy DOR admission; M2N may qualify finite model families using train/validation only",), ("DatasetManifest", "M2N diagnostic request", "data"), ("DiagnosticModelBundle", "score packs"))
        fec = EngineAdmission("FEC", "RUN", ("independent calibration role exists; DOR v0.4 may use only exact or explicitly declared ancestor context levels and remains fail-closed otherwise",), ("retained M2N score packs",), ("context-conditioned calibrated evidence",))
        dor = EngineAdmission("DOR", "RUN", ("DOR will arbitrate only FEC-supported evidence and preserve alternatives/abstention",), ("FEC outputs",), ("DiagnosisBundle",))
    return isgn, m2n, fec, dor


def build_plan(
    manifest: dict[str, Any],
    request: dict[str, Any],
    data_path: str | Path,
    output_dir: str | Path,
) -> DiagnosticPlan:
    q = qualify_task(manifest, request, data_path)
    admissions = engine_admissions(manifest, q)
    out = Path(output_dir)
    levels = normalize_context_levels(manifest)
    context_cols = list(levels[0].columns)
    cal = manifest.get("calibration") or {}
    task = q.task
    steps = [
        PlanStep("S00", "DOR", "TASK_QUALIFICATION", "COMPLETED", (), ("DatasetManifest", "M2N request", "data"), ("task_qualification.json",)),
        PlanStep("S05", "DOR", "REFERENCE_APPLICABILITY", "COMPLETED" if q.status != "UNSUPPORTED" else "BLOCKED", ("S00",), ("DatasetManifest", "data-role context values"), ("reference_applicability.json", "reference_applicability.csv"), parameters={"levels":[x.to_dict() for x in levels],"fallback_policy":(manifest.get("context") or {}).get("fallback_policy","DECLARED_ANCESTOR_ONLY")}),
        PlanStep("S10", "ISGN", "MECHANISM_ADMISSION", "SKIPPED" if admissions[0].status == "SKIP" else ("BLOCKED" if admissions[0].status == "BLOCKED" else "PLANNED"), ("S00",), ("DatasetManifest",), ("mechanism_admission.json",)),
        PlanStep("S20", "M2N", "DIAGNOSTIC_MODEL_DESIGN", "BLOCKED" if admissions[1].status == "BLOCKED" else "PLANNED", ("S00", "S10"), ("m2n_request_effective.json", str(data_path)), ("m2n/diagnostic_model_bundle.json",), command_template=("m2n", "diagnostic-design", "--request", "${m2n_request}", "--data", "${data}", "--out", "${m2n_out}")),
        PlanStep("S30", "FEC", "CALIBRATE_RETAINED_SCORE_PACKS_BY_LEVEL", "BLOCKED" if admissions[2].status == "BLOCKED" else "PLANNED", ("S20", "S05"), ("m2n retained calibration score packs", "declared context levels"), ("fec/*/*/reference.json",), parameters={"levels":[x.to_dict() for x in levels],"alpha":float(cal.get("alpha",0.10)),"missing_context_policy":cal.get("missing_context_policy","ABSTAIN")}),
        PlanStep("S40", "FEC", "INFER_RETAINED_SCORE_PACKS_BY_LEVEL", "BLOCKED" if admissions[2].status == "BLOCKED" else "PLANNED", ("S30",), ("m2n retained test score packs", "FEC level references"), ("fec/*/*/test_calibrated.csv",)),
        PlanStep("S45", "DOR", "SELECT_DECLARED_REFERENCE_LEVEL", "BLOCKED" if admissions[3].status == "BLOCKED" else "PLANNED", ("S40", "S05"), ("FEC per-level outputs", "reference_applicability.json"), ("fec/*/test_selected_reference.csv",)),
        PlanStep("S50", "DOR", "DIAGNOSTIC_ARBITRATION", "BLOCKED" if admissions[3].status == "BLOCKED" else "PLANNED", ("S45",), ("FEC outputs after DOR reference selection",), ("diagnosis.csv", "diagnosis_summary.json")),
        PlanStep("S60", "DOR", "BUNDLE_AND_PROVENANCE", "BLOCKED" if admissions[3].status == "BLOCKED" else "PLANNED", ("S50",), ("all prior artifacts",), ("diagnosis_bundle.json", "execution_report.json")),
    ]
    prov = {
        "manifest_contract": manifest.get("contract"),
        "manifest_sha256": None,
        "request_sha256": None,
        "data_sha256": sha256_file(data_path),
        "output_dir": str(out),
        "test_data_excluded_from_model_selection": True,
        "context_discovery_from_test_labels": False,
        "context_discovery_from_test_scores": False,
        "implicit_context_pooling": False,
    }
    plan_id = f"dor04-{uuid.uuid5(uuid.NAMESPACE_URL, prov['data_sha256'] + '|' + task).hex[:16]}"
    return DiagnosticPlan(plan_id, task, q, admissions, tuple(steps), prov)
