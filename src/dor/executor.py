from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .arbiter import arbitrate_anomaly_csv, arbitrate_anomaly_portfolio, arbitrate_class_portfolio
from .control import (
    build_plan,
    formalize_manifest,
    load_json,
    make_effective_m2n_request,
    sha256_file,
    write_json,
)
from .reference import (
    applicability_dataframe,
    build_reference_applicability,
    normalize_context_levels,
    select_hierarchical_fec_output,
)


def _run(cmd: list[str], log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.write_text(p.stdout or "", encoding="utf-8")
    return {"command": cmd, "returncode": int(p.returncode), "log": str(log_path), "stdout_tail": (p.stdout or "")[-4000:]}


def _require_cli(name: str) -> str:
    p = shutil.which(name)
    if not p:
        raise RuntimeError(f"required engine CLI not found on PATH: {name}")
    return p


def _artifact_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }



def _parse_json_set(x: Any) -> set[str]:
    try:
        return set(map(str, json.loads(str(x))))
    except Exception:
        return set()


def _posthoc_test_audit(
    task: str,
    diagnosis_csv: Path,
    *,
    id_column: str | None,
    m2n_test_scores: Path | None,
    normal_label: str | None,
) -> dict[str, Any]:
    """Attach/compute held-out test audit *after* DOR decisions are frozen.

    Test labels are deliberately not an input to model selection, context discovery,
    FEC thresholds, or row-level arbitration. This helper only audits already-written
    diagnosis rows and records that sequencing in the returned provenance.
    """
    d = pd.read_csv(diagnosis_csv)
    needs_labels = "true_label" not in d.columns or d.get("true_label", pd.Series(dtype=object)).isna().any()
    if needs_labels and m2n_test_scores is not None and m2n_test_scores.is_file():
        t = pd.read_csv(m2n_test_scores, dtype={"true_label": str})
        if "true_label" in t.columns and id_column and id_column in d.columns and id_column in t.columns:
            labels = t[[id_column, "true_label"]].copy()
            labels[id_column] = labels[id_column].astype(str)
            d[id_column] = d[id_column].astype(str)
            if labels[id_column].duplicated().any():
                raise ValueError("post-hoc test audit label source has duplicate IDs")
            label_map = labels.set_index(id_column)["true_label"]
            if "true_label" not in d.columns:
                d["true_label"] = d[id_column].map(label_map)
            else:
                d["true_label"] = d["true_label"].where(d["true_label"].notna(), d[id_column].map(label_map))
            d.to_csv(diagnosis_csv, index=False)
    audit: dict[str, Any] = {
        "sequence": "computed only after diagnosis.csv was written; excluded from selection/calibration/arbitration",
        "n_test_rows": int(len(d)),
        "labels_available": bool("true_label" in d.columns and d["true_label"].notna().all()),
    }
    if not audit["labels_available"]:
        return audit
    y = d["true_label"].astype(str)
    if task == "FAULT_CLASSIFICATION" and "diagnosis_set" in d.columns:
        covered = np.array([yy in _parse_json_set(ss) for yy, ss in zip(y, d["diagnosis_set"])], dtype=bool)
        supported = d["diagnosis_status"].isin(["SUPPORTED", "ALTERNATIVES"]).to_numpy()
        audit["empirical_set_coverage_all_rows"] = float(covered.mean()) if len(covered) else None
        if supported.any():
            audit["empirical_set_coverage_supported_rows"] = float(covered[supported].mean())
            if "diagnosis_set_size" in d.columns:
                audit["mean_prediction_set_size_supported_rows"] = float(d.loc[supported, "diagnosis_set_size"].mean())
            singleton = supported & (d.get("diagnosis_set_size", pd.Series(np.zeros(len(d), dtype=int))).to_numpy() == 1)
            if singleton.any():
                audit["singleton_supported_accuracy"] = float(covered[singleton].mean())
    elif task == "ANOMALY_DETECTION" and normal_label is not None and "diagnosis" in d.columns:
        supported = (d["diagnosis_status"] == "SUPPORTED").to_numpy()
        if supported.any():
            truth = np.where(y.to_numpy() == str(normal_label), "NORMAL", "ABNORMAL")
            pred = d["diagnosis"].astype(str).to_numpy()
            ts = truth[supported]; ps = pred[supported]
            audit["supported_rows"] = int(supported.sum())
            audit["empirical_supported_accuracy"] = float((ts == ps).mean())
            tp = int(((ts == "ABNORMAL") & (ps == "ABNORMAL")).sum())
            fp = int(((ts == "NORMAL") & (ps == "ABNORMAL")).sum())
            fn = int(((ts == "ABNORMAL") & (ps == "NORMAL")).sum())
            tn = int(((ts == "NORMAL") & (ps == "NORMAL")).sum())
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            tnr = tn / (tn + fp) if tn + fp else 0.0
            audit["abnormal_f1_supported_rows"] = float(f1)
            audit["balanced_accuracy_supported_rows"] = float((rec + tnr) / 2)
            audit["confusion_supported_rows"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return audit

def execute_chain(
    manifest_path: str | Path,
    request_path: str | Path,
    data_path: str | Path,
    output_dir: str | Path,
    *,
    dry_run: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_manifest = load_json(manifest_path)
    raw_request = load_json(request_path)
    manifest = formalize_manifest(raw_manifest, raw_request, data_path)
    effective_request = make_effective_m2n_request(raw_request, manifest)
    id_column = (manifest.get("identifiers") or [None])[0]
    applicability = build_reference_applicability(manifest, data_path, id_column=id_column)

    manifest_out = write_json(out / "dataset_manifest.json", manifest)
    request_out = write_json(out / "m2n_request_effective.json", effective_request)
    plan = build_plan(manifest, effective_request, data_path, out)
    plan.provenance["manifest_sha256"] = sha256_file(manifest_out)  # mapping intentionally mutable at runtime
    plan.provenance["request_sha256"] = sha256_file(request_out)
    write_json(out / "task_qualification.json", plan.qualification.to_dict())
    write_json(out / "engine_admissions.json", [x.to_dict() for x in plan.engine_admissions])
    write_json(out / "diagnostic_plan.json", plan.to_dict())
    write_json(out / "reference_applicability.json", applicability)
    applicability_dataframe(applicability).to_csv(out / "reference_applicability.csv", index=False)

    report: dict[str, Any] = {
        "contract": "DOR-EXECUTION-REPORT/0.4",
        "plan_id": plan.plan_id,
        "task": plan.task,
        "dry_run": dry_run,
        "resume": resume,
        "steps": [],
        "status": "PLANNED" if dry_run else "RUNNING",
    }
    if dry_run or plan.qualification.status == "UNSUPPORTED":
        report["status"] = "PLANNED" if dry_run else "BLOCKED"
        write_json(out / "execution_report.json", report)
        return report

    m2n_cli = _require_cli("m2n")
    fec_cli = _require_cli("fec")
    m2n_out = out / "m2n"
    bundle_path = m2n_out / "diagnostic_model_bundle.json"
    reused_m2n = False
    if resume and bundle_path.exists():
        prior = load_json(bundle_path)
        req_sha = sha256_file(request_out)
        data_sha = sha256_file(data_path)
        bindings_ok = prior.get("request_sha256") == req_sha and prior.get("data_sha256") == data_sha
        artifacts_ok = True
        artifact_errors = []
        if bindings_ok:
            for model in prior.get("models") or []:
                mp = m2n_out / str(model.get("artifact", ""))
                if not mp.is_file():
                    artifacts_ok = False; artifact_errors.append(f"missing model artifact: {mp}"); continue
                expected_model_sha = model.get("sha256")
                if expected_model_sha and sha256_file(mp) != expected_model_sha:
                    artifacts_ok = False; artifact_errors.append(f"model SHA mismatch: {mp}")
            for cid, art in (prior.get("score_artifacts") or {}).items():
                for role in ("calibration", "test", "contract"):
                    rel = art.get(role)
                    if rel and not (m2n_out / rel).is_file():
                        artifacts_ok = False; artifact_errors.append(f"missing score artifact {cid}/{role}: {rel}")
        if bindings_ok and artifacts_ok:
            reused_m2n = True
            report["steps"].append({"step_id":"S20","action":"reuse SHA-bound M2N DiagnosticModelBundle","returncode":0,"reused":True,"bundle":str(bundle_path)})
        else:
            why = []
            if not bindings_ok: why.append("request/data SHA-256 mismatch")
            why.extend(artifact_errors)
            report["steps"].append({"step_id":"S20","action":"resume rejected stale/incomplete M2N bundle","returncode":0,"reused":False,"reason":"; ".join(why)})
    if not reused_m2n:
        rec = _run([m2n_cli, "diagnostic-design", "--request", str(request_out), "--data", str(Path(data_path)), "--out", str(m2n_out)], out / "logs" / "S20_m2n.log")
        rec["step_id"] = "S20"; report["steps"].append(rec)
        if rec["returncode"] != 0:
            report["status"] = "FAILED"; write_json(out / "execution_report.json", report); raise RuntimeError("M2N diagnostic-design failed; see execution report/log")

    bundle = load_json(bundle_path)
    retained = list(bundle.get("retained_candidate_ids") or [])
    if not retained:
        report["status"] = "BLOCKED"; report["block_reason"] = "M2N retained no candidate for calibration"
        write_json(out / "execution_report.json", report); return report

    levels = normalize_context_levels(manifest)
    alpha = float((manifest.get("calibration") or {}).get("alpha", 0.10))
    fec_outputs: list[Path] = []
    fec_refs: list[Path] = []
    fec_level_outputs: list[Path] = []
    score_artifacts = bundle.get("score_artifacts") or {}
    selected_levels = set((applicability.get("summary") or {}).get("selected_level_counts", {}).keys())

    for idx, cid in enumerate(retained):
        art = score_artifacts.get(cid) or {}
        if not all(k in art for k in ("calibration", "test", "contract")):
            report["status"] = "BLOCKED"; report["block_reason"] = f"retained candidate {cid} lacks calibration/test/contract score artifacts"
            write_json(out / "execution_report.json", report); return report
        cal_scores = m2n_out / art["calibration"]
        test_scores = m2n_out / art["test"]
        score_contract = m2n_out / art["contract"]
        cdir = out / "fec" / cid.replace("/", "_")
        cdir.mkdir(parents=True, exist_ok=True)
        level_outputs: dict[str, Path] = {}

        for lidx, level in enumerate(levels):
            if level.level_id not in selected_levels:
                report["steps"].append({
                    "step_id": "S30", "candidate_id": cid, "reference_level": level.level_id,
                    "action": "skip FEC level: no test row admitted to this declared reference level",
                    "returncode": 0, "skipped": True,
                })
                continue
            ldir = cdir / level.level_id
            ref = ldir / "reference.json"
            pred = ldir / "test_calibrated.csv"
            ldir.mkdir(parents=True, exist_ok=True)
            context_cols = list(level.columns)
            context_arg = ",".join(context_cols)
            provenance = (
                f"DOR {plan.plan_id} candidate={cid} reference_level={level.level_id} "
                f"sharing_basis={level.sharing_basis}"
            )

            if plan.task == "ANOMALY_DETECTION":
                cmd = [
                    fec_cli, "anomaly-calibrate", "--calibration-scores", str(cal_scores),
                    "--reference", str(ref), "--score-contract", str(score_contract),
                    "--alpha", str(alpha), "--min-group-size", str(level.min_group_size),
                ]
                if context_cols:
                    cmd += ["--context-columns", context_arg]
                cmd += ["--provenance", provenance]
                infer_cmd = [fec_cli, "anomaly-infer", "--scores", str(test_scores), "--reference", str(ref), "--output", str(pred)]
            elif plan.task == "FAULT_CLASSIFICATION":
                if context_cols:
                    cmd = [
                        fec_cli, "class-context-calibrate", "--calibration-scores", str(cal_scores),
                        "--reference", str(ref), "--score-contract", str(score_contract),
                        "--context-columns", context_arg, "--alpha", str(alpha),
                        "--min-group-size", str(level.min_group_size), "--provenance", provenance,
                    ]
                    infer_cmd = [fec_cli, "class-context-infer", "--scores", str(test_scores), "--reference", str(ref), "--output", str(pred)]
                else:
                    cmd = [
                        fec_cli, "class-calibrate", "--calibration-scores", str(cal_scores),
                        "--reference", str(ref), "--score-contract", str(score_contract),
                        "--alpha", str(alpha), "--min-group-size", str(level.min_group_size),
                        "--provenance", provenance,
                    ]
                    infer_cmd = [fec_cli, "class-infer", "--scores", str(test_scores), "--reference", str(ref), "--output", str(pred)]
            else:
                raise RuntimeError(f"unsupported runtime task: {plan.task}")

            r1 = _run(cmd, out / "logs" / f"S30_fec_calibrate_{idx:02d}_{lidx:02d}.log")
            r1.update({"step_id": "S30", "candidate_id": cid, "reference_level": level.level_id})
            report["steps"].append(r1)
            if r1["returncode"] != 0:
                report["status"] = "FAILED"; write_json(out / "execution_report.json", report)
                raise RuntimeError(f"FEC calibration failed for {cid} level={level.level_id}")
            r2 = _run(infer_cmd, out / "logs" / f"S40_fec_infer_{idx:02d}_{lidx:02d}.log")
            r2.update({"step_id": "S40", "candidate_id": cid, "reference_level": level.level_id})
            report["steps"].append(r2)
            if r2["returncode"] != 0:
                report["status"] = "FAILED"; write_json(out / "execution_report.json", report)
                raise RuntimeError(f"FEC inference failed for {cid} level={level.level_id}")
            fec_refs.append(ref); fec_level_outputs.append(pred); level_outputs[level.level_id] = pred

        if not level_outputs:
            report["status"] = "BLOCKED"
            report["block_reason"] = f"no declared reference level was runnable for retained candidate {cid}"
            write_json(out / "execution_report.json", report); return report
        selected_pred = cdir / "test_selected_reference.csv"
        select_hierarchical_fec_output(level_outputs, applicability, selected_pred, id_column=id_column)
        report["steps"].append({
            "step_id": "S45", "candidate_id": cid,
            "action": "DOR select finest admitted declared reference level per test row",
            "returncode": 0, "output": str(selected_pred),
        })
        fec_outputs.append(selected_pred)

    diagnosis_csv = out / "diagnosis.csv"
    diagnosis_summary = out / "diagnosis_summary.json"
    if plan.task == "ANOMALY_DETECTION":
        if len(fec_outputs) == 1:
            arbitrate_anomaly_csv(fec_outputs[0], diagnosis_csv, diagnosis_summary)
        else:
            arbitrate_anomaly_portfolio(fec_outputs, diagnosis_csv, diagnosis_summary)
    else:
        label_col = str((manifest.get("supervision") or {}).get("label_column") or "true_label")
        # M2N score packs normalize label name to true_label, so DOR consumes that contract.
        arbitrate_class_portfolio(fec_outputs, diagnosis_csv, diagnosis_summary, id_column=(manifest.get("identifiers") or ["sample_id"])[0], true_label_column="true_label")
    report["steps"].append({"step_id": "S50", "action": "DOR arbitration", "returncode": 0, "output": str(diagnosis_csv)})

    # Test labels, if present, are joined/audited only *after* the diagnosis decisions exist.
    first_test_scores = None
    if retained:
        first_art = score_artifacts.get(retained[0]) or {}
        if first_art.get("test"):
            first_test_scores = m2n_out / first_art["test"]
    audit = _posthoc_test_audit(
        plan.task, diagnosis_csv,
        id_column=(manifest.get("identifiers") or [None])[0],
        m2n_test_scores=first_test_scores,
        normal_label=(manifest.get("supervision") or {}).get("normal_label"),
    )

    # DOR v0.4 final bundle binds hierarchy/applicability, model, FEC level references, and decisions.
    summary = load_json(diagnosis_summary)
    summary["posthoc_test_audit"] = audit
    write_json(diagnosis_summary, summary)
    diag_status = "SUPPORTED"
    if summary.get("abstain_fraction", 0) >= 1.0:
        diag_status = "ABSTAIN"
    elif summary.get("alternatives_fraction", 0) > 0:
        diag_status = "ALTERNATIVES"
    elif summary.get("supported_fraction", 0) <= 0:
        diag_status = "UNKNOWN"
    artifacts = [manifest_out, request_out, out / "task_qualification.json", out / "engine_admissions.json", out / "diagnostic_plan.json", out / "reference_applicability.json", out / "reference_applicability.csv", bundle_path, *fec_refs, *fec_level_outputs, *fec_outputs, diagnosis_csv, diagnosis_summary]
    diagnosis_bundle = {
        "contract": "DOR-DIAGNOSIS-BUNDLE/0.4",
        "plan_id": plan.plan_id,
        "task": plan.task,
        "task_qualification": plan.qualification.to_dict(),
        "engine_admissions": [x.to_dict() for x in plan.engine_admissions],
        "model_selection": bundle.get("selection"),
        "retained_candidate_ids": retained,
        "calibration": {
            "policy": (manifest.get("context") or {}).get("policy", "EXACT"),
            "context_levels": [x.to_dict() for x in levels],
            "alpha": alpha,
            "missing_context_policy": (manifest.get("calibration") or {}).get("missing_context_policy", "ABSTAIN"),
            "reference_applicability": "reference_applicability.json",
            "references": [str(x.relative_to(out)) for x in fec_refs],
            "selected_reference_outputs": [str(x.relative_to(out)) for x in fec_outputs],
        },
        "diagnosis_status": diag_status,
        "diagnosis_summary": summary,
        "test_audit": audit,
        "artifacts": [_artifact_record(p, out) for p in artifacts if p.exists()],
        "prohibited_claims": [
            "test labels do not alter M2N selection, context keys, calibration thresholds, or DOR rules",
            "scalar anomaly evidence is not a fault-class or root-cause probability",
            "class prediction sets are not physical root-cause probabilities",
            "ISGN skip/defer is a valid qualified path and must not be replaced by fabricated mechanism discovery",
            "DOR hierarchy admission is not a new conformal theorem; any coarser reference claim remains conditional on exchangeability within the declared ancestor group",
            "held-out test labels/scores do not discover or approve context pooling",
        ],
    }
    write_json(out / "diagnosis_bundle.json", diagnosis_bundle)
    report["steps"].append({"step_id": "S60", "action": "bundle/provenance", "returncode": 0, "output": str(out / 'diagnosis_bundle.json')})
    report["status"] = "COMPLETED"
    report["artifacts"] = [_artifact_record(p, out) for p in [out / "diagnostic_plan.json", bundle_path, diagnosis_csv, diagnosis_summary, out / "diagnosis_bundle.json"]]
    write_json(out / "execution_report.json", report)
    return report
