from __future__ import annotations

"""Operational alert-burden evidence for DOR anomaly deployments.

Selected benchmark segments are useful for model quality but are not a continuous
operations exposure denominator.  This evaluator explicitly distinguishes those
cases so 'false positives per selected-segment hour' cannot be misreported as
fleet false alarms/hour.
"""

from pathlib import Path
from typing import Any
import json
import math

import pandas as pd

CONTRACT = "DOR-DEPLOYMENT-OPS-EVIDENCE/1.0"
RESULT_CONTRACT = "DOR-DEPLOYMENT-OPS-EVALUATION/1.0"
VALID_COVERAGE = {"CONTIGUOUS_OPERATION", "SELECTED_SEGMENTS"}


def _load_json(value: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return json.loads(Path(value).read_text(encoding="utf-8"))


def evaluate_deployment_operations(spec: dict[str, Any], predictions: pd.DataFrame) -> dict[str, Any]:
    errors: list[str] = []
    if spec.get("contract") != CONTRACT:
        errors.append(f"contract must be {CONTRACT}")
    coverage_mode = str(spec.get("coverage_mode", ""))
    if coverage_mode not in VALID_COVERAGE:
        errors.append(f"coverage_mode must be one of {sorted(VALID_COVERAGE)}")

    cols = spec.get("columns") or {}
    required = [cols.get("true_label"), cols.get("prediction"), cols.get("duration_seconds")]
    if any(not x for x in required):
        errors.append("columns.true_label, columns.prediction, and columns.duration_seconds are required")
    else:
        missing = [str(c) for c in required if str(c) not in predictions.columns]
        if missing:
            errors.append(f"prediction columns missing: {missing}")

    if errors:
        return {"contract": RESULT_CONTRACT, "status": "BLOCKED", "production_gate_credit": False, "errors": errors}

    y = pd.to_numeric(predictions[str(cols["true_label"])], errors="coerce")
    pred = pd.to_numeric(predictions[str(cols["prediction"])], errors="coerce")
    duration = pd.to_numeric(predictions[str(cols["duration_seconds"])], errors="coerce")
    if y.isna().any() or not y.dropna().isin([0, 1]).all():
        errors.append("true_label must contain binary 0/1 values")
    if duration.isna().any() or (duration < 0).any():
        errors.append("duration_seconds must be finite and non-negative")

    supported = pd.Series(True, index=predictions.index)
    status_col = cols.get("status")
    if status_col:
        if status_col not in predictions.columns:
            errors.append(f"status column missing: {status_col}")
        else:
            supported = predictions[status_col].astype(str).eq(str(spec.get("supported_status", "SUPPORTED")))
    if pred[supported].isna().any() or not pred[supported].dropna().isin([0, 1]).all():
        errors.append("prediction must contain binary 0/1 values for supported rows")
    if errors:
        return {"contract": RESULT_CONTRACT, "status": "BLOCKED", "production_gate_credit": False, "errors": errors}

    nominal_supported = supported & y.eq(0)
    fp_mask = nominal_supported & pred.eq(1)
    anomaly_supported = supported & y.eq(1)
    fn_mask = anomaly_supported & pred.eq(0)
    nominal_hours = float(duration[nominal_supported].sum() / 3600.0)
    observed_hours = float(duration[supported].sum() / 3600.0)
    false_alarms = int(fp_mask.sum())
    false_alarm_rate = float(false_alarms / nominal_hours) if nominal_hours > 0 else None

    proxy = coverage_mode != "CONTIGUOUS_OPERATION"
    coverage = spec.get("coverage") or {}
    continuity_ok = (
        not proxy
        and coverage.get("complete_observation_window") is True
        and coverage.get("known_data_gaps_seconds") is not None
        and coverage.get("period_start_utc")
        and coverage.get("period_end_utc")
    )
    operational_credit = bool(continuity_ok and not errors)

    status = "PASS" if operational_credit else "PROXY_ONLY" if proxy else "BLOCKED"
    limitations: list[str] = []
    if proxy:
        limitations.append("selected benchmark segments are not a contiguous mission/plant exposure denominator; rate is a benchmark proxy only")
    if not proxy and not continuity_ok:
        limitations.append("continuous operational coverage metadata is incomplete")

    return {
        "contract": RESULT_CONTRACT,
        "status": status,
        "production_gate_credit": operational_credit,
        "coverage_mode": coverage_mode,
        "observed_supported_hours": observed_hours,
        "nominal_supported_hours": nominal_hours,
        "false_alarm_count": false_alarms,
        "false_alarm_rate_per_nominal_hour": false_alarm_rate,
        "missed_anomaly_segments": int(fn_mask.sum()),
        "supported_segments": int(supported.sum()),
        "abstain_or_unsupported_segments": int((~supported).sum()),
        "limitations": limitations,
        "errors": errors,
        "safety": {
            "selected_segment_proxy_is_not_operational_false_alarms_per_hour": True,
            "continuous_exposure_required_for_production_credit": True,
        },
    }


def evaluate_deployment_operations_from_files(spec_path: str | Path, predictions_path: str | Path) -> dict[str, Any]:
    return evaluate_deployment_operations(_load_json(spec_path), pd.read_csv(predictions_path))
