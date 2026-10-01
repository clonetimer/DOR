from __future__ import annotations

"""Prospective benchmark qualification for frozen DOR acceptance policies.

This module is intentionally stricter than ordinary model evaluation.  It does
not merely compare metrics to thresholds; it first checks whether the benchmark
is eligible to provide prospective evidence at all.  A benchmark can therefore
have excellent numbers and still receive no production credit when the test set
was previously observed, the candidate set changed after the policy freeze, or
source identity is not bound.
"""

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

POLICY_CONTRACT = "DOR-PROSPECTIVE-ACCEPTANCE-POLICY/1.0"
BENCHMARK_CONTRACT = "DOR-PROSPECTIVE-BENCHMARK/1.0"
RESULT_CONTRACT = "DOR-PROSPECTIVE-EVALUATION/1.0"


def _load(value: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return json.loads(Path(value).read_text(encoding="utf-8"))


def canonical_sha256(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256(raw).hexdigest()


def _required_bool(container: dict[str, Any], key: str, failures: list[str], *, expected: bool = True) -> None:
    if container.get(key) is not expected:
        failures.append(f"{key} must be {expected}")


def _compare(value: Any, threshold: float, op: str) -> bool:
    if value is None:
        return False
    x = float(value)
    if op == ">=":
        return x >= threshold
    if op == "<=":
        return x <= threshold
    raise ValueError(op)


def _anomaly_checks(policy: dict[str, Any], metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rules = policy["rules"]["opssat_like_anomaly_triage"]
    checks: list[dict[str, Any]] = []

    def add(name: str, value: Any, threshold: float, op: str) -> None:
        checks.append({"metric": name, "value": value, "operator": op, "threshold": threshold, "pass": _compare(value, threshold, op)})

    add("supported_fraction", metrics.get("supported_fraction"), float(rules["minimum_supported_fraction"]), ">=")
    overall = metrics.get("overall", {})
    add("overall_recall", overall.get("recall"), float(rules["minimum_overall_recall"]), ">=")
    add("overall_false_positive_rate", overall.get("false_positive_rate"), float(rules["maximum_overall_false_positive_rate"]), "<=")

    for channel, cm in sorted((metrics.get("channel_metrics") or {}).items()):
        anomaly_n = int(cm.get("anomaly_n") or 0)
        normal_n = int(cm.get("normal_n") or 0)
        if anomaly_n >= 5:
            add(f"channel[{channel}].recall", cm.get("recall"), float(rules["minimum_channel_recall_when_channel_has_at_least_5_anomalies"]), ">=")
        if normal_n >= 20:
            add(f"channel[{channel}].false_positive_rate", cm.get("false_positive_rate"), float(rules["maximum_channel_false_positive_rate_when_channel_has_at_least_20_normals"]), "<=")
    return checks


def _classification_checks(policy: dict[str, Any], metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rules = policy["rules"]["hydraulic_like_classification"]
    checks: list[dict[str, Any]] = []
    macro = metrics.get("macro_f1")
    checks.append({
        "metric": "macro_f1",
        "value": macro,
        "operator": ">=",
        "threshold": float(rules["minimum_macro_f1_on_new_session_or_asset"]),
        "pass": _compare(macro, float(rules["minimum_macro_f1_on_new_session_or_asset"]), ">="),
    })
    minimum = float(rules["minimum_per_class_recall"])
    for label, value in sorted((metrics.get("per_class_recall") or {}).items()):
        checks.append({"metric": f"class[{label}].recall", "value": value, "operator": ">=", "threshold": minimum, "pass": _compare(value, minimum, ">=")})
    if not metrics.get("per_class_recall"):
        checks.append({"metric": "per_class_recall", "value": None, "operator": "required", "threshold": minimum, "pass": False})
    return checks


def evaluate_prospective_benchmark(policy: dict[str, Any], benchmark: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    eligibility_failures: list[str] = []

    if policy.get("contract") != POLICY_CONTRACT:
        errors.append(f"policy contract must be {POLICY_CONTRACT}")
    if benchmark.get("contract") != BENCHMARK_CONTRACT:
        errors.append(f"benchmark contract must be {BENCHMARK_CONTRACT}")

    expected_policy_sha = canonical_sha256(policy)
    if benchmark.get("policy_sha256") != expected_policy_sha:
        eligibility_failures.append("benchmark policy_sha256 does not bind the exact frozen policy")

    source = benchmark.get("source") or {}
    freeze = benchmark.get("freeze") or {}
    integrity = benchmark.get("integrity") or {}

    if not str(source.get("dataset_id", "")).strip():
        eligibility_failures.append("source.dataset_id is required")
    if not str(source.get("source_sha256", "")).strip():
        eligibility_failures.append("source.source_sha256 is required")
    _required_bool(source, "independent_holdout", eligibility_failures)
    _required_bool(source, "new_session_or_asset", eligibility_failures)
    _required_bool(source, "first_seen_after_policy_freeze", eligibility_failures)

    _required_bool(freeze, "candidate_set_frozen_before_test", eligibility_failures)
    _required_bool(freeze, "hyperparameters_frozen_before_test", eligibility_failures)
    _required_bool(freeze, "acceptance_thresholds_frozen_before_test", eligibility_failures)
    _required_bool(freeze, "test_labels_must_not_change_model_or_threshold", eligibility_failures)

    _required_bool(integrity, "historical_test_results_known_before_current_design", eligibility_failures, expected=False)
    _required_bool(integrity, "test_labels_used_for_candidate_selection", eligibility_failures, expected=False)
    _required_bool(integrity, "test_labels_used_for_threshold_calibration", eligibility_failures, expected=False)

    kind = str(benchmark.get("benchmark_kind", "")).strip()
    metrics = benchmark.get("metrics") or {}
    checks: list[dict[str, Any]] = []
    if not errors:
        try:
            if kind == "ANOMALY_TRIAGE":
                checks = _anomaly_checks(policy, metrics)
            elif kind == "CLASSIFICATION":
                checks = _classification_checks(policy, metrics)
                split_rules = policy["rules"]["hydraulic_like_classification"]
                if split_rules.get("test_group_must_be_unseen") and source.get("test_group_unseen") is not True:
                    eligibility_failures.append("source.test_group_unseen must be true for classification")
            else:
                errors.append("benchmark_kind must be ANOMALY_TRIAGE or CLASSIFICATION")
        except KeyError as exc:
            errors.append(f"policy is missing required rule: {exc}")

    eligible = not errors and not eligibility_failures
    metrics_pass = bool(checks) and all(bool(x.get("pass")) for x in checks)
    if not eligible:
        status = "INELIGIBLE"
    elif metrics_pass:
        status = "PASS"
    else:
        status = "FAIL"

    return {
        "contract": RESULT_CONTRACT,
        "status": status,
        "production_gate_credit": bool(status == "PASS"),
        "benchmark_kind": kind,
        "policy_sha256": expected_policy_sha,
        "dataset_id": source.get("dataset_id"),
        "source_sha256": source.get("source_sha256"),
        "eligibility": {
            "eligible": eligible,
            "failures": eligibility_failures,
            "principle": "metric success cannot repair a non-prospective or unbound evaluation",
        },
        "metric_checks": checks,
        "metric_summary": {
            "checks": len(checks),
            "passed": sum(bool(x.get("pass")) for x in checks),
            "all_passed": metrics_pass,
        },
        "errors": errors,
        "safety": {
            "current_or_historical_test_outcomes_must_not_drive_model_changes": True,
            "exact_frozen_policy_binding_required": True,
            "new_session_or_asset_required_for_production_credit": True,
        },
    }


def evaluate_prospective_benchmark_from_files(policy_path: str | Path, benchmark_path: str | Path) -> dict[str, Any]:
    return evaluate_prospective_benchmark(_load(policy_path), _load(benchmark_path))
