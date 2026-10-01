import pandas as pd

from dor.deployment_ops import evaluate_deployment_operations


def _frame():
    return pd.DataFrame({
        "truth": [0, 0, 1, 1],
        "prediction": [0, 1, 1, 0],
        "duration_s": [1800, 1800, 900, 900],
        "status": ["SUPPORTED"] * 4,
    })


def _spec(mode):
    return {
        "contract": "DOR-DEPLOYMENT-OPS-EVIDENCE/1.0",
        "coverage_mode": mode,
        "supported_status": "SUPPORTED",
        "columns": {"true_label": "truth", "prediction": "prediction", "duration_seconds": "duration_s", "status": "status"},
        "coverage": {},
    }


def test_selected_segments_are_proxy_only_even_with_a_rate():
    result = evaluate_deployment_operations(_spec("SELECTED_SEGMENTS"), _frame())
    assert result["status"] == "PROXY_ONLY"
    assert result["production_gate_credit"] is False
    assert result["false_alarm_rate_per_nominal_hour"] == 1.0


def test_contiguous_operation_requires_complete_coverage_metadata():
    result = evaluate_deployment_operations(_spec("CONTIGUOUS_OPERATION"), _frame())
    assert result["status"] == "BLOCKED"
    assert result["production_gate_credit"] is False


def test_contiguous_operation_with_declared_coverage_can_receive_credit():
    spec = _spec("CONTIGUOUS_OPERATION")
    spec["coverage"] = {
        "complete_observation_window": True,
        "known_data_gaps_seconds": 0,
        "period_start_utc": "2026-01-01T00:00:00Z",
        "period_end_utc": "2026-01-01T01:30:00Z",
    }
    result = evaluate_deployment_operations(spec, _frame())
    assert result["status"] == "PASS"
    assert result["production_gate_credit"] is True
