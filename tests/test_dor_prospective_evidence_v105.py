import copy

from dor.prospective_evidence import canonical_sha256, evaluate_prospective_benchmark


def _policy():
    return {
        "contract": "DOR-PROSPECTIVE-ACCEPTANCE-POLICY/1.0",
        "rules": {
            "opssat_like_anomaly_triage": {
                "minimum_supported_fraction": .98,
                "minimum_overall_recall": .85,
                "maximum_overall_false_positive_rate": .08,
                "minimum_channel_recall_when_channel_has_at_least_5_anomalies": .75,
                "maximum_channel_false_positive_rate_when_channel_has_at_least_20_normals": .12,
            },
            "hydraulic_like_classification": {
                "minimum_macro_f1_on_new_session_or_asset": .90,
                "minimum_per_class_recall": .80,
                "test_group_must_be_unseen": True,
            },
        },
    }


def _anomaly_benchmark(policy):
    return {
        "contract": "DOR-PROSPECTIVE-BENCHMARK/1.0",
        "policy_sha256": canonical_sha256(policy),
        "benchmark_kind": "ANOMALY_TRIAGE",
        "source": {
            "dataset_id": "new-mission-slice",
            "source_sha256": "a" * 64,
            "independent_holdout": True,
            "new_session_or_asset": True,
            "first_seen_after_policy_freeze": True,
        },
        "freeze": {
            "candidate_set_frozen_before_test": True,
            "hyperparameters_frozen_before_test": True,
            "acceptance_thresholds_frozen_before_test": True,
            "test_labels_must_not_change_model_or_threshold": True,
        },
        "integrity": {
            "historical_test_results_known_before_current_design": False,
            "test_labels_used_for_candidate_selection": False,
            "test_labels_used_for_threshold_calibration": False,
        },
        "metrics": {
            "supported_fraction": .99,
            "overall": {"recall": .90, "false_positive_rate": .05},
            "channel_metrics": {
                "A": {"anomaly_n": 10, "normal_n": 30, "recall": .80, "false_positive_rate": .10},
            },
        },
    }


def test_clean_prospective_benchmark_can_pass():
    policy = _policy()
    result = evaluate_prospective_benchmark(policy, _anomaly_benchmark(policy))
    assert result["status"] == "PASS"
    assert result["production_gate_credit"] is True


def test_passing_metrics_do_not_repair_known_historical_test():
    policy = _policy()
    benchmark = _anomaly_benchmark(policy)
    benchmark["integrity"]["historical_test_results_known_before_current_design"] = True
    result = evaluate_prospective_benchmark(policy, benchmark)
    assert result["metric_summary"]["all_passed"] is True
    assert result["status"] == "INELIGIBLE"
    assert result["production_gate_credit"] is False


def test_exact_policy_hash_is_required():
    policy = _policy()
    benchmark = _anomaly_benchmark(policy)
    benchmark["policy_sha256"] = "0" * 64
    result = evaluate_prospective_benchmark(policy, benchmark)
    assert result["status"] == "INELIGIBLE"
    assert any("policy_sha256" in x for x in result["eligibility"]["failures"])


def test_metric_failure_remains_failure_for_eligible_benchmark():
    policy = _policy()
    benchmark = _anomaly_benchmark(policy)
    benchmark["metrics"]["overall"]["recall"] = .50
    result = evaluate_prospective_benchmark(policy, benchmark)
    assert result["status"] == "FAIL"
    assert result["eligibility"]["eligible"] is True
