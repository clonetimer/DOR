from pathlib import Path
import json
import numpy as np
import pandas as pd

from m2n import (
    run_diagnostic_model_design,
    predict_diagnostic_design,
    list_diagnostic_families,
)


def _classification_request():
    return {
        "contract": "M2N-DIAGNOSTIC-MODEL-REQUEST/1.0",
        "task": "FAULT_CLASSIFICATION",
        "data": {
            "split_column": "split",
            "train_values": ["train"],
            "validation_values": ["val"],
            "calibration_values": ["cal"],
            "test_values": ["test"],
            "id_columns": ["sample_id"],
        },
        "features": {
            "raw_columns": ["x1", "x2"],
            "mechanism_columns": ["interaction"],
            "views": ["RAW", "MECHANISM"],
        },
        "supervision": {"label_column": "fault_label"},
        "qualification": {
            "min_train_rows": 50,
            "min_validation_rows": 30,
            "min_calibration_rows": 20,
            "min_validation_rows_per_class": 3,
        },
        "candidate_policy": {"families": ["logistic_regression"]},
        "training": {"restarts": [11, 12, 13], "final_seed": 99},
        "selection": {
            "primary_metric": "macro_f1",
            "practical_margin": 0.08,
            "restart_stability": 0.75,
        },
    }


def _write_xor_mechanism_data(path: Path):
    rng = np.random.default_rng(123)
    n = 720
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    interaction = x1 * x2
    labels = np.where(interaction >= 0.0, "FAULT_A", "NORMAL")
    split = np.array(["train"] * 360 + ["val"] * 160 + ["cal"] * 100 + ["test"] * 100, dtype=object)
    idx = rng.permutation(n)
    df = pd.DataFrame({
        "sample_id": np.arange(n),
        "x1": x1[idx],
        "x2": x2[idx],
        "interaction": interaction[idx],
        "fault_label": labels[idx],
        "split": split,
    })
    df.to_csv(path, index=False)


def test_diagnostic_registry_contains_non_neural_and_neural_families():
    ids = {x.id for x in list_diagnostic_families("FAULT_CLASSIFICATION")}
    assert "logistic_regression" in ids
    assert "random_forest" in ids
    assert "mlp" in ids


def test_fault_classification_selects_mechanism_view_and_exports_fec_ready_scores(tmp_path: Path):
    req = _classification_request()
    rp = tmp_path / "request.json"
    rp.write_text(json.dumps(req), encoding="utf-8")
    dp = tmp_path / "faults.csv"
    _write_xor_mechanism_data(dp)

    out = tmp_path / "design"
    r = run_diagnostic_model_design(rp, dp, out)
    assert r.qualification.status in ("QUALIFIED", "CONDITIONALLY_QUALIFIED")
    assert r.status == "SELECTED"
    assert r.selected_candidate_id == "mechanism__logistic_regression"

    bundle = json.loads((out / "diagnostic_model_bundle.json").read_text())
    assert bundle["contract"] == "M2N-DIAGNOSTIC-MODEL-BUNDLE/1.0"
    assert bundle["intended_consumers"] == ["FEC", "DOR"]
    assert bundle["selection"]["selected_candidate_id"] == "mechanism__logistic_regression"
    assert (out / "scores" / "calibration_scores.csv").is_file()
    scores = pd.read_csv(out / "scores" / "calibration_scores.csv")
    assert "score__FAULT_A" in scores.columns
    assert "score__NORMAL" in scores.columns
    assert "true_label" in scores.columns
    contract = json.loads((out / "score_contract.json").read_text())
    assert "FEC class conformal calibration" in contract["eligible_downstream"]
    assert any("root-cause" in x for x in contract["prohibited_claims"])


def test_diagnostic_predict_runs_frozen_selected_model(tmp_path: Path):
    req = _classification_request()
    rp = tmp_path / "request.json"
    rp.write_text(json.dumps(req), encoding="utf-8")
    dp = tmp_path / "faults.csv"
    _write_xor_mechanism_data(dp)
    out = tmp_path / "design"
    r = run_diagnostic_model_design(rp, dp, out)
    assert r.status == "SELECTED"

    pred = tmp_path / "pred.csv"
    predict_diagnostic_design(out, dp, pred)
    got = pd.read_csv(pred)
    assert "mechanism__logistic_regression__prediction" in got.columns
    assert "mechanism__logistic_regression__score__FAULT_A" in got.columns
    assert len(got) == 720


def test_normal_only_anomaly_detection_retains_portfolio_without_fake_winner(tmp_path: Path):
    rng = np.random.default_rng(7)
    n = 260
    df = pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "split": ["train"] * 160 + ["val"] * 100,
    })
    dp = tmp_path / "normal.csv"
    df.to_csv(dp, index=False)
    req = {
        "contract": "M2N-DIAGNOSTIC-MODEL-REQUEST/1.0",
        "task": "ANOMALY_DETECTION",
        "data": {"split_column": "split", "train_values": ["train"], "validation_values": ["val"]},
        "features": {"raw_columns": ["x1", "x2"], "views": ["RAW"]},
        "supervision": {"trusted_normal_train": True},
        "candidate_policy": {"families": ["pca_reconstruction", "isolation_forest"]},
        "training": {"restarts": [1, 2, 3], "final_seed": 4},
        "selection": {"primary_metric": "roc_auc", "practical_margin": 0.02, "restart_stability": 0.75},
    }
    rp = tmp_path / "request.json"
    rp.write_text(json.dumps(req), encoding="utf-8")
    out = tmp_path / "design"
    r = run_diagnostic_model_design(rp, dp, out)
    assert r.status == "PORTFOLIO"
    assert r.selected_candidate_id is None
    assert len(json.loads((out / "diagnostic_model_bundle.json").read_text())["models"]) == 2
    assert any("do not manufacture a winner" in x for x in r.selection.reasons)


def test_no_implicit_random_split_is_allowed(tmp_path: Path):
    df = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "label": ["N", "F", "N", "F"]})
    dp = tmp_path / "d.csv"
    df.to_csv(dp, index=False)
    req = {
        "contract": "M2N-DIAGNOSTIC-MODEL-REQUEST/1.0",
        "task": "FAULT_CLASSIFICATION",
        "data": {"split_column": "split", "train_values": ["train"], "validation_values": ["val"]},
        "features": {"raw_columns": ["x"]},
        "supervision": {"label_column": "label"},
    }
    rp = tmp_path / "r.json"
    rp.write_text(json.dumps(req), encoding="utf-8")
    r = run_diagnostic_model_design(rp, dp, tmp_path / "o")
    assert r.status == "UNSUPPORTED"
    assert any("split_column" in x for x in r.qualification.reasons)
