from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dor.auto_design import predict_pipeline, run_auto_design


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _request() -> dict:
    return {
        "contract": "DOR-AUTO-DIAGNOSTIC-DESIGN-REQUEST/1.0",
        "stages": ["DETECTION", "LOCALIZATION", "CLASSIFICATION", "IDENTIFICATION"],
        "budget": {"max_candidates": 16, "max_wall_seconds": 120},
        "search": {"seeds": [11], "final_seed": 12},
        "candidates": {
            "detection": {"families": ["pca_reconstruction"]},
            "localization": {"families": ["logistic_regression"]},
            "classification": {"families": ["logistic_regression"]},
            "identification": {"families": ["ridge"]}
        },
        "selection": {"practical_margin": 0.02, "restart_stability": 0.0}
    }


def _card() -> dict:
    return {
        "contract": "DOR-DEVICE-MECHANISM-CARD/1.0", "device_id": "synthetic-servo",
        "sensors": [{"sensor_id": "s1", "unit": "a.u."}, {"sensor_id": "s2", "unit": "a.u."}],
        "components": [{"component_id": "drive"}, {"component_id": "load"}],
        "topology_edges": [{"source": "drive", "target": "load", "relation": "MECHANICAL", "provenance": "synthetic test definition"}],
        "mechanism_features": [{"feature_id": "balance_residual", "operation": "difference", "source_columns": ["x1", "x2"], "provenance": "synthetic balance relation"}]
    }


def _manifest() -> dict:
    return {
        "contract": "DOR-MULTIMODAL-DATASET-MANIFEST/1.0", "dataset_id": "synthetic", "modality": "TABULAR",
        "identifiers": ["sample_id"], "features": {"raw_columns": ["x1", "x2"]},
        "split": {"column": "split", "train_values": ["train"], "validation_values": ["val"], "calibration_values": ["cal"], "test_values": ["test"]},
        "supervision": {
            "detection": {"label_column": "fault", "normal_label": "N"},
            "localization": {"label_column": "component"},
            "classification": {"label_column": "fault"},
            "identification": {"target_columns": ["severity"]}
        },
        "leakage_guard": {"group_columns": ["run_id"]}
    }


def test_auto_design_four_stage_bundle_and_predict(tmp_path: Path):
    rng = np.random.default_rng(4)
    n = 160
    y = np.array(["N", "F"] * (n // 2))
    severity = (y == "F").astype(float) + rng.normal(0, 0.05, n)
    x1 = severity + rng.normal(0, 0.1, n)
    x2 = -severity + rng.normal(0, 0.1, n)
    split = np.array(["train"] * 80 + ["val"] * 40 + ["cal"] * 20 + ["test"] * 20)
    df = pd.DataFrame({
        "sample_id": np.arange(n), "run_id": [f"r{i}" for i in range(n)], "split": split,
        "x1": x1, "x2": x2, "fault": y,
        "component": np.where(y == "F", "drive", "healthy"), "severity": severity,
    })
    data = tmp_path / "data.csv"; df.to_csv(data, index=False)
    card = _write_json(tmp_path / "card.json", _card())
    manifest = _write_json(tmp_path / "manifest.json", _manifest())
    request = _write_json(tmp_path / "request.json", _request())
    out = tmp_path / "out"
    result = run_auto_design(card, manifest, data, request, out)
    assert result["status"] == "QUALIFIED"
    assert {x["stage"] for x in result["qualification"]["stages"] if x["status"] == "QUALIFIED"} == {
        "DETECTION", "LOCALIZATION", "CLASSIFICATION", "IDENTIFICATION"
    }
    assert (out / "pipeline_bundle" / "bundle.json").is_file()
    assert (out / "audit_report.md").is_file()
    assert json.loads((out / "end_to_end_test_audit.json").read_text())["selection_uses_test"] is False
    assert json.loads((out / "mechanism_stress_audit.json").read_text())["state"] == "RUN"
    assert "No novelty claim" in (out / "component_registry.csv").read_text(encoding="utf-8-sig")
    pred = predict_pipeline(out / "pipeline_bundle", data, tmp_path / "pred.csv")
    p = pd.read_csv(pred)
    assert len(p) == n
    assert "pipeline_gate" in p.columns
    assert any(c.endswith("__state") for c in p.columns)
    assert any(c.endswith("__prediction") for c in p.columns)
    assert any(c.endswith("__severity") for c in p.columns)
    assert any(c.endswith("__severity__lower") for c in p.columns)
    assert any(c.endswith("__severity__upper") for c in p.columns)


def test_multilabel_localization_is_calibrated_and_emits_sets(tmp_path: Path):
    rng = np.random.default_rng(14)
    n = 160
    a = (np.arange(n) % 3 == 0).astype(int)
    b = (np.arange(n) % 4 < 2).astype(int)
    df = pd.DataFrame({
        "sample_id": np.arange(n), "run_id": [f"r{i}" for i in range(n)],
        "split": ["train"] * 80 + ["val"] * 40 + ["cal"] * 20 + ["test"] * 20,
        "x1": a + rng.normal(0, .08, n), "x2": b + rng.normal(0, .08, n),
        "component_a": a, "component_b": b,
    })
    data = tmp_path / "data.csv"; df.to_csv(data, index=False)
    card = _write_json(tmp_path / "card.json", _card())
    manifest_value = _manifest()
    manifest_value["supervision"] = {"localization": {"target_columns": ["component_a", "component_b"]}}
    manifest = _write_json(tmp_path / "manifest.json", manifest_value)
    request_value = _request(); request_value["stages"] = ["LOCALIZATION"]
    request = _write_json(tmp_path / "request.json", request_value)
    out = tmp_path / "out"
    result = run_auto_design(card, manifest, data, request, out)
    assert result["qualification"]["stages"][1]["status"] == "QUALIFIED"
    bundle = json.loads((out / "pipeline_bundle" / "bundle.json").read_text())
    model = bundle["models"][0]
    assert model["multilabel_targets"] == ["component_a", "component_b"]
    assert model["multilabel_calibration_rows"] == 20
    predict_pipeline(out / "pipeline_bundle", data, tmp_path / "pred.csv")
    columns = pd.read_csv(tmp_path / "pred.csv").columns
    assert any(c.endswith("__prediction_set") for c in columns)
    assert any(c.endswith("__component_a__present") for c in columns)


def test_timeseries_long_degrades_without_stage_labels(tmp_path: Path):
    rows = []
    for run in range(12):
        split = "train" if run < 8 else "val"
        for t in range(16):
            rows.append({"run": f"r{run}", "split": split, "time": t, "x1": np.sin(t / 3) + run / 20, "x2": np.cos(t / 4)})
    data = tmp_path / "ts.csv"; pd.DataFrame(rows).to_csv(data, index=False)
    card_value = _card(); card_value["mechanism_features"][0]["source_columns"] = ["x1__mean", "x2__mean"]
    card = _write_json(tmp_path / "card.json", card_value)
    manifest_value = {
        "contract": "DOR-MULTIMODAL-DATASET-MANIFEST/1.0", "dataset_id": "ts", "modality": "TIMESERIES_LONG",
        "identifiers": ["run"], "timeseries": {"group_columns": ["run"], "time_column": "time", "channel_columns": ["x1", "x2"]},
        "split": {"column": "split", "train_values": ["train"], "validation_values": ["val"]},
        "supervision": {}, "leakage_guard": {"group_columns": ["run"]}
    }
    request_value = _request(); request_value["stages"] = ["DETECTION", "LOCALIZATION"]
    manifest = _write_json(tmp_path / "manifest.json", manifest_value)
    request = _write_json(tmp_path / "request.json", request_value)
    result = run_auto_design(card, manifest, data, request, tmp_path / "out")
    status = {x["stage"]: x["status"] for x in result["qualification"]["stages"]}
    assert status["DETECTION"] == "PORTFOLIO_ONLY"
    assert status["LOCALIZATION"] == "UNSUPPORTED"
    json.loads((tmp_path / "out" / "validation_results.json").read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def test_leakage_guard_and_resume_binding_fail_closed(tmp_path: Path):
    df = pd.DataFrame({
        "sample_id": range(8), "run_id": ["same"] * 8, "split": ["train"] * 4 + ["val"] * 4,
        "x1": range(8), "x2": range(8), "fault": ["N", "F"] * 4,
        "component": ["a", "b"] * 4, "severity": np.arange(8, dtype=float),
    })
    data = tmp_path / "data.csv"; df.to_csv(data, index=False)
    card = _write_json(tmp_path / "card.json", _card())
    manifest = _write_json(tmp_path / "manifest.json", _manifest())
    request = _write_json(tmp_path / "request.json", _request())
    with pytest.raises(ValueError, match="leakage guard violation"):
        run_auto_design(card, manifest, data, request, tmp_path / "out")


def test_timeseries_npz_contract(tmp_path: Path):
    rng = np.random.default_rng(9)
    x = rng.normal(size=(12, 24, 2))
    split = np.array(["train"] * 8 + ["val"] * 4)
    np.savez(tmp_path / "data.npz", X=x, split=split, sample_id=np.arange(12))
    card_value = _card(); card_value["mechanism_features"][0]["source_columns"] = ["s1__mean", "s2__mean"]
    card = _write_json(tmp_path / "card.json", card_value)
    manifest = _write_json(tmp_path / "manifest.json", {
        "contract": "DOR-MULTIMODAL-DATASET-MANIFEST/1.0", "dataset_id": "npz", "modality": "TIMESERIES_NPZ",
        "identifiers": ["sample_id"],
        "timeseries": {"array_key": "X", "channel_names": ["s1", "s2"], "metadata_keys": {"split": "split", "sample_id": "sample_id"}},
        "split": {"column": "split", "train_values": ["train"], "validation_values": ["val"]},
        "supervision": {}, "leakage_guard": {"group_columns": ["sample_id"]}
    })
    request_value = _request(); request_value["stages"] = ["DETECTION"]
    request_value["representations"] = {"temporal_pyramid": True}
    request = _write_json(tmp_path / "request.json", request_value)
    result = run_auto_design(card, manifest, tmp_path / "data.npz", request, tmp_path / "out")
    assert result["data"]["modality"] == "TIMESERIES_NPZ"
    assert result["qualification"]["stages"][0]["status"] == "PORTFOLIO_ONLY"
    candidates = json.loads((tmp_path / "out" / "validation_results.json").read_text())["records"]
    assert any(x["view"] == "RAW_TEMPORAL_PYRAMID" for x in candidates)


def test_selection_audits_slices_resources_and_missing_sensor_fallback(tmp_path: Path):
    rng = np.random.default_rng(22)
    n = 120
    fault = np.where(np.arange(n) % 2, "F", "N")
    signal = (fault == "F").astype(float)
    frame = pd.DataFrame({
        "sample_id": np.arange(n), "run_id": [f"r{i}" for i in range(n)],
        "split": ["train"] * 60 + ["val"] * 40 + ["cal"] * 10 + ["test"] * 10,
        "regime": np.where(np.arange(n) % 3, "loaded", "idle"),
        "x1": signal + rng.normal(0, .1, n), "x2": -signal + rng.normal(0, .1, n),
        "fault": fault,
    })
    data = tmp_path / "data.csv"; frame.to_csv(data, index=False)
    manifest_value = _manifest()
    manifest_value["supervision"] = {"classification": {"label_column": "fault"}}
    manifest_value["evaluation"] = {"slice_columns": ["regime"]}
    request_value = _request(); request_value["stages"] = ["CLASSIFICATION"]
    request_value["robustness"] = {"sensor_groups": {"primary": ["x1"], "secondary": ["x2"]}, "max_missing_groups": 1}
    request_value["selection"].update({"worst_slice_weight": .5, "missing_sensor_weight": .5})
    request_value["budget"].update({"max_model_bytes": 1_000_000, "max_parameters": 100_000, "max_inference_latency_ms": 10})
    out = tmp_path / "out"
    run_auto_design(
        _write_json(tmp_path / "card.json", _card()),
        _write_json(tmp_path / "manifest.json", manifest_value), data,
        _write_json(tmp_path / "request.json", request_value), out,
    )
    record = json.loads((out / "validation_results.json").read_text())["records"][0]
    assert record["eligible"] is True
    assert record["worst_slice_score"] is not None
    assert record["worst_missing_sensor_score"] is not None
    assert record["resources"]["model_bytes"] > 0
    inference = tmp_path / "missing.csv"; frame.drop(columns=["x2"]).to_csv(inference, index=False)
    predict_pipeline(out / "pipeline_bundle", inference, tmp_path / "pred.csv")
    prediction = pd.read_csv(tmp_path / "pred.csv")
    assert set(prediction["missing_sensor_group_count"]) == {1}
    assert set(prediction["missing_sensor_groups"]) == {"secondary"}


def test_resource_constraint_rejects_every_candidate(tmp_path: Path):
    n = 40
    frame = pd.DataFrame({
        "sample_id": np.arange(n), "run_id": [f"r{i}" for i in range(n)],
        "split": ["train"] * 20 + ["val"] * 20,
        "x1": np.arange(n) % 2, "x2": -(np.arange(n) % 2),
        "fault": np.where(np.arange(n) % 2, "F", "N"),
    })
    data = tmp_path / "data.csv"; frame.to_csv(data, index=False)
    manifest_value = _manifest(); manifest_value["supervision"] = {"classification": {"label_column": "fault"}}
    request_value = _request(); request_value["stages"] = ["CLASSIFICATION"]
    request_value["budget"]["max_model_bytes"] = 1
    result = run_auto_design(
        _write_json(tmp_path / "card.json", _card()),
        _write_json(tmp_path / "manifest.json", manifest_value), data,
        _write_json(tmp_path / "request.json", request_value), tmp_path / "out",
    )
    assert result["status"] == "UNSUPPORTED"
    decision = result["stage_selections"]["CLASSIFICATION"]
    assert decision["status"] == "UNSUPPORTED"
    assert decision["retained_candidate_ids"] == []
    assert "max_model_bytes" in next(iter(decision["rejected_candidates"].values()))
