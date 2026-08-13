from __future__ import annotations

import io
import json
from pathlib import Path
import zipfile

import numpy as np

from dor.auto_design import run_auto_design
import pandas as pd

from dor.public_adapters import prepare_spacecraft_dataset, prepare_uci_hydraulic


def _official_shape_fixture(path: Path) -> None:
    profile = np.tile(np.array([100, 100, 0, 130, 0]), (2205, 1))
    profile[::3, 0] = 20
    profile[1::4, 1] = 80
    profile[2::5, 2] = 2
    profile[3::7, 3] = 90
    signal = np.sin(np.arange(60)[None, :] / 8) + np.arange(2205)[:, None] / 2205
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value, fmt in (("profile.txt", profile, "%d"), ("TS1.txt", signal, "%.6f")):
            buffer = io.StringIO(); np.savetxt(buffer, value, delimiter="\t", fmt=fmt)
            archive.writestr(f"hydraulic_systems/{name}", buffer.getvalue())


def test_uci_adapter_binds_source_and_runs_multirate_design(tmp_path: Path):
    archive = tmp_path / "uci.zip"; _official_shape_fixture(archive)
    prepared = tmp_path / "prepared"
    binding = prepare_uci_hydraulic(archive, prepared, sensors=["TS1"], block_size=5)
    assert binding["cycles"] == 2205
    assert binding["sensors"] == ["TS1"]
    manifest = json.loads((prepared / "dataset_manifest.json").read_text())
    assert manifest["modality"] == "MULTIRATE_NPZ"
    request = {
        "contract": "DOR-AUTO-DIAGNOSTIC-DESIGN-REQUEST/1.0", "stages": ["DETECTION"],
        "budget": {"max_candidates": 1, "max_wall_seconds": 60},
        "search": {"seeds": [1], "final_seed": 2},
        "candidates": {"detection": {"families": ["pca_reconstruction"]}}
    }
    request_path = tmp_path / "request.json"; request_path.write_text(json.dumps(request))
    result = run_auto_design(prepared / "device_card.json", prepared / "dataset_manifest.json", prepared / "uci_hydraulic_multirate.npz", request_path, tmp_path / "design")
    assert result["data"]["modality"] == "MULTIRATE_NPZ"
    assert result["data"]["rows_after_ingestion"] == 2205


def _spacecraft_fixture(root: Path, *, phm: bool = True) -> None:
    rows = []
    for subsystem in (["eps", "thermal"] if phm else ["adcs"]):
        for mc in range(10):
            for condition in ("nominal", "battery_fault"):
                sample_id = f"{subsystem}:{condition}:mc_{mc:02d}"
                relative = Path(subsystem) / condition / f"mc_{mc:02d}" / "telemetry.csv"
                path = root / relative; path.parent.mkdir(parents=True, exist_ok=True)
                signal = np.linspace(0, 1, 12) + (condition != "nominal") * 2
                pd.DataFrame({
                    "time_s": np.arange(12), "packet_valid": 1,
                    "eps.measurement": signal if subsystem == "eps" else signal / 2,
                    "thermal.measurement": signal if subsystem == "thermal" else signal / 2,
                    "eps.execution_residual": signal * 10,
                }).to_csv(path, index=False)
                row = {
                    "sample_id": sample_id, "trajectory_pair_id": f"{subsystem}:mc_{mc:02d}",
                    "sample_index": len(rows), "pair_seed": mc, "mission_id": f"mission_{subsystem}",
                    "condition_id": condition, "diagnostic_class": "nominal" if condition == "nominal" else f"{subsystem}_fault",
                    "telemetry_file": str(relative), "labels_file": "not_consumed.csv", "row_count": 12, "field_count": 5,
                }
                if phm:
                    row.update({"subsystem": subsystem, "fault_injection_phase": "none" if condition == "nominal" else "settled"})
                else:
                    row.update({"scenario_id": "nominal_baseline" if condition == "nominal" else "fault", "operating_phase_at_fault": "settled"})
                rows.append(row)
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "sample_index.csv", index=False)
    (root / "dataset_manifest.json").write_text(json.dumps({"schema": "sat-sim.whole-spacecraft-phm-dataset.v6.0"} if phm else {"schema": "adcs"}))


def test_spacecraft_adapter_excludes_truth_and_groups_trajectories(tmp_path: Path):
    source = tmp_path / "phm"; _spacecraft_fixture(source)
    out = tmp_path / "prepared"
    binding = prepare_spacecraft_dataset(source, out, feature_profile="causal_raw")
    frame = pd.read_csv(out / "spacecraft_features.csv.gz")
    manifest = json.loads((out / "dataset_manifest.json").read_text())
    assert binding["source_schema"] == "PHM_V6"
    assert binding["channel_count"] == 2
    assert all("residual" not in c and "valid" not in c for c in manifest["features"]["raw_columns"])
    assert not ({"event_parameters", "labels_file"} & set(frame.columns))
    assert frame.groupby("trajectory_pair_id")["split"].nunique().max() == 1
    assert set(frame["split"]) == {"train", "validation", "calibration", "test"}
    assert set(frame["localization_label"]) == {"healthy", "eps", "thermal"}


def test_spacecraft_adapter_rejects_escaping_telemetry_path(tmp_path: Path):
    source = tmp_path / "phm"; _spacecraft_fixture(source)
    index = pd.read_csv(source / "sample_index.csv")
    index.loc[0, "telemetry_file"] = "../outside.csv"
    index.to_csv(source / "sample_index.csv", index=False)
    try:
        prepare_spacecraft_dataset(source, tmp_path / "out")
    except ValueError as exc:
        assert "escapes dataset root" in str(exc)
    else:
        raise AssertionError("path traversal must be rejected")


def test_spacecraft_adapter_encodes_partial_and_total_missingness(tmp_path: Path):
    source = tmp_path / "phm"; _spacecraft_fixture(source)
    index = pd.read_csv(source / "sample_index.csv")
    path = source / index.iloc[0]["telemetry_file"]
    telemetry = pd.read_csv(path)
    telemetry.loc[:, "eps.measurement"] = np.nan
    telemetry.loc[:5, "thermal.measurement"] = np.nan
    telemetry.to_csv(path, index=False)
    out = tmp_path / "prepared"
    prepare_spacecraft_dataset(source, out)
    frame = pd.read_csv(out / "spacecraft_features.csv.gz")
    first = frame.loc[frame["sample_id"] == index.iloc[0]["sample_id"]].iloc[0]
    assert first["eps.measurement__observed_fraction"] == 0.0
    assert first["eps.measurement__mean"] == 0.0
    assert first["thermal.measurement__observed_fraction"] == 0.5
