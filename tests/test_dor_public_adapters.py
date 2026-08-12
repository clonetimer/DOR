from __future__ import annotations

import io
import json
from pathlib import Path
import zipfile

import numpy as np

from dor.auto_design import run_auto_design
from dor.public_adapters import prepare_uci_hydraulic


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
