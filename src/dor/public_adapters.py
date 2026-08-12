from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np


UCI_HYDRAULIC_URL = "https://archive.ics.uci.edu/static/public/447/condition%2Bmonitoring%2Bof%2Bhydraulic%2Bsystems.zip"
UCI_SENSORS = {
    "PS1": (100, "bar"), "PS2": (100, "bar"), "PS3": (100, "bar"),
    "PS4": (100, "bar"), "PS5": (100, "bar"), "PS6": (100, "bar"),
    "EPS1": (100, "W"), "FS1": (10, "l/min"), "FS2": (10, "l/min"),
    "TS1": (1, "degC"), "TS2": (1, "degC"), "TS3": (1, "degC"),
    "TS4": (1, "degC"), "VS1": (1, "mm/s"), "CE": (1, "%"),
    "CP": (1, "kW"), "SE": (1, "%"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _member(archive: zipfile.ZipFile, filename: str) -> str:
    matches = [name for name in archive.namelist() if name.rsplit("/", 1)[-1].lower() == filename.lower()]
    if len(matches) != 1:
        raise ValueError(f"archive must contain exactly one {filename!r}; found {len(matches)}")
    return matches[0]


def _read_matrix(archive: zipfile.ZipFile, filename: str) -> np.ndarray:
    raw = archive.read(_member(archive, filename))
    value = np.loadtxt(io.BytesIO(raw), dtype=np.float32)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError(f"{filename} is not a finite two-dimensional matrix")
    return value


def prepare_uci_hydraulic(
    archive_path: str | Path,
    output_dir: str | Path,
    *,
    sensors: list[str] | None = None,
    seed: int = 447,
    block_size: int = 5,
) -> dict:
    source = Path(archive_path)
    if not source.is_file() or not zipfile.is_zipfile(source):
        raise ValueError("UCI Hydraulic source must be the official ZIP archive")
    selected = list(dict.fromkeys(sensors or UCI_SENSORS))
    unknown = sorted(set(selected) - set(UCI_SENSORS))
    if unknown:
        raise ValueError(f"unknown UCI Hydraulic sensors: {unknown}")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    with zipfile.ZipFile(source) as archive:
        profile = _read_matrix(archive, "profile.txt")
        if profile.shape != (2205, 5):
            raise ValueError(f"official profile shape must be (2205, 5), got {profile.shape}")
        arrays = {sensor: _read_matrix(archive, f"{sensor}.txt") for sensor in selected}
    for sensor, values in arrays.items():
        expected_width = 60 * UCI_SENSORS[sensor][0]
        if values.shape != (2205, expected_width):
            raise ValueError(f"{sensor} shape must be (2205, {expected_width}), got {values.shape}")

    cycles = np.arange(2205)
    blocks = cycles // block_size
    unique_blocks = np.unique(blocks)
    rng = np.random.default_rng(seed); rng.shuffle(unique_blocks)
    n = len(unique_blocks)
    cut1, cut2, cut3 = int(.60 * n), int(.80 * n), int(.90 * n)
    assignment = {}
    for role, values in zip(("train", "val", "cal", "test"), np.split(unique_blocks, [cut1, cut2, cut3])):
        assignment.update({int(block): role for block in values})
    split = np.asarray([assignment[int(block)] for block in blocks])
    cooler, valve, pump, accumulator, stable = profile.astype(int).T
    normal = (cooler == 100) & (valve == 100) & (pump == 0) & (accumulator == 130) & (stable == 0)
    payload = {sensor: values for sensor, values in arrays.items()}
    payload.update({
        "sample_id": cycles, "split_group": np.asarray([f"block_{x}" for x in blocks]), "split": split,
        "cooler": cooler, "valve": valve, "pump": pump, "accumulator": accumulator, "stable": stable,
        "fault_state": np.where(normal, "NORMAL", "ABNORMAL"),
        "fault_cooler": (cooler != 100).astype(np.int8), "fault_valve": (valve != 100).astype(np.int8),
        "fault_pump": (pump != 0).astype(np.int8), "fault_accumulator": (accumulator != 130).astype(np.int8),
    })
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    npz_path = out / "uci_hydraulic_multirate.npz"
    np.savez_compressed(npz_path, **payload)
    metadata_keys = {name: name for name in payload if name not in arrays}
    manifest = {
        "contract": "DOR-MULTIMODAL-DATASET-MANIFEST/1.0", "dataset_id": "uci-hydraulic-raw-447",
        "modality": "MULTIRATE_NPZ", "identifiers": ["sample_id"],
        "timeseries": {"sensor_arrays": {s: s for s in selected}, "metadata_keys": metadata_keys},
        "split": {"column": "split", "train_values": ["train"], "validation_values": ["val"], "calibration_values": ["cal"], "test_values": ["test"]},
        "supervision": {
            "detection": {"label_column": "fault_state", "normal_label": "NORMAL"},
            "localization": {"target_columns": ["fault_cooler", "fault_valve", "fault_pump", "fault_accumulator"]}
        },
        "leakage_guard": {"group_columns": ["split_group"], "block_size_cycles": block_size}
    }
    card = {
        "contract": "DOR-DEVICE-MECHANISM-CARD/1.0", "device_id": "uci-hydraulic-test-rig",
        "sensors": [{"sensor_id": s, "sampling_hz": UCI_SENSORS[s][0], "unit": UCI_SENSORS[s][1]} for s in selected],
        "components": [{"component_id": x} for x in ("cooler", "valve", "pump", "accumulator")],
        "topology_edges": [], "mechanism_features": []
    }
    binding = {
        "contract": "DOR-PUBLIC-SOURCE-BINDING/1.0", "dataset_id": "uci-hydraulic",
        "official_url": UCI_HYDRAULIC_URL, "source_sha256": _sha256(source),
        "prepared_npz_sha256": _sha256(npz_path), "cycles": 2205, "sensors": selected,
        "split_seed": seed, "split_block_size": block_size,
    }
    for name, value in (("dataset_manifest.json", manifest), ("device_card.json", card), ("source_binding.json", binding)):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return binding
