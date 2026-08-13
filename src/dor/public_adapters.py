from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


UCI_HYDRAULIC_URL = "https://archive.ics.uci.edu/static/public/447/condition%2Bmonitoring%2Bof%2Bhydraulic%2Bsystems.zip"
UCI_SENSORS = {
    "PS1": (100, "bar"), "PS2": (100, "bar"), "PS3": (100, "bar"),
    "PS4": (100, "bar"), "PS5": (100, "bar"), "PS6": (100, "bar"),
    "EPS1": (100, "W"), "FS1": (10, "l/min"), "FS2": (10, "l/min"),
    "TS1": (1, "degC"), "TS2": (1, "degC"), "TS3": (1, "degC"),
    "TS4": (1, "degC"), "VS1": (1, "mm/s"), "CE": (1, "%"),
    "CP": (1, "kW"), "SE": (1, "%"),
}

SPACECRAFT_SOURCE_SCHEMAS = {
    "sat-sim.whole-spacecraft-adcs-paired-dataset.v5.2.2": "ADCS_V5_2_2",
    "sat-sim.whole-spacecraft-phm-dataset.v6.0": "PHM_V6",
}
SPACECRAFT_TIME_COLUMNS = {"time_s", "packet_receive_time_s", "packet_sequence"}
SPACECRAFT_NON_RAW_TOKENS = (
    "valid", "status", "safe", "residual", "innovation", "invalid_count",
    "solution_fresh", "cell_healthy", "saturated",
)


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


def _spacecraft_schema(root: Path, manifest: dict, index: pd.DataFrame) -> str:
    schema = str(manifest.get("schema_version") or manifest.get("schema") or "")
    if schema in SPACECRAFT_SOURCE_SCHEMAS:
        return SPACECRAFT_SOURCE_SCHEMAS[schema]
    if {"subsystem", "fault_injection_phase"}.issubset(index.columns):
        return "PHM_V6"
    if {"operating_phase_at_fault", "scenario_id"}.issubset(index.columns):
        return "ADCS_V5_2_2"
    raise ValueError(f"unsupported spacecraft dataset schema at {root}")


def _safe_source_file(root: Path, relative: object) -> Path:
    value = str(relative)
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"source path escapes dataset root: {value!r}") from exc
    if not candidate.is_file():
        raise ValueError(f"source file is missing: {value!r}")
    return candidate


def _spacecraft_channel_allowed(name: str, profile: str) -> bool:
    lower = name.lower()
    if lower in SPACECRAFT_TIME_COLUMNS:
        return False
    if profile == "deployable_full":
        return True
    return not any(token in lower for token in SPACECRAFT_NON_RAW_TOKENS)


def _compact_series_features(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    if x.size < 2:
        raise ValueError("spacecraft telemetry channels require at least two samples")
    finite = np.isfinite(x)
    observed = x[finite]
    result = {"observed_fraction": float(np.mean(finite))}
    if observed.size == 0:
        result.update({name: 0.0 for name in ("mean", "std", "min", "max", "rms", "slope", "delta", "temporal__diff_std")})
        return result
    t = np.flatnonzero(finite).astype(float)
    diff = np.diff(observed)
    slope = float(np.polyfit(t, observed, 1)[0]) if observed.size >= 2 else 0.0
    diff_std = float(np.std(diff)) if diff.size else 0.0
    result.update({
        "mean": float(np.mean(observed)),
        "std": float(np.std(observed)),
        "min": float(np.min(observed)),
        "max": float(np.max(observed)),
        "rms": float(np.sqrt(np.mean(np.square(observed)))),
        "slope": slope,
        "delta": float(observed[-1] - observed[0]),
        "temporal__diff_std": diff_std,
    })
    return result


def _split_roles(index: pd.DataFrame, strata_column: str) -> dict[str, str]:
    """Assign complete trajectory groups, with each stratum represented per role."""
    group_col = "trajectory_pair_id"
    if group_col not in index or index[group_col].isna().any():
        raise ValueError("spacecraft sample index requires complete trajectory_pair_id values")
    group_strata = index[[group_col, strata_column]].drop_duplicates()
    if group_strata[group_col].duplicated().any():
        raise ValueError("one trajectory_pair_id maps to multiple split strata")
    assignment: dict[str, str] = {}
    role_cycle = ("train", "train", "train", "train", "train", "train", "validation", "validation", "calibration", "test")
    for _, part in group_strata.groupby(strata_column, sort=True, dropna=False):
        groups = sorted(part[group_col].astype(str))
        if len(groups) < 4:
            raise ValueError(f"split stratum {part.iloc[0][strata_column]!r} has fewer than four trajectory groups")
        for position, group in enumerate(groups):
            # Ten Monte-Carlo groups are canonical; smaller/larger fixtures retain all four roles.
            if len(groups) == 10:
                role = role_cycle[position]
            else:
                fraction = position / len(groups)
                role = "train" if fraction < .6 else "validation" if fraction < .8 else "calibration" if fraction < .9 else "test"
            assignment[group] = role
    return assignment


def _adcs_component(label: str) -> str:
    value = str(label)
    if value == "nominal":
        return "healthy"
    if "rw_" in value:
        return "reaction_wheel_assembly"
    if "star_" in value:
        return "star_tracker"
    if "sun_" in value:
        return "sun_sensor"
    if "gyro" in value or "imu" in value:
        return "imu"
    if "control_loop" in value:
        return "adcs_controller"
    return "adcs_unknown"


def prepare_spacecraft_dataset(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    feature_profile: str = "causal_raw",
    max_samples: int | None = None,
) -> dict:
    """Stream a sat-sim spacecraft file tree into a leakage-bounded DOR table.

    The source tree is only read. Injected parameters, event times, paths and
    per-timestep label files are deliberately excluded from model features.
    """
    if feature_profile not in {"causal_raw", "deployable_full"}:
        raise ValueError("feature_profile must be 'causal_raw' or 'deployable_full'")
    root = Path(dataset_root).resolve()
    index_path = root / "sample_index.csv"
    source_manifest_path = root / "dataset_manifest.json"
    if not root.is_dir() or not index_path.is_file() or not source_manifest_path.is_file():
        raise ValueError("spacecraft dataset root requires sample_index.csv and dataset_manifest.json")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    index = pd.read_csv(index_path)
    required = {"sample_id", "trajectory_pair_id", "sample_index", "mission_id", "condition_id", "diagnostic_class", "telemetry_file", "row_count", "field_count"}
    missing = sorted(required - set(index.columns))
    if missing:
        raise ValueError(f"spacecraft sample index missing columns: {missing}")
    if index["sample_id"].astype(str).duplicated().any():
        raise ValueError("spacecraft sample_id values must be unique")
    if max_samples is not None:
        if max_samples < 4:
            raise ValueError("max_samples must be at least four")
        index = index.sort_values("sample_index").head(max_samples).copy()
    schema = _spacecraft_schema(root, source_manifest, index)
    stratum = "subsystem" if schema == "PHM_V6" else "mission_id"
    split_roles = _split_roles(index, stratum)

    first_path = _safe_source_file(root, index.iloc[0]["telemetry_file"])
    first = pd.read_csv(first_path)
    numeric = [c for c in first.columns if pd.api.types.is_numeric_dtype(first[c])]
    channels = [c for c in numeric if _spacecraft_channel_allowed(str(c), feature_profile)]
    if not channels:
        raise ValueError("no eligible numeric spacecraft telemetry channels")

    rows: list[dict[str, object]] = []
    for record in index.to_dict("records"):
        telemetry_path = _safe_source_file(root, record["telemetry_file"])
        telemetry = pd.read_csv(telemetry_path, usecols=channels)
        if len(telemetry) != int(record["row_count"]):
            raise ValueError(f"row count mismatch for {record['sample_id']!r}")
        row: dict[str, object] = {
            "sample_id": str(record["sample_id"]),
            "trajectory_pair_id": str(record["trajectory_pair_id"]),
            "split": split_roles[str(record["trajectory_pair_id"])],
            "mission_id": str(record["mission_id"]),
            "fault_state": "NORMAL" if str(record["condition_id"]) == "nominal" else "ABNORMAL",
            "diagnostic_class": str(record["diagnostic_class"]),
            "condition_id": str(record["condition_id"]),
        }
        if schema == "PHM_V6":
            row["subsystem"] = str(record["subsystem"])
            row["localization_label"] = "healthy" if row["fault_state"] == "NORMAL" else str(record["subsystem"])
            row["fault_injection_phase"] = str(record["fault_injection_phase"])
        else:
            row["scenario_id"] = str(record["scenario_id"])
            row["operating_phase_at_fault"] = str(record["operating_phase_at_fault"])
            row["localization_label"] = _adcs_component(str(record["diagnostic_class"]))
        for channel in channels:
            values = telemetry[channel].to_numpy(dtype=float)
            for stat, value in _compact_series_features(values).items():
                row[f"{channel}__{stat}"] = value
        rows.append(row)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_path = out / "spacecraft_features.csv.gz"
    frame = pd.DataFrame(rows)
    frame.to_csv(data_path, index=False, compression="gzip")
    feature_columns = [c for c in frame.columns if "__" in c]
    slices = [stratum, "fault_injection_phase"] if schema == "PHM_V6" else ["mission_id", "scenario_id", "operating_phase_at_fault"]
    manifest = {
        "contract": "DOR-MULTIMODAL-DATASET-MANIFEST/1.0",
        "dataset_id": f"{root.name}-{feature_profile}",
        "modality": "TABULAR",
        "identifiers": ["sample_id"],
        "features": {"raw_columns": feature_columns},
        "split": {"column": "split", "train_values": ["train"], "validation_values": ["validation"], "calibration_values": ["calibration"], "test_values": ["test"]},
        "supervision": {
            "detection": {"label_column": "fault_state", "normal_label": "NORMAL"},
            "localization": {"label_column": "localization_label"},
            "classification": {"label_column": "diagnostic_class"},
        },
        "evaluation": {"slice_columns": slices},
        "leakage_guard": {"group_columns": ["trajectory_pair_id"], "excluded_source_columns": ["event_start_s", "activation_time_s", "fault_manifestation_time_s", "event_parameters", "spacecraft_parameters", "labels_file"]},
    }
    components = sorted(set(frame["localization_label"].astype(str)) - {"healthy"})
    card = {
        "contract": "DOR-DEVICE-MECHANISM-CARD/1.0",
        "device_id": f"synthetic-spacecraft-{schema.lower()}",
        "domain": "synthetic spacecraft fault diagnosis",
        "sensors": [{"sensor_id": c.split(".", 1)[0] if "." in c else c} for c in sorted({c.split(".", 1)[0] if "." in c else c for c in channels})],
        "components": [{"component_id": c} for c in components],
        "topology_edges": [],
        "mechanism_features": [],
    }
    binding = {
        "contract": "DOR-SPACECRAFT-SOURCE-BINDING/1.0",
        "source_schema": schema,
        "source_root": str(root),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "source_index_sha256": _sha256(index_path),
        "prepared_data_sha256": _sha256(data_path),
        "feature_profile": feature_profile,
        "sample_count": len(frame),
        "channel_count": len(channels),
        "feature_count": len(feature_columns),
        "role_counts": frame["split"].value_counts().sort_index().to_dict(),
        "counterfactual_policy": "paired healthy trajectories are excluded from inputs and reserved as an offline upper-bound reference",
        "claim_boundary": "synthetic simulator evidence; no flight or hardware accuracy claim",
    }
    for name, value in (("dataset_manifest.json", manifest), ("device_card.json", card), ("source_binding.json", binding)):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return binding
