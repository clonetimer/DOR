"""EHSS diagnosis-dataset -> ISGN PublicDataset bridge.

The bridge deliberately separates the EHSS *observation contract* from an ISGN
*discovery view*.  The source dataset contains asynchronous/held sensor chains
plus simulator truth.  A bridge view states exactly which public measurements
are promoted to candidate states and which exogenous task variables are passed
as inputs/context.

Simulator ``truth_*`` columns are never used by the default bridge.  The
``truth-audit`` view exists only for simulation-side observability audits and is
explicitly marked as non-production evidence in the returned contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import csv
import hashlib
import json

import numpy as np

from ..io import from_arrays, save_npz
from ..contracts import PublicDataset

EHSS_PROTOCOL_VERSION = "ehss-diagnosis-dataset/1.3"
EHSS_CONTRACT_VERSION = "isgn-ehss-observation-contract/1.0"

# EHSS public measurement projection.  ``column`` is the value column in
# timeseries.csv; metadata_sensor is the key used by metadata.json.
_MEASUREMENTS: Dict[str, Dict[str, str]] = {
    "position": {"column": "measurement_position_m", "truth": "truth_position_m", "dim": "displacement", "unit": "m"},
    "velocity": {"column": "measurement_velocity_m_s", "truth": "truth_velocity_m_s", "dim": "velocity", "unit": "m/s"},
    "pressure_a": {"column": "measurement_pressure_a_pa", "truth": "truth_pressure_a_pa", "dim": "pressure", "unit": "Pa"},
    "pressure_b": {"column": "measurement_pressure_b_pa", "truth": "truth_pressure_b_pa", "dim": "pressure", "unit": "Pa"},
    "temperature_a": {"column": "measurement_temperature_a_k", "truth": "truth_temperature_a_k", "dim": "temperature", "unit": "K"},
    "temperature_b": {"column": "measurement_temperature_b_k", "truth": "truth_temperature_b_k", "dim": "temperature", "unit": "K"},
    "supply_pressure": {"column": "measurement_supply_pressure_pa", "truth": "truth_supply_pressure_pa", "dim": "pressure", "unit": "Pa"},
    "supply_flow": {"column": "measurement_supply_flow_m3_s", "truth": "truth_supply_flow_m3_s", "dim": "volume_flow", "unit": "m^3/s"},
    "valve_spool": {"column": "measurement_valve_spool_m", "truth": "truth_valve_spool_m", "dim": "displacement", "unit": "m"},
    "valve_current": {"column": "measurement_valve_current_a", "truth": "truth_valve_current_a", "dim": "current", "unit": "A"},
    "pump_speed": {"column": "measurement_pump_speed_rad_s", "truth": "truth_pump_speed_rad_s", "dim": "speed", "unit": "rad/s"},
}

# Exogenous NetworkInput variables.  Only command_m is required to satisfy the
# persistent-excitation test; the remaining quantities are declared as context
# because a single task may vary them slowly or only once.
_INPUTS: Tuple[Dict[str, str], ...] = (
    {"name": "command", "column": "command_m", "dim": "displacement", "unit": "m", "role": "input"},
    {"name": "supply_pressure_target", "column": "supply_pressure_pa", "dim": "pressure", "unit": "Pa", "role": "context"},
    {"name": "reservoir_temperature", "column": "reservoir_temperature_k", "dim": "temperature", "unit": "K", "role": "context"},
    {"name": "external_load", "column": "external_load_n", "dim": "force", "unit": "N", "role": "context"},
)

_VIEWS: Dict[str, Tuple[str, ...]] = {
    # Complete public sensor projection. Temperature channels are 100 Hz in the
    # current diagnosis profile and are sample-held on the 500 Hz export grid.
    "measured-full": tuple(_MEASUREMENTS.keys()),
    # Directly aligned public sensors whose declared sample period is <= the
    # 2 ms dataset export interval in the current P1 dataset.  Runtime validation
    # re-checks this property instead of trusting the name.
    "measured-fast": (
        "position", "velocity", "pressure_a", "pressure_b", "supply_pressure",
        "supply_flow", "valve_spool", "valve_current", "pump_speed",
    ),
    # Smaller actuator/hydraulic view useful for scoped discovery.  It is not
    # claimed to be the full EHSS state graph.
    "actuator-core": (
        "position", "velocity", "pressure_a", "pressure_b", "valve_spool", "valve_current",
    ),
    # Same observable projection but sourced from simulator truth.  This view is
    # forbidden as the default and is labelled simulation_audit_only.
    "truth-audit": tuple(_MEASUREMENTS.keys()),
}


@dataclass
class EHSSBridgeResult:
    dataset: PublicDataset
    contract: Dict[str, Any]


def available_ehss_views() -> Tuple[str, ...]:
    return tuple(_VIEWS.keys())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _trajectory_id(meta: Mapping[str, Any]) -> int:
    key = "|".join(str(meta.get(k, "")) for k in ("scenario_id", "split", "experiment_id", "task_id", "seed"))
    raw = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(raw, "big", signed=False) & ((1 << 63) - 1)


def _read_scenario(path: str | Path) -> Tuple[Path, Dict[str, Any], List[Dict[str, str]], List[str]]:
    p = Path(path)
    if p.is_file():
        raise ValueError(f"EHSS scenario must be a directory, got file: {p}")
    meta_path = p / "metadata.json"
    csv_path = p / "timeseries.csv"
    if not meta_path.is_file() or not csv_path.is_file():
        raise FileNotFoundError(f"expected metadata.json and timeseries.csv under {p}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("schema_version") != EHSS_PROTOCOL_VERSION:
        raise ValueError(f"unsupported EHSS schema_version={meta.get('schema_version')!r}; expected {EHSS_PROTOCOL_VERSION!r}")
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if len(rows) < 2:
        raise ValueError("EHSS scenario must contain at least two time rows")
    return p, meta, rows, fields


def _sensor_metadata(meta: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in meta.get("sensors", []):
        ch = str(item.get("channel", ""))
        if ch:
            out[ch] = dict(item)
    return out


def _float_column(rows: Sequence[Mapping[str, str]], column: str) -> np.ndarray:
    try:
        a = np.asarray([float(r[column]) for r in rows], dtype=np.float64)
    except KeyError as e:
        raise ValueError(f"missing EHSS column: {column}") from e
    except ValueError as e:
        raise ValueError(f"non-numeric EHSS column: {column}") from e
    if not np.isfinite(a).all():
        raise ValueError(f"NaN/Inf detected in EHSS column: {column}")
    return a


def _validate_time(rows: Sequence[Mapping[str, str]]) -> Tuple[np.ndarray, float, float]:
    t = _float_column(rows, "time_s")
    dt = np.diff(t)
    if not np.all(dt > 0):
        raise ValueError("EHSS time_s must be strictly increasing")
    nominal = float(np.median(dt))
    max_err = float(np.max(np.abs(dt - nominal)))
    # EHSS writer timestamps can jitter by ~10 ns at a 1 ms export grid due to
    # floating output-time accumulation.  Accept 20 ppm here; this is a file-grid
    # parsing tolerance, not an Observation Gate physics/identifiability threshold.
    tol = max(1e-9, nominal * 2e-5)
    if max_err > tol:
        raise ValueError(f"EHSS export grid is not uniform enough: dt={nominal:.12g}, max_error={max_err:.3g}, tolerance={tol:.3g}")
    return t, nominal, max_err


def _validate_identity(meta: Mapping[str, Any], rows: Sequence[Mapping[str, str]]) -> None:
    checks = {
        "schema_version": meta.get("schema_version"),
        "scenario_id": meta.get("scenario_id"),
        "split": meta.get("split"),
        "experiment_id": meta.get("experiment_id"),
        "task_id": meta.get("task_id"),
        "seed": str(meta.get("seed")),
    }
    for col, expected in checks.items():
        vals = {r.get(col, "") for r in rows}
        if vals != {str(expected)}:
            raise ValueError(f"EHSS row metadata mismatch for {col}: expected {expected!r}, found {sorted(vals)!r}")


def _validate_fault_policy(meta: Mapping[str, Any], rows: Sequence[Mapping[str, str]], allow_faulted: bool) -> None:
    faults = list(meta.get("faults", []))
    active = max(int(float(r.get("active_fault_count", "0") or 0)) for r in rows)
    if (faults or active > 0) and not allow_faulted:
        raise ValueError("faulted EHSS scenarios are blocked by default for normal-mechanism discovery; pass allow_faulted=True only for explicit PHM/fault analysis")


def _measurement_state_schema(name: str, sensor_meta: Mapping[str, Dict[str, Any]]) -> Dict[str, Any]:
    spec = _MEASUREMENTS[name]
    sensor = sensor_meta.get(name)
    if sensor is None:
        raise ValueError(f"metadata.json does not declare sensor channel {name!r}")
    return {
        "name": name,
        "dim": spec["dim"],
        "unit": spec["unit"],
        "source_column": spec["column"],
        "observation_role": "measurement",
        "sensor": sensor,
    }


def _truth_state_schema(name: str) -> Dict[str, Any]:
    spec = _MEASUREMENTS[name]
    return {
        "name": name,
        "dim": spec["dim"],
        "unit": spec["unit"],
        "source_column": spec["truth"],
        "observation_role": "truth",
        "mechanism_use": "simulation_audit_only",
    }


def _input_schema() -> List[Dict[str, Any]]:
    return [dict(x) for x in _INPUTS]


def _build_single(path: str | Path, *, view: str, allow_faulted: bool) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any], int, float]:
    if view not in _VIEWS:
        raise ValueError(f"unknown EHSS view {view!r}; choose one of {', '.join(available_ehss_views())}")
    p, meta, rows, fields = _read_scenario(path)
    _validate_identity(meta, rows)
    _validate_fault_policy(meta, rows, allow_faulted)
    t, dt, dt_err = _validate_time(rows)
    sensor_meta = _sensor_metadata(meta)

    state_names = _VIEWS[view]
    truth = view == "truth-audit"
    state_items: List[Dict[str, Any]] = []
    state_arrays: List[np.ndarray] = []
    for name in state_names:
        spec = _MEASUREMENTS[name]
        if truth:
            col = spec["truth"]
            state_items.append(_truth_state_schema(name))
        else:
            col = spec["column"]
            valid_col = col + "_valid"
            updated_col = col + "_updated"
            for required in (col, valid_col, updated_col):
                if required not in fields:
                    raise ValueError(f"missing EHSS measurement contract column: {required}")
            valid = _float_column(rows, valid_col)
            if not np.all(valid == 1.0):
                bad = int(np.sum(valid != 1.0))
                raise ValueError(f"selected measurement {name!r} has {bad} invalid rows; bridge does not silently impute invalid sensor data")
            state_items.append(_measurement_state_schema(name, sensor_meta))
        state_arrays.append(_float_column(rows, col))

    if view == "measured-fast":
        # This view's invariant is objective: every selected sensor must update at
        # least as fast as the exported trajectory grid according to metadata.
        slower = [name for name in state_names if float(sensor_meta[name].get("sample_period_s", np.inf)) > dt * (1.0 + 1e-6)]
        if slower:
            raise ValueError(f"measured-fast invariant violated; slower-than-export sensors: {slower}")

    input_items = _input_schema()
    input_arrays = [_float_column(rows, x["column"]) for x in input_items]

    X = np.column_stack(state_arrays)
    U_full = np.column_stack(input_arrays)
    U = U_full[:-1]
    tid = _trajectory_id(meta)
    source = {
        "scenario_dir": str(p),
        "metadata_sha256": _sha256(p / "metadata.json"),
        "timeseries_sha256": _sha256(p / "timeseries.csv"),
        "schema_version": meta.get("schema_version"),
        "model_version": meta.get("model_version"),
        "parameter_profile": meta.get("parameter_profile"),
        "scenario_id": meta.get("scenario_id"),
        "split": meta.get("split"),
        "experiment_id": meta.get("experiment_id"),
        "task_id": meta.get("task_id"),
        "seed": meta.get("seed"),
        "fault_count_declared": len(meta.get("faults", [])),
        "sample_count": len(rows),
        "time_start_s": float(t[0]),
        "time_end_s": float(t[-1]),
        "export_dt_s": dt,
        "max_dt_error_s": dt_err,
        "trajectory_id": tid,
    }
    schema = {
        "states": state_items,
        "inputs": input_items,
        "bridge": {
            "name": "ehss",
            "contract_version": EHSS_CONTRACT_VERSION,
            "source_protocol": EHSS_PROTOCOL_VERSION,
            "view": view,
            "source_class": "simulator_truth" if truth else "public_measurement",
            "mechanism_use": "simulation_audit_only" if truth else "candidate_observation",
        },
    }
    return X, U, schema, source, tid, dt


def load_ehss_scenarios(paths: Iterable[str | Path], *, view: str = "measured-fast", allow_faulted: bool = False) -> EHSSBridgeResult:
    paths = list(paths)
    if not paths:
        raise ValueError("at least one EHSS scenario directory is required")
    built = [_build_single(p, view=view, allow_faulted=allow_faulted) for p in paths]
    first_X, first_U, schema, _, _, first_dt = built[0]
    for X, U, s, source, tid, dt in built[1:]:
        if X.shape != first_X.shape or U.shape != first_U.shape:
            raise ValueError("all EHSS trajectories in one PublicDataset must share horizon and view dimensions")
        if abs(dt - first_dt) > max(1e-9, first_dt * 2e-5):
            raise ValueError("all EHSS trajectories in one PublicDataset must share a fixed dt")
        if s != schema:
            # Sensor metadata can differ by scenario.  Public schema must be common
            # for ISGN; differing observation chains should not be silently pooled.
            raise ValueError("EHSS scenarios have different observation schema/sensor metadata and cannot be pooled into one PublicDataset")

    X_all = np.stack([x[0] for x in built], axis=0)
    U_all = np.stack([x[1] for x in built], axis=0)
    tids = np.asarray([x[4] for x in built], dtype=np.int64)
    if len(np.unique(tids)) != len(tids):
        raise ValueError("trajectory_id collision detected")
    ds = from_arrays(
        X_all,
        U_all,
        schema=schema,
        dt=first_dt,
        trajectory_id=tids,
        regime="ehss",
        variant=f"ehss-{view}",
    )

    selected = list(_VIEWS[view])
    omitted = [x for x in _MEASUREMENTS if x not in selected]
    source_records = [x[3] for x in built]
    contract = {
        "contract_version": EHSS_CONTRACT_VERSION,
        "source_protocol": EHSS_PROTOCOL_VERSION,
        "bridge_view": view,
        "production_mechanism_eligible": view != "truth-audit",
        "truth_policy": "truth_* is excluded from production discovery; truth-audit is simulation-side evidence only",
        "public_dataset": {
            "X_shape": list(ds.X.shape),
            "U_shape": list(ds.U.shape),
            "dt_s": float(ds.dt),
            "trajectory_id": [int(x) for x in ds.trajectory_id],
            "regime": ds.regime,
            "variant": ds.variant,
        },
        "selected_state_channels": selected,
        "omitted_measurement_channels": omitted,
        "inputs": [dict(x) for x in _INPUTS],
        "schema": schema,
        "sources": source_records,
        "constraints": {
            "fixed_dt_required": True,
            "invalid_measurements_imputed": False,
            "faulted_scenarios_allowed": bool(allow_faulted),
            "sensor_delay_compensated": False,
            "sensor_sample_hold_removed": False,
            "state_estimation_applied": False,
        },
    }
    return EHSSBridgeResult(ds, contract)


def load_ehss_scenario(path: str | Path, *, view: str = "measured-fast", allow_faulted: bool = False) -> EHSSBridgeResult:
    return load_ehss_scenarios([path], view=view, allow_faulted=allow_faulted)


def save_ehss_bridge(result: EHSSBridgeResult, npz_path: str | Path, contract_path: str | Path) -> None:
    npz_path = Path(npz_path)
    contract_path = Path(contract_path)
    save_npz(result.dataset, npz_path)
    contract_path.write_text(json.dumps(result.contract, indent=2, ensure_ascii=False), encoding="utf-8")
