#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M3 real-data contract for ISGN-Real v0.5.

The M3 contract is deliberately independent of the training code and of any
particular download provider.  The scientific split unit is an *experiment / trial*,
never a sliding window or an individual row.

Primary source adapter implemented here:
  dos Reis et al. (2021/2022) DC motor voltage/current/speed workbook.

A small PMSM profile adapter is also included only for parser smoke tests; one
profile is not sufficient to pass M3.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import json
import math
import re
import zipfile
import gzip
import pickle
from collections import defaultdict

import numpy as np


class M3DataError(ValueError):
    pass


def _norm_name(x: Any) -> str:
    s = str(x).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", s)


DC_ALIASES: Dict[str, Tuple[str, ...]] = {
    "time": ("time", "timestamp", "instant", "measurementtime", "timeus"),
    "encoder_count": ("encodercount", "encoder", "count"),
    "speed": ("velocity", "speed", "motorspeed", "rpm"),
    "raw_current": ("rawcurrent",),
    "current": ("current", "motorcurrent", "calculatedcurrent"),
    "raw_voltage_a1": ("rawvoltagea1",),
    "raw_voltage_b1": ("rawvoltageb1",),
    "voltage_a1": ("voltagea1",),
    "voltage_b1": ("voltageb1",),
    "motor_voltage": ("motorvoltage", "voltagemotor", "appliedvoltage", "vmotor"),
    "motor_status": ("motorstatus", "status"),
    "pwm": ("pwm", "pwmcommand", "commandpwm"),
}

PMSM_REQUIRED = (
    "ambient", "coolant", "u_d", "u_q", "motor_speed", "torque",
    "i_d", "i_q", "pm", "stator_yoke", "stator_tooth", "stator_winding", "profile_id",
)


@dataclass
class RealTrajectory:
    source: str
    trajectory_id: str
    motor_id: str
    excitation_family: str
    load_level: Optional[float]
    dt: float
    time_s: np.ndarray
    X: np.ndarray              # [T, state_dim]
    U: np.ndarray              # [T-1, input_dim]
    state_names: Tuple[str, ...]
    input_names: Tuple[str, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self, min_samples: int = 20) -> Dict[str, Any]:
        errors: List[str] = []
        warnings: List[str] = []
        t = np.asarray(self.time_s, dtype=float)
        X = np.asarray(self.X, dtype=float)
        U = np.asarray(self.U, dtype=float)
        if t.ndim != 1:
            errors.append("time_s must be 1-D")
        if X.ndim != 2:
            errors.append("X must be 2-D")
        if U.ndim != 2:
            errors.append("U must be 2-D")
        if X.ndim == 2 and len(t) != len(X):
            errors.append("len(time_s) must equal X.shape[0]")
        if X.ndim == 2 and U.ndim == 2 and len(U) not in (len(X), max(len(X)-1, 0)):
            errors.append("U must have T or T-1 rows")
        if len(t) < min_samples:
            errors.append(f"trajectory shorter than {min_samples} samples")
        if not np.isfinite(t).all() or not np.isfinite(X).all() or not np.isfinite(U).all():
            errors.append("NaN/Inf detected")
        if len(t) > 1:
            d = np.diff(t)
            if not np.all(d > 0):
                errors.append("time is not strictly monotonic")
            med = float(np.median(d))
            jitter = float(np.std(d) / max(abs(med), 1e-12))
            if abs(med - self.dt) > max(5e-4, 0.05 * abs(self.dt)):
                warnings.append(f"stored dt={self.dt:g} differs from median dt={med:g}")
            if jitter > 0.10:
                warnings.append(f"sampling jitter ratio is high ({jitter:.3f})")
        input_std = np.std(U, axis=0) if U.size else np.zeros((len(self.input_names),))
        if U.size and float(np.max(input_std)) < 1e-8:
            warnings.append("input has essentially no within-trajectory variation")
        return {
            "trajectory_id": self.trajectory_id,
            "ok": not errors,
            "errors": errors,
            "warnings": warnings,
            "n_samples": int(len(t)),
            "duration_s": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
            "dt": float(self.dt),
            "input_std": [float(v) for v in np.atleast_1d(input_std)],
            "state_min": [float(v) for v in np.nanmin(X, axis=0)] if X.size else [],
            "state_max": [float(v) for v in np.nanmax(X, axis=0)] if X.size else [],
            "input_min": [float(v) for v in np.nanmin(U, axis=0)] if U.size else [],
            "input_max": [float(v) for v in np.nanmax(U, axis=0)] if U.size else [],
        }

    def summary(self) -> Dict[str, Any]:
        d = asdict(self)
        for k in ("time_s", "X", "U"):
            d.pop(k, None)
        d["n_samples"] = int(len(self.time_s))
        d["duration_s"] = float(self.time_s[-1]-self.time_s[0]) if len(self.time_s)>1 else 0.0
        return d


def _resolve_columns(df: pd.DataFrame, aliases: Dict[str, Sequence[str]], required: Sequence[str]) -> Dict[str, str]:
    normalized = {_norm_name(c): str(c) for c in df.columns}
    out: Dict[str, str] = {}
    for canonical, names in aliases.items():
        for a in names:
            if _norm_name(a) in normalized:
                out[canonical] = normalized[_norm_name(a)]
                break
    missing = [k for k in required if k not in out]
    if missing:
        raise M3DataError(f"missing required columns {missing}; available={list(df.columns)}")
    return out


def parse_dc_sheet_metadata(sheet_name: str) -> Dict[str, Any]:
    """Parse experiment semantics from a published workbook tab name.

    This parser is conservative: unknown names remain `unknown` rather than being guessed.
    """
    s = str(sheet_name).strip()
    low = s.lower().replace(" ", "")
    motor = "A" if "motora" in low else ("B" if "motorb" in low else "unknown")
    family = "unknown"
    freq: Optional[float] = None
    load_level: Optional[float] = None
    if "prbs7" in low:
        family = "prbs7"
    elif "prbs9" in low:
        family = "prbs9"
    elif "sine" in low:
        family = "sine"
    elif "triangle" in low:
        family = "triangle"
    elif "square" in low:
        family = "square"
        freq = 0.25
    elif "stepnoload" in low:
        family = "step_noload"
        load_level = 0.0
    elif "steploaded" in low:
        family = "step_loaded"
    m = re.search(r"(0\.1|10|1)hz", low)
    if m:
        freq = float(m.group(1))
    return {
        "sheet_name": s,
        "motor_id": motor,
        "excitation_family": family,
        "frequency_hz": freq,
        "load_level": load_level,
        "loaded": family == "step_loaded",
    }


def _split_on_time_resets(df, time_col: str, min_samples: int = 20):
    import pandas as pd
    """Split a tab into trials using blank rows and/or time resets.

    The published source records time in microseconds and repeats an excitation multiple times.
    We intentionally do not use fixed trial counts to avoid hard-coding spreadsheet layout.
    """
    work = df.copy()
    # Preserve blank-row boundaries before dropping NaNs.
    numeric_time = pd.to_numeric(work[time_col], errors="coerce")
    boundaries = [0]
    last_valid: Optional[float] = None
    for i, v in enumerate(numeric_time.to_numpy()):
        if not np.isfinite(v):
            if boundaries[-1] != i:
                boundaries.append(i)
            last_valid = None
            continue
        if last_valid is not None and v <= last_valid:
            if boundaries[-1] != i:
                boundaries.append(i)
        last_valid = float(v)
    boundaries.append(len(work))
    boundaries = sorted(set(boundaries))
    chunks: List[pd.DataFrame] = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        part = work.iloc[a:b].dropna(how="all")
        if len(part) >= min_samples:
            chunks.append(part.reset_index(drop=True))
    # If no resets were found but table is long, treat the sheet itself as one trajectory.
    if not chunks:
        part = work.dropna(how="all")
        if len(part) >= min_samples:
            chunks = [part.reset_index(drop=True)]
    return chunks


def _dc_chunk_to_trajectory(
    df,
    meta: Dict[str, Any],
    trial_index: int,
    source_name: str,
    min_samples: int = 20,
) -> RealTrajectory:
    import pandas as pd
    cols = _resolve_columns(df, DC_ALIASES, required=("time", "current", "speed", "motor_voltage"))
    t_raw = pd.to_numeric(df[cols["time"]], errors="coerce").to_numpy(float)
    current = pd.to_numeric(df[cols["current"]], errors="coerce").to_numpy(float)
    speed = pd.to_numeric(df[cols["speed"]], errors="coerce").to_numpy(float)
    voltage = pd.to_numeric(df[cols["motor_voltage"]], errors="coerce").to_numpy(float)
    valid = np.isfinite(t_raw) & np.isfinite(current) & np.isfinite(speed) & np.isfinite(voltage)
    t_raw, current, speed, voltage = t_raw[valid], current[valid], speed[valid], voltage[valid]
    if len(t_raw) < min_samples:
        raise M3DataError(f"trial {trial_index} too short after filtering ({len(t_raw)})")
    # Source definition records Arduino measurement time in microseconds at 100 Hz.
    t = (t_raw - t_raw[0]) * 1e-6
    d = np.diff(t)
    if len(d) == 0 or not np.all(d > 0):
        raise M3DataError("time is not strictly increasing after split")
    dt = float(np.median(d))
    # X stores measured physical states at sample times; U[k] drives X[k] -> X[k+1].
    X = np.column_stack([current, speed]).astype(np.float64)
    U = voltage[:-1, None].astype(np.float64)
    metadata = dict(meta)
    metadata.update({
        "trial_index": int(trial_index),
        "source_file": source_name,
        "pwm_present": "pwm" in cols,
        "motor_status_present": "motor_status" in cols,
    })
    if "pwm" in cols:
        pwm = pd.to_numeric(df.loc[valid, cols["pwm"]], errors="coerce").to_numpy(float)
        metadata["pwm_min"] = float(np.nanmin(pwm))
        metadata["pwm_max"] = float(np.nanmax(pwm))
    if "motor_status" in cols:
        status = pd.to_numeric(df.loc[valid, cols["motor_status"]], errors="coerce").to_numpy(float)
        metadata["motor_status_values"] = sorted(float(v) for v in np.unique(status[np.isfinite(status)]))
    tid = f"{meta['sheet_name']}::trial{trial_index:03d}"
    return RealTrajectory(
        source="dc_motor_2021",
        trajectory_id=tid,
        motor_id=str(meta["motor_id"]),
        excitation_family=str(meta["excitation_family"]),
        load_level=meta.get("load_level"),
        dt=dt,
        time_s=t,
        X=X,
        U=U,
        state_names=("current", "speed"),
        input_names=("motor_voltage",),
        metadata=metadata,
    )


def _xlsx_col_index(ref: str) -> int:
    m = re.match(r"([A-Z]+)", str(ref))
    if not m:
        return 0
    n = 0
    for ch in m.group(1):
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def _xlsx_col_name(index: int) -> str:
    n = int(index) + 1
    out = ""
    while n:
        n, r = divmod(n - 1, 26)
        out = chr(65 + r) + out
    return out


def _load_dc_motor_xlsx_stream(path: Path, min_samples: int = 20) -> List[RealTrajectory]:
    """Fast, layout-aware reader for the published DC-motor workbook.

    The source workbook uses two layouts simultaneously:
      1) repeated trials stacked vertically in a 12-column block;
      2) Step sheets with many 12-column blocks placed horizontally every 13 columns,
         with additional trials stacked vertically inside each block.

    Each cell containing the literal ``time`` starts an independent trial block.  This avoids
    hard-coding trial counts and prevents the old A:L-only parser from silently dropping most
    step/loaded trials.
    """
    try:
        from lxml import etree as LET  # fast streaming; explicit M3 parser dependency
    except ImportError as e:
        raise M3DataError("lxml is required for fast MotorsData_rev.xlsx streaming") from e

    MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    RID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    PKG = "{http://schemas.openxmlformats.org/package/2006/relationships}"

    cache = path.with_suffix(path.suffix + ".m3cache.pkl.gz")
    source_sig = (int(path.stat().st_size), int(path.stat().st_mtime_ns))
    if cache.exists():
        try:
            with gzip.open(cache, "rb") as f:
                payload = pickle.load(f)
            if payload.get("source_sig") == source_sig:
                return payload["trajectories"]
        except Exception:
            pass

    out: List[RealTrajectory] = []
    errors: List[str] = []

    with zipfile.ZipFile(path, "r") as z:
        ss_root = LET.fromstring(z.read("xl/sharedStrings.xml"))
        shared = ["".join(t.text or "" for t in si.iter(MAIN + "t")) for si in ss_root.findall(MAIN + "si")]

        wb = LET.fromstring(z.read("xl/workbook.xml"))
        rel_root = LET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        rels = {r.get("Id"): "xl/" + r.get("Target") for r in rel_root.findall(PKG + "Relationship")}
        sheets = []
        for node in wb.find(MAIN + "sheets"):
            sheets.append((node.get("name"), rels[node.get(RID + "id")]))

        def cell_value(c):
            typ = c.get("t")
            v = c.find(MAIN + "v")
            if typ == "s" and v is not None and v.text is not None:
                return shared[int(v.text)]
            if typ == "inlineStr":
                return "".join(t.text or "" for t in c.iter(MAIN + "t"))
            if v is None or v.text is None:
                return None
            try:
                return float(v.text)
            except Exception:
                return v.text

        for sheet_name, sheet_path in sheets:
            if _norm_name(sheet_name) == "description":
                continue
            meta0 = parse_dc_sheet_metadata(sheet_name)
            active: Dict[int, Dict[str, Any]] = {}
            pending_meta: Dict[int, Dict[str, Any]] = {}
            seq = defaultdict(int)

            def finish(base_col: int):
                a = active.pop(base_col, None)
                if not a:
                    return
                rows = a["rows"]
                if len(rows) < min_samples:
                    return
                arr = np.asarray(rows, dtype=np.float64)
                # [time_us,current,speed_rpm,voltage,pwm,status]
                valid = np.all(np.isfinite(arr[:, :4]), axis=1)
                arr = arr[valid]
                if len(arr) < min_samples:
                    return
                t_raw = arr[:, 0]
                # A block is required to be monotonic.  If malformed, reject only this block.
                if len(t_raw) < 2 or not np.all(np.diff(t_raw) > 0):
                    errors.append(f"{sheet_name}/{_xlsx_col_name(base_col)}: non-monotonic trial")
                    return
                t = (t_raw - t_raw[0]) * 1e-6
                dt = float(np.median(np.diff(t)))
                X = np.column_stack([arr[:, 1], arr[:, 2]]).astype(np.float64)
                U = arr[:-1, 3:4].astype(np.float64)
                seq[base_col] += 1
                md = dict(meta0)
                md.update(a.get("meta", {}))
                md.update({
                    "trial_index": int(seq[base_col]),
                    "block_column": _xlsx_col_name(base_col),
                    "source_file": str(path),
                    "pwm_present": bool(np.isfinite(arr[:, 4]).any()),
                    "motor_status_present": bool(np.isfinite(arr[:, 5]).any()),
                    "speed_source_unit": "rpm",
                })
                if np.isfinite(arr[:, 4]).any():
                    md["pwm_min"] = float(np.nanmin(arr[:, 4])); md["pwm_max"] = float(np.nanmax(arr[:, 4]))
                if np.isfinite(arr[:, 5]).any():
                    md["motor_status_values"] = sorted(float(v) for v in np.unique(arr[np.isfinite(arr[:,5]), 5]))
                load_level = md.get("load_level", meta0.get("load_level"))
                tid = f"{sheet_name}::{_xlsx_col_name(base_col)}::trial{seq[base_col]:03d}"
                out.append(RealTrajectory(
                    source="dc_motor_2021", trajectory_id=tid, motor_id=str(meta0["motor_id"]),
                    excitation_family=str(meta0["excitation_family"]), load_level=load_level,
                    dt=dt, time_s=t, X=X, U=U,
                    state_names=("current", "speed"), input_names=("motor_voltage",), metadata=md,
                ))

            with z.open(sheet_path) as fh:
                for _, row in LET.iterparse(fh, events=("end",), tag=MAIN + "row"):
                    rowvals: Dict[int, Any] = {}
                    for c in row.findall(MAIN + "c"):
                        rowvals[_xlsx_col_index(c.get("r", "A1"))] = cell_value(c)

                    # Descriptor rows terminate the previous trial in that horizontal block.
                    for j, v in list(rowvals.items()):
                        if isinstance(v, str) and v.startswith("Trial Description"):
                            finish(j)
                            md: Dict[str, Any] = {}
                            step = rowvals.get(j + 1)
                            if isinstance(step, (int, float)) and math.isfinite(float(step)):
                                md["step_percent"] = float(step)
                            if rowvals.get(j + 2) == "Load: ":
                                lv = rowvals.get(j + 3)
                                if isinstance(lv, (int, float)) and math.isfinite(float(lv)):
                                    md["load_level"] = float(lv)
                            elif meta0.get("excitation_family") == "step_noload":
                                md["load_level"] = 0.0
                            pending_meta[j] = md

                    # Every `time` header creates a new independent trial block.
                    for j, v in list(rowvals.items()):
                        if v != "time":
                            continue
                        finish(j)
                        header = {}
                        for k in range(j, j + 12):
                            hv = rowvals.get(k)
                            if isinstance(hv, str):
                                header[_norm_name(hv)] = k
                        required_norm = [_norm_name("time"), _norm_name("Current"), _norm_name("Velocity"), _norm_name("MotorVoltage")]
                        if not all(x in header for x in required_norm):
                            errors.append(f"{sheet_name}/{_xlsx_col_name(j)} header missing required columns")
                            continue
                        active[j] = {"header": header, "rows": [], "meta": pending_meta.pop(j, {})}

                    # Append numeric sample rows independently for every active horizontal block.
                    for j, a in list(active.items()):
                        tv = rowvals.get(j)
                        if not isinstance(tv, (int, float)) or not math.isfinite(float(tv)):
                            continue
                        h = a["header"]
                        def get(name, default=np.nan):
                            v = rowvals.get(h.get(_norm_name(name), -1), default)
                            try:
                                return float(v)
                            except Exception:
                                return float("nan")
                        a["rows"].append((
                            float(tv), get("Current"), get("Velocity"), get("MotorVoltage"),
                            get("PWM"), get("MotorStatus"),
                        ))
                    row.clear()

            for j in list(active):
                finish(j)

    if not out:
        raise M3DataError("no usable trajectories from xlsx stream; " + "; ".join(errors[:8]))
    # Cache only data parsed from this trusted local workbook.  Cache is invalidated by file sig.
    try:
        with gzip.open(cache, "wb", compresslevel=3) as f:
            pickle.dump({"source_sig": source_sig, "trajectories": out, "errors": errors}, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass
    return out


def load_dc_motor_2021(path: str | Path, min_samples: int = 20) -> List[RealTrajectory]:
    """Load the 2021 DC motor workbook or a directory of per-tab CSV exports."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.is_file() and p.suffix.lower() in (".xlsx", ".xlsm"):
        return _load_dc_motor_xlsx_stream(p, min_samples=min_samples)

    # CSV support is retained for locally exported copies; pandas is imported lazily so the
    # primary xlsx runtime does not depend on an Excel engine.
    try:
        import pandas as pd
    except ImportError as e:
        raise M3DataError("pandas is required only for CSV/DC-export loading") from e
    tables: List[Tuple[str, Any]] = []
    if p.is_file() and p.suffix.lower() == ".csv":
        tables.append((p.stem, pd.read_csv(p)))
    elif p.is_dir():
        for f in sorted(p.glob("*.csv")):
            tables.append((f.stem, pd.read_csv(f)))
    else:
        raise M3DataError(f"unsupported source: {p}")

    out: List[RealTrajectory] = []
    errors: List[str] = []
    for name, df in tables:
        if df.empty:
            continue
        meta = parse_dc_sheet_metadata(name)
        try:
            cols = _resolve_columns(df, DC_ALIASES, required=("time", "current", "speed", "motor_voltage"))
            chunks = _split_on_time_resets(df, cols["time"], min_samples=min_samples)
            for k, chunk in enumerate(chunks, start=1):
                try:
                    out.append(_dc_chunk_to_trajectory(chunk, meta, k, str(p), min_samples=min_samples))
                except M3DataError as e:
                    errors.append(f"{name}/trial{k}: {e}")
        except M3DataError as e:
            errors.append(f"{name}: {e}")
    if not out:
        raise M3DataError("no usable trajectories; " + "; ".join(errors[:8]))
    return out


def load_pmsm_profile_csv(path: str | Path, nominal_dt: float = 0.5) -> List[RealTrajectory]:
    """Load PMSM profiles as a *smoke-test adapter*.

    This adapter is not the preferred M3 target because the frozen M2 equation grammar does
    not contain a general loss/product library needed for a faithful PMSM thermal model.
    """
    p = Path(path)
    import pandas as pd
    df = pd.read_csv(p)
    cols = {_norm_name(c): str(c) for c in df.columns}
    missing = [c for c in PMSM_REQUIRED if _norm_name(c) not in cols]
    if missing:
        raise M3DataError(f"PMSM missing columns {missing}")
    out: List[RealTrajectory] = []
    pid_col = cols[_norm_name("profile_id")]
    for pid, g in df.groupby(pid_col, sort=False):
        g = g.reset_index(drop=True)
        n = len(g)
        if n < 4:
            continue
        t = np.arange(n, dtype=float) * nominal_dt
        X = g[[cols[_norm_name("pm")], cols[_norm_name("stator_yoke")],
               cols[_norm_name("stator_tooth")], cols[_norm_name("stator_winding")]]].to_numpy(float)
        Ufull = g[[cols[_norm_name("u_d")], cols[_norm_name("u_q")], cols[_norm_name("motor_speed")],
                   cols[_norm_name("torque")], cols[_norm_name("i_d")], cols[_norm_name("i_q")],
                   cols[_norm_name("ambient")], cols[_norm_name("coolant")]]].to_numpy(float)
        out.append(RealTrajectory(
            source="pmsm_temperature_profile",
            trajectory_id=f"profile_{pid}", motor_id="PMSM", excitation_family="profile",
            load_level=None, dt=nominal_dt, time_s=t, X=X, U=Ufull[:-1],
            state_names=("pm","stator_yoke","stator_tooth","stator_winding"),
            input_names=("u_d","u_q","motor_speed","torque","i_d","i_q","ambient","coolant"),
            metadata={"profile_id": str(pid), "warning": "parser smoke only; not sufficient for M3 PASS"},
        ))
    if not out:
        raise M3DataError("no PMSM profiles found")
    return out


def audit_trajectories(trajs: Sequence[RealTrajectory]) -> Dict[str, Any]:
    rows = [t.validate() for t in trajs]
    ok = [r for r in rows if r["ok"]]
    by_motor: Dict[str, int] = {}
    by_family: Dict[str, int] = {}
    for t in trajs:
        by_motor[t.motor_id] = by_motor.get(t.motor_id, 0) + 1
        by_family[t.excitation_family] = by_family.get(t.excitation_family, 0) + 1
    dts = np.asarray([t.dt for t in trajs], dtype=float)
    return {
        "n_trajectories": len(trajs),
        "n_valid": len(ok),
        "all_valid": len(ok) == len(trajs),
        "by_motor": by_motor,
        "by_excitation_family": by_family,
        "dt_median": float(np.median(dts)) if len(dts) else None,
        "dt_min": float(np.min(dts)) if len(dts) else None,
        "dt_max": float(np.max(dts)) if len(dts) else None,
        "trajectory_audits": rows,
    }


def propose_dc_motor_splits(trajs: Sequence[RealTrajectory]) -> Dict[str, List[str]]:
    """Conservative M3 split proposal using whole trajectories.

    - loaded step trials are never used to fit the no-load operational graph; they are reserved
      for grammar-coverage stress because the external load torque is not part of the public state/input.
    - motor B is reserved for cross-machine stress rather than mixed into IID training.
    - exact train/validation assignment is finalized only after the real workbook audit confirms
      trial counts and sheet layout.
    """
    train_pool: List[str] = []
    iid_pool: List[str] = []
    near_ood: List[str] = []
    far_ood: List[str] = []
    cross_machine: List[str] = []
    cross_machine_loaded: List[str] = []
    coverage_stress: List[str] = []
    unassigned: List[str] = []
    for t in trajs:
        fam = t.excitation_family
        freq = t.metadata.get("frequency_hz")
        # Keep the two stress axes orthogonal: Motor-A loaded isolates missing load input;
        # Motor-B loaded is a combined cross-machine+load condition and is diagnostic only.
        if t.motor_id == "B":
            (cross_machine_loaded if fam == "step_loaded" else cross_machine).append(t.trajectory_id)
            continue
        if fam == "step_loaded":
            coverage_stress.append(t.trajectory_id); continue
        if t.motor_id != "A":
            unassigned.append(t.trajectory_id); continue
        if fam in ("prbs7", "prbs9"):
            train_pool.append(t.trajectory_id)
        elif fam in ("sine", "triangle") and freq in (0.1, 1.0):
            near_ood.append(t.trajectory_id)
        elif fam in ("sine", "triangle") and freq == 10.0:
            far_ood.append(t.trajectory_id)
        elif fam in ("square", "step_noload"):
            iid_pool.append(t.trajectory_id)
        else:
            unassigned.append(t.trajectory_id)
    return {
        "train_pool": train_pool,
        "iid_holdout_pool": iid_pool,
        "near_ood_pool": near_ood,
        "far_ood_pool": far_ood,
        "cross_machine_stress": cross_machine,
        "cross_machine_loaded": cross_machine_loaded,
        "grammar_coverage_stress": coverage_stress,
        "unassigned": unassigned,
    }


def save_audit(path: str | Path, payload: Dict[str, Any]) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
