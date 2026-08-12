#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training preparation for M3 real-system validation.

This module is deliberately separate from the source adapters.  It converts validated
``RealTrajectory`` objects into the canonical ``PublicDataset`` used by the frozen M2
ISGN graph and baselines.

Scientific invariants:
  * split unit is a full source trajectory / experiment;
  * segmentation happens only *after* split assignment;
  * a chunk never crosses a source trajectory or a split boundary;
  * source physical coordinates are preserved (no z-score inside grammar operators);
  * DC motor speed is converted from RPM to rad/s before system identification;
  * uniform resampling is permitted only within one trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import math
import numpy as np

from .grammar import DIMENSIONS
from .contracts import PublicDataset
from .real_motor import RealTrajectory, M3DataError


REAL_DC_SCHEMA: Dict = {
    "states": [
        {"name": "i", "dim": "current"},
        {"name": "w", "dim": "speed"},
    ],
    "inputs": [
        {"name": "V", "dim": "voltage"},
    ],
}


@dataclass(frozen=True)
class ChunkRecord:
    split: str
    trajectory_id: str
    start_index: int
    stop_index: int


@dataclass
class PreparedSplit:
    name: str
    public: PublicDataset
    chunks: List[ChunkRecord]
    source_trajectory_ids: Tuple[str, ...]


def _rpm_to_rad_s(v: np.ndarray) -> np.ndarray:
    return np.asarray(v, dtype=np.float64) * (2.0 * math.pi / 60.0)


def resample_trajectory(
    traj: RealTrajectory,
    target_dt: Optional[float] = None,
    *,
    convert_speed_to_rad_s: bool = True,
) -> RealTrajectory:
    """Uniformly resample one trajectory without mixing source trials.

    Linear interpolation is used only as a sampling-grid correction.  The target grid
    is clipped to the measured time span and never extrapolates.
    """
    audit = traj.validate()
    if not audit["ok"]:
        raise M3DataError(f"invalid trajectory {traj.trajectory_id}: {audit['errors']}")
    dt = float(target_dt if target_dt is not None else traj.dt)
    if not math.isfinite(dt) or dt <= 0:
        raise M3DataError("target_dt must be positive")
    t0, t1 = float(traj.time_s[0]), float(traj.time_s[-1])
    n = int(math.floor((t1 - t0) / dt + 1e-9)) + 1
    if n < 3:
        raise M3DataError(f"resampled trajectory {traj.trajectory_id} is too short")
    grid = t0 + np.arange(n, dtype=np.float64) * dt
    # Make the source input a sample-time signal before interpolation.  U[k] drives
    # X[k] -> X[k+1]; for interpolation we assign U[-1] to the final sample only.
    src_u = np.asarray(traj.U, dtype=np.float64)
    if len(src_u) == len(traj.X) - 1:
        src_u_samples = np.vstack([src_u, src_u[-1:]])
    elif len(src_u) == len(traj.X):
        src_u_samples = src_u
    else:
        raise M3DataError("U must have T or T-1 rows")
    X = np.column_stack([
        np.interp(grid, traj.time_s, np.asarray(traj.X)[:, j])
        for j in range(traj.X.shape[1])
    ])
    U_samples = np.column_stack([
        np.interp(grid, traj.time_s, src_u_samples[:, j])
        for j in range(src_u_samples.shape[1])
    ])
    metadata = dict(traj.metadata)
    metadata.update({
        "source_dt": float(traj.dt),
        "resampled_dt": dt,
        "speed_source_unit": metadata.get("speed_source_unit", "rpm"),
        "speed_training_unit": "rad/s" if convert_speed_to_rad_s else metadata.get("speed_source_unit", "rpm"),
    })
    if convert_speed_to_rad_s:
        try:
            idx = traj.state_names.index("speed")
        except ValueError:
            try:
                idx = traj.state_names.index("motor_speed")
            except ValueError:
                idx = -1
        if idx >= 0:
            X[:, idx] = _rpm_to_rad_s(X[:, idx])
    return RealTrajectory(
        source=traj.source,
        trajectory_id=traj.trajectory_id,
        motor_id=traj.motor_id,
        excitation_family=traj.excitation_family,
        load_level=traj.load_level,
        dt=dt,
        time_s=grid,
        X=X,
        U=U_samples[:-1],
        state_names=traj.state_names,
        input_names=traj.input_names,
        metadata=metadata,
    )


def split_prbs_train_val_iid(
    trajectories: Sequence[RealTrajectory],
    *,
    train_fraction: float = 0.60,
    val_fraction: float = 0.20,
) -> Dict[str, List[str]]:
    """Finalize deterministic whole-trial splits for Motor-A PRBS experiments.

    Trials are grouped by PRBS family so both PRBS7 and PRBS9 contribute to train/val/IID
    whenever enough repeats are available.  No row/window is assigned independently.
    Other excitation families keep the M3 protocol buckets.
    """
    if not (0 < train_fraction < 1 and 0 <= val_fraction < 1 and train_fraction + val_fraction < 1):
        raise ValueError("invalid split fractions")
    by_family: Dict[str, List[RealTrajectory]] = {"prbs7": [], "prbs9": []}
    out = {
        "train": [], "val": [], "iid": [], "near_ood": [], "far_ood": [],
        "cross_machine": [], "cross_machine_loaded": [], "coverage_stress": [], "unassigned": [],
    }
    for tr in trajectories:
        fam = tr.excitation_family
        # Do not confound missing-load coverage with cross-machine transfer.
        if tr.motor_id == "B":
            out["cross_machine_loaded" if fam == "step_loaded" else "cross_machine"].append(tr.trajectory_id)
            continue
        if fam == "step_loaded":
            out["coverage_stress"].append(tr.trajectory_id); continue
        if tr.motor_id != "A":
            out["unassigned"].append(tr.trajectory_id); continue
        if fam in by_family:
            by_family[fam].append(tr); continue
        freq = tr.metadata.get("frequency_hz")
        if fam in ("sine", "triangle") and freq in (0.1, 1.0):
            out["near_ood"].append(tr.trajectory_id)
        elif fam in ("sine", "triangle") and freq == 10.0:
            out["far_ood"].append(tr.trajectory_id)
        elif fam in ("square", "step_noload"):
            out["iid"].append(tr.trajectory_id)
        else:
            out["unassigned"].append(tr.trajectory_id)
    for fam in ("prbs7", "prbs9"):
        rows = sorted(by_family[fam], key=lambda t: t.trajectory_id)
        n = len(rows)
        if n == 0:
            continue
        n_train = max(1, int(math.floor(train_fraction * n)))
        n_val = int(math.floor(val_fraction * n))
        if n >= 3:
            n_val = max(1, n_val)
        # preserve at least one IID PRBS trial when possible
        if n >= 3 and n_train + n_val >= n:
            n_train = max(1, n - n_val - 1)
        ids = [r.trajectory_id for r in rows]
        out["train"].extend(ids[:n_train])
        out["val"].extend(ids[n_train:n_train+n_val])
        out["iid"].extend(ids[n_train+n_val:])
    # Global no-leak assertion.
    seen: Dict[str, str] = {}
    for bucket, ids in out.items():
        for tid in ids:
            if tid in seen:
                raise AssertionError(f"trajectory leakage: {tid} in {seen[tid]} and {bucket}")
            seen[tid] = bucket
    return out


def _segments_from_trajectory(
    tr: RealTrajectory,
    split: str,
    horizon: int,
    stride: Optional[int],
    max_chunks_per_trajectory: Optional[int],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[ChunkRecord]]:
    if horizon < 2:
        raise ValueError("horizon must be >= 2 transitions")
    stride = int(stride or horizon)
    if stride <= 0:
        raise ValueError("stride must be positive")
    # Each PublicDataset chunk has X length horizon+1 and U length horizon.
    last_start = len(tr.X) - (horizon + 1)
    if last_start < 0:
        return [], [], []
    starts = list(range(0, last_start + 1, stride))
    if max_chunks_per_trajectory is not None and len(starts) > max_chunks_per_trajectory:
        # Deterministic coverage of the trajectory rather than taking only the beginning.
        idx = np.linspace(0, len(starts)-1, max_chunks_per_trajectory).round().astype(int)
        starts = [starts[j] for j in sorted(set(idx.tolist()))]
    xs: List[np.ndarray] = []
    us: List[np.ndarray] = []
    recs: List[ChunkRecord] = []
    for s in starts:
        e = s + horizon + 1
        xs.append(np.asarray(tr.X[s:e], dtype=np.float32))
        us.append(np.asarray(tr.U[s:e-1], dtype=np.float32))
        recs.append(ChunkRecord(split, tr.trajectory_id, s, e))
    return xs, us, recs


def prepare_public_split(
    trajectories: Sequence[RealTrajectory],
    trajectory_ids: Iterable[str],
    *,
    name: str,
    horizon: int = 200,
    stride: Optional[int] = None,
    max_chunks_per_trajectory: Optional[int] = 8,
    target_dt: Optional[float] = None,
) -> PreparedSplit:
    wanted = set(trajectory_ids)
    selected = [t for t in trajectories if t.trajectory_id in wanted]
    missing = wanted - {t.trajectory_id for t in selected}
    if missing:
        raise M3DataError(f"unknown trajectory IDs in {name}: {sorted(missing)}")
    if not selected:
        raise M3DataError(f"split {name} is empty")
    if target_dt is None:
        target_dt = float(np.median([t.dt for t in selected]))
    xs: List[np.ndarray] = []
    us: List[np.ndarray] = []
    recs: List[ChunkRecord] = []
    for tr in selected:
        rt = resample_trajectory(tr, target_dt=target_dt, convert_speed_to_rad_s=True)
        x1, u1, r1 = _segments_from_trajectory(rt, name, horizon, stride, max_chunks_per_trajectory)
        xs.extend(x1); us.extend(u1); recs.extend(r1)
    if not xs:
        raise M3DataError(f"split {name} has no trajectory long enough for horizon={horizon}")
    X = np.stack(xs, axis=0)
    U = np.stack(us, axis=0)
    # PublicDataset trajectory_id is chunk-level numeric identity because training code expects
    # a compact ndarray.  Original source trajectory identity remains in ChunkRecord and is
    # what split/leakage audits use.
    pub = PublicDataset(
        X=X,
        U=U,
        trajectory_id=np.arange(len(xs), dtype=np.int64),
        schema=REAL_DC_SCHEMA,
        dt=float(target_dt),
        regime=name,
        variant="real_dc_motor_2021",
    )
    return PreparedSplit(
        name=name,
        public=pub,
        chunks=recs,
        source_trajectory_ids=tuple(sorted({r.trajectory_id for r in recs})),
    )


def prepare_all_splits(
    trajectories: Sequence[RealTrajectory],
    split_ids: Mapping[str, Sequence[str]],
    *,
    horizon: int = 200,
    stride: Optional[int] = None,
    max_chunks_per_trajectory: Optional[int] = 8,
    target_dt: Optional[float] = None,
) -> Dict[str, PreparedSplit]:
    """Prepare every nonempty split with one common sampling interval."""
    ids_all = {tid for ids in split_ids.values() for tid in ids}
    base = [t for t in trajectories if t.trajectory_id in ids_all]
    if not base:
        raise M3DataError("no trajectories selected")
    if target_dt is None:
        target_dt = float(np.median([t.dt for t in base]))
    out: Dict[str, PreparedSplit] = {}
    for name, ids in split_ids.items():
        if not ids:
            continue
        out[name] = prepare_public_split(
            trajectories, ids, name=name, horizon=horizon, stride=stride,
            max_chunks_per_trajectory=max_chunks_per_trajectory, target_dt=target_dt,
        )
    # Hard split-leakage check at original trajectory level.
    owner: Dict[str, str] = {}
    for name, split in out.items():
        for tid in split.source_trajectory_ids:
            if tid in owner and owner[tid] != name:
                raise AssertionError(f"source trajectory leakage: {tid} in {owner[tid]} and {name}")
            owner[tid] = name
    return out


def split_manifest(prepared: Mapping[str, PreparedSplit]) -> Dict:
    return {
        name: {
            "n_chunks": int(p.public.X.shape[0]),
            "horizon": int(p.public.U.shape[1]),
            "dt": float(p.public.dt),
            "source_trajectory_ids": list(p.source_trajectory_ids),
        }
        for name, p in prepared.items()
    }
