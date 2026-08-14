#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pre-discovery Observation Gate for ISGN Fusion.

The gate does *not* discover a Mechanism Graph.  It audits whether the public
measurement/command trajectories contain enough usable information to justify
running a mechanism-discovery backend at all.

It intentionally separates five evidence families:
  1. trajectory/sampling sanity,
  2. persistent excitation of non-context inputs,
  3. source-column identifiability / collinearity,
  4. per-state derivative reliability under the declared sampling interval,
  5. temporal sufficiency of the declared instantaneous observation tuple.

A FAIL is a refusal to claim that the *full declared state graph* is currently
identifiable.  It is not a statement that the physical system has no mechanism.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.signal import savgol_filter

from .contracts import PublicDataset


@dataclass(frozen=True)
class ObservationGateConfig:
    min_horizon: int = 32
    hankel_order: int = 4
    min_hankel_rank_ratio: float = 0.60
    warn_source_condition: float = 1e5
    fail_source_condition: float = 1e8
    warn_derivative_noise_ratio: float = 1.0
    fail_derivative_noise_ratio: float = 2.0
    fail_probe_nrmse: float = 0.80
    warn_probe_nrmse: float = 1.20
    fail_bandwidth_nyquist_ratio: float = 0.60
    warn_bandwidth_nyquist_ratio: float = 0.65
    min_variation: float = 1e-10
    svd_relative_rank_tol: float = 1e-8
    smoothing_window: int = 11
    smoothing_polyorder: int = 3
    # Sensor-contract heuristics.  These do not alter mechanism-discovery
    # thresholds; they ask whether observed variation is distinguishable from
    # the declared measurement noise/quantization floor.
    warn_measurement_variation_ratio: float = 5.0
    fail_measurement_variation_ratio: float = 2.0
    # Optional per-sample timebase audit.  Existing fixed-rate datasets remain
    # backward compatible when ``sample_time_s`` is absent.
    warn_timebase_p95_relative_error: float = 0.05
    fail_timebase_p95_relative_error: float = 0.25
    warn_timebase_median_relative_error: float = 0.02
    fail_timebase_median_relative_error: float = 0.10
    fail_timebase_min_interval_ratio: float = 0.50
    fail_timebase_max_interval_ratio: float = 2.00
    # Causal temporal-sufficiency probe.  This is a preflight diagnostic, not a
    # mechanism learner: it asks whether finite observation history contains
    # substantially more held-out derivative information than the declared
    # instantaneous state/input tuple.
    temporal_history_lags: tuple[int, ...] = (1, 2, 4, 8, 16)
    temporal_ridge_alpha: float = 1.0
    temporal_min_train_rows_per_feature: float = 5.0
    temporal_fail_min_instant_nrmse: float = 0.20
    temporal_fail_max_history_nrmse: float = 0.50
    temporal_fail_max_gain_ratio: float = 0.40
    temporal_fail_min_absolute_gain: float = 0.20
    temporal_warn_min_instant_nrmse: float = 0.20
    temporal_warn_max_gain_ratio: float = 0.70
    temporal_warn_min_absolute_gain: float = 0.10


@dataclass
class GateCheck:
    name: str
    status: str  # PASS / WARN / FAIL / N/A
    metrics: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StateObservationAudit:
    state: str
    status: str
    std: float
    derivative_noise_ratio: float
    derivative_probe_nrmse: float
    f95_nyquist_ratio: float
    high_frequency_energy_fraction: float
    reason: str = ""
    recommendation: str = ""
    sensor_sample_period_s: Optional[float] = None
    sensor_delay_s: Optional[float] = None
    sensor_noise_std: Optional[float] = None
    sensor_quantization_step: Optional[float] = None
    sensor_floor_std: Optional[float] = None
    variation_to_sensor_floor: Optional[float] = None
    sensor_period_to_export_dt: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ObservationGateReport:
    status: str  # QUALIFIED / CONDITIONALLY_QUALIFIED / OBSERVATION_INSUFFICIENT
    can_run_discovery: bool
    checks: List[GateCheck]
    states: List[StateObservationAudit]
    usable_states: List[str]
    blocked_states: List[str]
    recommendations: List[str]
    config: Dict[str, Any]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "can_run_discovery": self.can_run_discovery,
            "checks": [x.to_dict() for x in self.checks],
            "states": [x.to_dict() for x in self.states],
            "usable_states": list(self.usable_states),
            "blocked_states": list(self.blocked_states),
            "recommendations": list(self.recommendations),
            "config": dict(self.config),
            "summary": dict(self.summary),
        }


def _effective_rank(A: np.ndarray, *, rel_tol: float) -> tuple[int, float, np.ndarray]:
    A = np.asarray(A, dtype=np.float64)
    if A.size == 0 or min(A.shape) == 0:
        return 0, float("inf"), np.zeros((0,), dtype=float)
    s = np.linalg.svd(A, compute_uv=False)
    if len(s) == 0 or s[0] <= 0:
        return 0, float("inf"), s
    rank = int(np.sum(s > max(float(s[0]) * rel_tol, 1e-12)))
    cond = float(s[0] / max(float(s[-1]), 1e-20))
    return rank, cond, s


def _standardized_columns(A: np.ndarray, names: List[str], min_variation: float):
    A = np.asarray(A, dtype=np.float64)
    centered = A - np.mean(A, axis=0, keepdims=True)
    sd = np.std(centered, axis=0)
    active = sd > min_variation
    inactive = [names[i] for i in range(len(names)) if not active[i]]
    if np.any(active):
        Z = centered[:, active] / sd[active]
    else:
        Z = np.zeros((len(A), 0), dtype=np.float64)
    active_names = [names[i] for i in range(len(names)) if active[i]]
    return Z, active_names, inactive


def _timebase_integrity(ds: PublicDataset, config: ObservationGateConfig) -> GateCheck:
    """Audit optional observed sample times against the nominal fixed ``dt``.

    Solvers in Toolkit 2.x still operate on a fixed nominal interval.  The
    optional time vector therefore acts as provenance/evidence: large gaps,
    non-monotone timestamps, or sustained rate mismatch must be refused before
    derivative-based discovery silently treats them as ordinary adjacent samples.
    """
    if ds.sample_time_s is None:
        return GateCheck(
            "timebase_integrity", "N/A", {"sample_time_present": False},
            "sample_time_s is not provided; nominal dt cannot be checked against observed timestamps",
            "provide sample_time_s for operational data when logging gaps or clock jitter are possible",
        )
    time=np.asarray(ds.sample_time_s,dtype=np.float64)
    expected=tuple(np.asarray(ds.X).shape[:2])
    if time.shape != expected:
        return GateCheck(
            "timebase_integrity", "FAIL", {"sample_time_shape":list(time.shape),"expected_shape":list(expected)},
            "sample_time_s shape does not match the state observation grid",
            "rebuild the observation contract with one sample time for every X sample",
        )
    if not np.isfinite(time).all():
        return GateCheck(
            "timebase_integrity", "FAIL", {"sample_time_present":True},
            "sample_time_s contains NaN/Inf",
            "repair or explicitly segment the acquisition timebase before discovery",
        )
    delta=np.diff(time,axis=1)
    if delta.size == 0 or np.any(delta <= 0.0):
        return GateCheck(
            "timebase_integrity", "FAIL", {"nonpositive_interval_count":int(np.sum(delta<=0.0))},
            "sample_time_s is not strictly increasing within every trajectory",
            "sort/segment the acquisition by physical time; do not identify dynamics across duplicate or reversed timestamps",
        )
    nominal=float(ds.dt)
    ratio=delta/nominal
    rel=np.abs(ratio-1.0)
    median_interval=float(np.median(delta))
    median_rel=float(abs(median_interval-nominal)/nominal)
    p95_rel=float(np.percentile(rel,95.0))
    min_ratio=float(np.min(ratio)); max_ratio=float(np.max(ratio))
    metrics={
        "sample_time_present":True,
        "declared_dt_s":nominal,
        "median_interval_s":median_interval,
        "min_interval_s":float(np.min(delta)),
        "max_interval_s":float(np.max(delta)),
        "median_relative_error":median_rel,
        "p95_relative_error":p95_rel,
        "min_interval_ratio":min_ratio,
        "max_interval_ratio":max_ratio,
        "interval_count":int(delta.size),
    }
    hard=(median_rel > config.fail_timebase_median_relative_error
          or p95_rel > config.fail_timebase_p95_relative_error
          or min_ratio < config.fail_timebase_min_interval_ratio
          or max_ratio > config.fail_timebase_max_interval_ratio)
    if hard:
        return GateCheck(
            "timebase_integrity", "FAIL", metrics,
            "observed sample timing is incompatible with the declared fixed-rate dt",
            "split logging gaps and resample/validate jitter on a physical timebase before mechanism discovery",
        )
    if median_rel > config.warn_timebase_median_relative_error or p95_rel > config.warn_timebase_p95_relative_error:
        return GateCheck(
            "timebase_integrity", "WARN", metrics,
            "observed sample timing has material jitter relative to the declared fixed-rate dt",
            "preserve the timebase and verify resampling/filtered identification before interpreting fast dynamics",
        )
    return GateCheck(
        "timebase_integrity", "PASS", metrics,
        "observed sample timing is consistent with the declared nominal dt", "",
    )


def _hankel_excitation(ds: PublicDataset, config: ObservationGateConfig) -> GateCheck:
    inputs = list(ds.schema.get("inputs", []))
    dyn_idx = [i for i, v in enumerate(inputs) if v.get("role", "input") != "context"]
    dyn_names = [inputs[i]["name"] for i in dyn_idx]
    if not dyn_idx:
        return GateCheck("persistent_excitation", "N/A", {"dynamic_inputs": []},
                         "no non-context input is declared", "audit autonomous-state excitation separately if required")
    order = min(config.hankel_order, int(ds.U.shape[1]))
    if order < 2:
        return GateCheck("persistent_excitation", "FAIL", {"dynamic_inputs": dyn_names, "order": order},
                         "trajectory horizon is too short for a persistent-excitation audit",
                         "collect longer trajectories")
    rows = []
    U = np.asarray(ds.U, dtype=np.float64)
    for n in range(U.shape[0]):
        for t in range(order - 1, U.shape[1]):
            rows.append(np.concatenate([U[n, t-k, dyn_idx] for k in range(order)]))
    H = np.asarray(rows, dtype=np.float64)
    col_names = [f"{name}[t-{k}]" for k in range(order) for name in dyn_names]
    Z, active_names, inactive = _standardized_columns(H, col_names, config.min_variation)
    rank, cond, _ = _effective_rank(Z, rel_tol=1e-6)
    ratio = float(rank / max(len(col_names), 1))
    metrics = {
        "dynamic_inputs": dyn_names,
        "hankel_order": order,
        "rank": rank,
        "n_columns": len(col_names),
        "rank_ratio": ratio,
        "condition_number": cond,
        "inactive_lag_columns": inactive,
    }
    if inactive or ratio < config.min_hankel_rank_ratio:
        return GateCheck("persistent_excitation", "FAIL", metrics,
                         "non-context commands do not provide enough independent temporal excitation",
                         "use PRBS/chirp/multisine or otherwise vary commands independently across the target time scales")
    return GateCheck("persistent_excitation", "PASS", metrics,
                     "command history has adequate numerical rank", "")


def _source_identifiability(ds: PublicDataset, config: ObservationGateConfig) -> GateCheck:
    X = np.asarray(ds.X, dtype=np.float64)
    U = np.asarray(ds.U, dtype=np.float64)
    xm = 0.5 * (X[:, :-1] + X[:, 1:])
    A = np.concatenate([xm, U], axis=-1).reshape(-1, X.shape[-1] + U.shape[-1])
    names = [v["name"] for v in ds.schema.get("states", [])] + [v["name"] for v in ds.schema.get("inputs", [])]
    Z, active_names, inactive = _standardized_columns(A, names, config.min_variation)
    rank, cond, _ = _effective_rank(Z, rel_tol=config.svd_relative_rank_tol)
    metrics = {
        "rank": rank,
        "n_declared_sources": len(names),
        "n_active_sources": len(active_names),
        "condition_number": cond,
        "inactive_sources": inactive,
        "active_sources": active_names,
    }
    if inactive:
        return GateCheck("source_identifiability", "FAIL", metrics,
                         "one or more declared state/input sources do not vary in the calibration data",
                         "vary the inactive source, remove it from the claimed graph scope, or add an independent experiment")
    if rank < len(names) or cond > config.fail_source_condition:
        return GateCheck("source_identifiability", "FAIL", metrics,
                         "candidate state/input sources are rank-deficient or nearly collinear",
                         "decorrelate inputs/contexts or add independent sensing/experiments before attributing separate mechanisms")
    if cond > config.warn_source_condition:
        return GateCheck("source_identifiability", "WARN", metrics,
                         "source matrix is full rank but poorly conditioned",
                         "increase independent excitation before interpreting individual coefficients")
    return GateCheck("source_identifiability", "PASS", metrics,
                     "declared source columns are numerically distinguishable", "")


def _state_derivative_probe(ds: PublicDataset, config: ObservationGateConfig) -> List[StateObservationAudit]:
    X = np.asarray(ds.X, dtype=np.float64)
    U = np.asarray(ds.U, dtype=np.float64)
    dt = float(ds.dt)
    N, T, D = X.shape
    names = [v["name"] for v in ds.schema.get("states", [])]
    if len(names) != D:
        names = [f"x{i}" for i in range(D)]
    maxw = T if T % 2 == 1 else T - 1
    w = min(int(config.smoothing_window), maxw)
    if w % 2 == 0:
        w -= 1
    if w >= 5:
        p = min(int(config.smoothing_polyorder), w - 2)
        Xs = savgol_filter(X, window_length=w, polyorder=p, axis=1, mode="interp")
    else:
        Xs = X
    d_raw = np.diff(X, axis=1) / dt
    d_smooth = np.diff(Xs, axis=1) / dt
    noise_ratio = np.sqrt(np.mean((d_raw-d_smooth)**2, axis=(0,1))) / (np.sqrt(np.mean(d_smooth**2, axis=(0,1))) + 1e-12)

    # A deliberately simple local dynamics probe.  It is diagnostic only: high
    # error by itself does not reject nonlinear mechanisms.  It becomes a hard
    # failure only together with derivative/noise evidence.
    xm = 0.5 * (X[:, :-1] + X[:, 1:])
    A = np.concatenate([xm, U, np.ones((*xm.shape[:2], 1), dtype=float)], axis=-1)
    ntr = max(1, int(np.floor(0.70 * N)))
    Atr = A[:ntr].reshape(-1, A.shape[-1]); Ytr = d_raw[:ntr].reshape(-1, D)
    if ntr < N:
        Ava = A[ntr:].reshape(-1, A.shape[-1]); Yva = d_raw[ntr:].reshape(-1, D)
    else:
        Ava, Yva = Atr, Ytr
    coef = np.linalg.lstsq(Atr, Ytr, rcond=None)[0]
    pred = Ava @ coef
    probe = np.sqrt(np.mean((Yva-pred)**2, axis=0)) / (np.std(Yva, axis=0) + 1e-12)

    fs = 1.0 / dt
    nyq = 0.5 * fs
    audits: List[StateObservationAudit] = []
    for j, name in enumerate(names):
        z = X[:,:,j] - np.mean(X[:,:,j], axis=1, keepdims=True)
        spec = np.abs(np.fft.rfft(z, axis=1))**2
        power = np.mean(spec, axis=0)
        freq = np.fft.rfftfreq(T, dt)
        if len(power): power[0] = 0.0
        total = float(np.sum(power))
        if total <= 1e-30:
            hf = 0.0; f95 = 0.0
        else:
            hf = float(np.sum(power[freq >= 0.4*fs]) / total)
            cs = np.cumsum(power)
            idx = int(np.searchsorted(cs, 0.95*cs[-1]))
            f95 = float(freq[min(idx, len(freq)-1)] / max(nyq, 1e-12))
        sd = float(np.std(X[:,:,j]))
        nr = float(noise_ratio[j]); pr = float(probe[j])
        status = "PASS"; reason = "state increments are numerically usable at the declared sampling interval"; rec = ""
        if sd <= config.min_variation:
            status = "FAIL"; reason = "state has essentially no variation"; rec = "collect a trajectory in which this state is actually excited"
        elif (nr > config.fail_derivative_noise_ratio and pr > config.fail_probe_nrmse) or (nr > 1.5 and f95 > config.fail_bandwidth_nyquist_ratio):
            status = "FAIL"; reason = "state derivative is not reliably resolved at the declared sampling/noise level"
            rec = "increase sampling/bandwidth, improve sensing, or use an explicitly validated weak/filtered identification frontend before mechanism discovery"
        elif nr > config.warn_derivative_noise_ratio or f95 > config.warn_bandwidth_nyquist_ratio or pr > config.warn_probe_nrmse:
            status = "WARN"; reason = "state dynamics are marginal for direct derivative-based identification"
            rec = "prefer filtered/weak-form identification and verify this state separately before interpreting physical parameters"
        audits.append(StateObservationAudit(name,status,sd,nr,pr,f95,hf,reason,rec))
    return audits


def _attach_sensor_contract_metrics(ds: PublicDataset, states: List[StateObservationAudit]) -> None:
    specs = list(ds.schema.get("states", []))
    by_name = {str(x.get("name")): x for x in specs}
    dt = float(ds.dt)
    for audit in states:
        spec = by_name.get(audit.state, {})
        sensor = spec.get("sensor") if isinstance(spec, dict) else None
        if not isinstance(sensor, dict):
            continue
        period = float(sensor.get("sample_period_s", 0.0) or 0.0)
        delay = float(sensor.get("delay_s", 0.0) or 0.0)
        noise = abs(float(sensor.get("noise_std", 0.0) or 0.0))
        quant = abs(float(sensor.get("quantization_step", 0.0) or 0.0))
        # Standard deviation of a uniform quantization error is q/sqrt(12).
        floor = float(np.sqrt(noise * noise + (quant * quant) / 12.0))
        ratio = None if floor <= 0.0 else float(audit.std / max(floor, 1e-30))
        audit.sensor_sample_period_s = period
        audit.sensor_delay_s = delay
        audit.sensor_noise_std = noise
        audit.sensor_quantization_step = quant
        audit.sensor_floor_std = floor
        audit.variation_to_sensor_floor = ratio
        audit.sensor_period_to_export_dt = None if dt <= 0 else float(period / dt)



def _causal_ridge_nrmse(
    Xtr: np.ndarray,
    Ytr: np.ndarray,
    Xva: np.ndarray,
    Yva: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    """Deterministic standardized ridge probe used only by temporal preflight."""
    Xtr = np.asarray(Xtr, dtype=np.float64)
    Ytr = np.asarray(Ytr, dtype=np.float64)
    Xva = np.asarray(Xva, dtype=np.float64)
    Yva = np.asarray(Yva, dtype=np.float64)
    if Ytr.ndim == 1:
        Ytr = Ytr[:, None]
    if Yva.ndim == 1:
        Yva = Yva[:, None]
    mu = np.mean(Xtr, axis=0, keepdims=True)
    sd = np.std(Xtr, axis=0, keepdims=True)
    active = sd.ravel() > 1e-12
    if not np.any(active):
        pred = np.broadcast_to(np.mean(Ytr, axis=0, keepdims=True), Yva.shape)
    else:
        Ztr = (Xtr[:, active] - mu[:, active]) / sd[:, active]
        Zva = (Xva[:, active] - mu[:, active]) / sd[:, active]
        ym = np.mean(Ytr, axis=0, keepdims=True)
        Yc = Ytr - ym
        gram = Ztr.T @ Ztr
        reg = max(float(alpha), 0.0) * np.eye(gram.shape[0], dtype=np.float64)
        try:
            coef = np.linalg.solve(gram + reg, Ztr.T @ Yc)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(gram + reg, Ztr.T @ Yc, rcond=None)[0]
        pred = Zva @ coef + ym
    rmse = np.sqrt(np.mean((Yva - pred) ** 2, axis=0))
    scale = np.std(Yva, axis=0) + 1e-12
    return rmse / scale


def _temporal_sufficiency_check(ds: PublicDataset, config: ObservationGateConfig) -> GateCheck:
    """Compare causal instantaneous and finite-history derivative predictability.

    A large held-out gain from history is evidence that the declared observation
    tuple is not temporally sufficient for an *instantaneous full-state* claim.
    It can arise from omitted physical state, sensor/observer dynamics, delay or
    distributed dynamics.  The probe deliberately does not decide which remedy
    is correct and never treats lag coefficients as physical mechanisms.
    """
    X = np.asarray(ds.X, dtype=np.float64)
    U = np.asarray(ds.U, dtype=np.float64)
    N, T, D = X.shape
    names = [str(v.get("name", f"x{i}")) for i, v in enumerate(ds.schema.get("states", []))]
    if len(names) != D:
        names = [f"x{i}" for i in range(D)]
    if N < 4 or T < 8:
        return GateCheck(
            "temporal_sufficiency", "N/A",
            {"n_trajectories": int(N), "horizon": int(T - 1)},
            "too few independent trajectories or samples for a held-out temporal-sufficiency audit",
            "collect multiple independent trajectories before making an instantaneous full-state mechanism claim",
        )

    ntr = max(2, int(np.floor(0.70 * N)))
    if ntr >= N:
        ntr = N - 1
    train_idx = np.arange(ntr)
    val_idx = np.arange(ntr, N)
    dt = float(ds.dt)
    input_meta = list(ds.schema.get("inputs", []))
    context_idx = [i for i, meta in enumerate(input_meta)
                   if meta.get("role", "input") == "context"]
    context_names = [str(input_meta[i].get("name", f"u{i}")) for i in context_idx]

    def rows(indices: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
        feats: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for n in indices:
            for t in range(lag, T - 1):
                parts = []
                for k in range(lag + 1):
                    xk = X[n, t - k]
                    parts.append(xk)
                    if U.shape[-1]:
                        uk = U[n, t - k]
                        parts.append(uk)
                        # Declared context is allowed to modulate local state
                        # dynamics.  Without this tiny bilinear basis, a valid
                        # parameter-varying Markov system (for example, an
                        # oscillator whose frequency changes with speed) can be
                        # misdiagnosed as hidden temporal state merely because
                        # an additive linear diagnostic cannot express X*c.
                        # Dynamic inputs are deliberately excluded from this
                        # expansion so this remains a conservative preflight,
                        # not a generic nonlinear mechanism learner.
                        if context_idx:
                            ck = uk[context_idx]
                            parts.append((xk[:, None] * ck[None, :]).reshape(-1))
                feats.append(np.concatenate(parts) if parts else np.empty((0,), dtype=np.float64))
                targets.append((X[n, t + 1] - X[n, t]) / dt)
        if not feats:
            return np.zeros((0, 0), dtype=np.float64), np.zeros((0, D), dtype=np.float64)
        return np.asarray(feats, dtype=np.float64), np.asarray(targets, dtype=np.float64)

    X0tr, Y0tr = rows(train_idx, 0)
    X0va, Y0va = rows(val_idx, 0)
    if len(X0tr) == 0 or len(X0va) == 0:
        return GateCheck(
            "temporal_sufficiency", "N/A", {},
            "no held-out rows are available for the temporal-sufficiency audit",
            "collect multiple independent trajectories",
        )
    inst = _causal_ridge_nrmse(X0tr, Y0tr, X0va, Y0va, alpha=config.temporal_ridge_alpha)
    best = inst.copy()
    best_lag = np.zeros(D, dtype=int)
    tested: list[int] = []
    skipped: dict[int, str] = {}
    for lag in sorted({int(x) for x in config.temporal_history_lags if int(x) > 0}):
        if lag >= T - 2:
            skipped[lag] = "lag_exceeds_trajectory_horizon"
            continue
        Xhtr, Yhtr = rows(train_idx, lag)
        Xhva, Yhva = rows(val_idx, lag)
        nfeat = max(int(Xhtr.shape[1]), 1)
        if len(Xhtr) < float(config.temporal_min_train_rows_per_feature) * nfeat or len(Xhva) < max(16, D * 4):
            skipped[lag] = "insufficient_rows_for_history_dimension"
            continue
        hn = _causal_ridge_nrmse(Xhtr, Yhtr, Xhva, Yhva, alpha=config.temporal_ridge_alpha)
        tested.append(lag)
        improved = hn < best
        best[improved] = hn[improved]
        best_lag[improved] = lag

    if not tested:
        return GateCheck(
            "temporal_sufficiency", "N/A",
            {"instantaneous_nrmse": {names[i]: float(inst[i]) for i in range(D)}, "skipped_lags": skipped},
            "trajectory count/horizon is insufficient for a stable finite-history comparison",
            "collect longer or more independent trajectories before interpreting temporal sufficiency",
        )

    ratio = best / np.maximum(inst, 1e-12)
    gain = inst - best
    fail_states = []
    warn_states = []
    for i, name in enumerate(names):
        hard = (
            inst[i] >= config.temporal_fail_min_instant_nrmse
            and best[i] <= config.temporal_fail_max_history_nrmse
            and ratio[i] <= config.temporal_fail_max_gain_ratio
            and gain[i] >= config.temporal_fail_min_absolute_gain
        )
        soft = (
            inst[i] >= config.temporal_warn_min_instant_nrmse
            and ratio[i] <= config.temporal_warn_max_gain_ratio
            and gain[i] >= config.temporal_warn_min_absolute_gain
        )
        if hard:
            fail_states.append(name)
        elif soft:
            warn_states.append(name)

    metrics = {
        "diagnostic_only": True,
        "causal_instantaneous_features": (
            "X[t], U[t], X[t]xcontext[t]" if context_idx else "X[t], U[t]"
        ),
        "context_interactions_enabled": bool(context_idx),
        "context_inputs": context_names,
        "history_lags_tested": tested,
        "skipped_lags": skipped,
        "instantaneous_nrmse": {names[i]: float(inst[i]) for i in range(D)},
        "best_history_nrmse": {names[i]: float(best[i]) for i in range(D)},
        "best_history_lag_steps": {names[i]: int(best_lag[i]) for i in range(D)},
        "history_gain_ratio": {names[i]: float(ratio[i]) for i in range(D)},
        "absolute_nrmse_gain": {names[i]: float(gain[i]) for i in range(D)},
        "insufficient_states": fail_states,
        "marginal_states": warn_states,
        "train_trajectories": int(len(train_idx)),
        "validation_trajectories": int(len(val_idx)),
    }
    if fail_states:
        return GateCheck(
            "temporal_sufficiency", "FAIL", metrics,
            "finite observation history materially outpredicts the declared instantaneous state/input tuple for one or more states",
            "treat the affected observation scope as dynamically incomplete; add independently measured state or a validated state-estimation/temporal frontend before full-state mechanism discovery, and do not interpret lag coefficients as mechanisms",
        )
    if warn_states:
        return GateCheck(
            "temporal_sufficiency", "WARN", metrics,
            "finite observation history provides material held-out predictive gain for one or more states",
            "audit omitted state, delay, sensor dynamics or distributed memory before interpreting an instantaneous mechanism graph",
        )
    return GateCheck(
        "temporal_sufficiency", "PASS", metrics,
        "finite history does not provide a material held-out advantage over the declared instantaneous observation tuple",
        "",
    )

def _measurement_chain_check(ds: PublicDataset, states: List[StateObservationAudit], config: ObservationGateConfig) -> GateCheck:
    _attach_sensor_contract_metrics(ds, states)
    with_sensor = [s for s in states if s.sensor_sample_period_s is not None]
    if not with_sensor:
        return GateCheck(
            "measurement_chain", "N/A", {"sensor_metadata_states": []},
            "state schema does not provide sensor noise/sample/delay metadata",
            "provide a measurement contract when mechanism claims depend on real sensor observability",
        )

    low_fail = [s.state for s in with_sensor
                if s.variation_to_sensor_floor is not None and s.variation_to_sensor_floor < config.fail_measurement_variation_ratio]
    low_warn = [s.state for s in with_sensor
                if s.variation_to_sensor_floor is not None
                and config.fail_measurement_variation_ratio <= s.variation_to_sensor_floor < config.warn_measurement_variation_ratio]
    held = [s.state for s in with_sensor
            if (s.sensor_period_to_export_dt or 0.0) > 1.0 + 1e-6]
    downsampled = [s.state for s in with_sensor
                   if 0.0 < (s.sensor_period_to_export_dt or 0.0) < 1.0 - 1e-6]
    delayed = [s.state for s in with_sensor if (s.sensor_delay_s or 0.0) > float(ds.dt) + 1e-12]
    metrics = {
        "sensor_metadata_states": [s.state for s in with_sensor],
        "variation_to_sensor_floor": {s.state: s.variation_to_sensor_floor for s in with_sensor},
        "sensor_period_to_export_dt": {s.state: s.sensor_period_to_export_dt for s in with_sensor},
        "delay_s": {s.state: s.sensor_delay_s for s in with_sensor},
        "resolution_insufficient_states": low_fail,
        "low_margin_states": low_warn,
        "sample_held_states": held,
        "native_sensor_faster_than_export_states": downsampled,
        "delay_longer_than_export_dt_states": delayed,
    }
    if low_fail:
        return GateCheck(
            "measurement_chain", "FAIL", metrics,
            "one or more measured states vary at or below the declared sensor noise/quantization floor",
            "increase physical excitation/observation duration or improve the affected sensing chain before claiming those state mechanisms",
        )
    if low_warn or held:
        return GateCheck(
            "measurement_chain", "WARN", metrics,
            "one or more measured states have limited sensor margin or are sample-held on the export grid",
            "use update-aware filtering/resampling and preserve the sensor contract before interpreting derivative-based mechanisms",
        )
    return GateCheck(
        "measurement_chain", "PASS", metrics,
        "declared sensor noise/quantization is below observed state variation on the analyzed trajectories", "",
    )


def _sampling_bandwidth_check(states: List[StateObservationAudit], config: ObservationGateConfig) -> GateCheck:
    risky=[s for s in states if s.f95_nyquist_ratio > config.warn_bandwidth_nyquist_ratio]
    hard=[s for s in states if s.derivative_noise_ratio > 1.5 and s.f95_nyquist_ratio > config.fail_bandwidth_nyquist_ratio]
    metrics={
        "max_f95_nyquist_ratio": max([s.f95_nyquist_ratio for s in states]+[0.0]),
        "near_nyquist_states": [s.state for s in risky],
        "sampling_limited_states": [s.state for s in hard],
    }
    if hard:
        return GateCheck("sampling_bandwidth","FAIL",metrics,
                         "one or more state channels combine high derivative noise with energy close to Nyquist",
                         "increase sensor/sample bandwidth or restrict the claimed mechanism graph to states whose dynamics are resolved")
    if risky:
        return GateCheck("sampling_bandwidth","WARN",metrics,
                         "one or more state channels place substantial dynamic content close to the sampling limit",
                         "verify with a higher-rate acquisition or an anti-alias/observer study before interpreting fast mechanisms")
    return GateCheck("sampling_bandwidth","PASS",metrics,
                     "observed state bandwidth is not concentrated near Nyquist","")


def assess_observation(ds: PublicDataset, *, config: Optional[ObservationGateConfig]=None) -> ObservationGateReport:
    """Audit whether ``ds`` is qualified for full-state Mechanism Discovery."""
    cfg = config or ObservationGateConfig()
    checks: List[GateCheck] = []
    X = np.asarray(ds.X)
    U = np.asarray(ds.U)
    finite = bool(np.isfinite(X).all() and np.isfinite(U).all() and np.isfinite(ds.dt) and ds.dt > 0)
    if not finite:
        checks.append(GateCheck("data_integrity","FAIL",{},"NaN/Inf or invalid dt detected","repair the data contract before discovery"))
    elif X.ndim != 3 or U.ndim != 3 or X.shape[0] != U.shape[0] or X.shape[1] != U.shape[1]+1:
        checks.append(GateCheck("data_integrity","FAIL",{"X_shape":list(X.shape),"U_shape":list(U.shape)},"PublicDataset shapes are inconsistent","rebuild the trajectory-level dataset"))
    elif U.shape[1] < cfg.min_horizon:
        checks.append(GateCheck("data_integrity","FAIL",{"horizon":int(U.shape[1]),"min_horizon":cfg.min_horizon},"trajectory horizon is too short","collect longer continuous trials"))
    else:
        checks.append(GateCheck("data_integrity","PASS",{"n_trajectories":int(X.shape[0]),"horizon":int(U.shape[1]),"dt":float(ds.dt)},"trajectory-level data contract is valid",""))

    # If observed sample times are available, reject incompatible timebases
    # before derivative/excitation diagnostics assume equally spaced samples.
    if not any(c.status == "FAIL" for c in checks):
        checks.append(_timebase_integrity(ds,cfg))

    # Avoid numerical diagnostics on malformed or temporally invalid arrays.
    if any(c.status == "FAIL" for c in checks):
        states=[]
    else:
        checks.append(_hankel_excitation(ds,cfg))
        checks.append(_source_identifiability(ds,cfg))
        states=_state_derivative_probe(ds,cfg)
        checks.append(_temporal_sufficiency_check(ds,cfg))
        checks.append(_measurement_chain_check(ds,states,cfg))
        checks.append(_sampling_bandwidth_check(states,cfg))

    fails=[c for c in checks if c.status=="FAIL"] + [s for s in states if s.status=="FAIL"]
    warns=[c for c in checks if c.status=="WARN"] + [s for s in states if s.status=="WARN"]
    if fails:
        status="OBSERVATION_INSUFFICIENT"; can=False
    elif warns:
        status="CONDITIONALLY_QUALIFIED"; can=True
    else:
        status="QUALIFIED"; can=True
    state_blocked=[s.state for s in states if s.status == "FAIL"]
    contract_blocked=[]
    for c in checks:
        if c.name == "measurement_chain":
            contract_blocked.extend(c.metrics.get("resolution_insufficient_states", []))
        if c.name == "temporal_sufficiency":
            contract_blocked.extend(c.metrics.get("insufficient_states", []))
    blocked=[]
    for name in [*state_blocked,*contract_blocked]:
        if name not in blocked:
            blocked.append(name)
    usable=[s.state for s in states if s.state not in blocked]
    recs=[]
    for x in [*checks,*states]:
        r=getattr(x,"recommendation","")
        if r and r not in recs: recs.append(r)
    summary={
        "n_trajectories":int(X.shape[0]) if X.ndim>=1 else 0,
        "horizon":int(U.shape[1]) if U.ndim==3 else 0,
        "dt":float(ds.dt) if np.isfinite(ds.dt) else None,
        "n_states":int(X.shape[-1]) if X.ndim==3 else 0,
        "n_inputs":int(U.shape[-1]) if U.ndim==3 else 0,
        "fail_count":len(fails),
        "warn_count":len(warns),
        "blocked_state_count":len(blocked),
    }
    return ObservationGateReport(status,can,checks,states,usable,blocked,recs,asdict(cfg),summary)
