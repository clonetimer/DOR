from __future__ import annotations

"""Mechanism-aware, evidence-bounded automatic diagnostic pipeline design.

The runtime deliberately searches a finite and inspectable portfolio.  It does
not claim global NAS optimality and it refuses stages whose supervision or
identifiability contract is absent.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import csv
import hashlib
import io
import json
import math
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REQUEST_CONTRACT = "DOR-AUTO-DIAGNOSTIC-DESIGN-REQUEST/1.0"
DEVICE_CONTRACT = "DOR-DEVICE-MECHANISM-CARD/1.0"
MANIFEST_CONTRACT = "DOR-MULTIMODAL-DATASET-MANIFEST/1.0"
BUNDLE_CONTRACT = "DOR-DIAGNOSTIC-PIPELINE-BUNDLE/1.0"
STAGES = ("DETECTION", "LOCALIZATION", "CLASSIFICATION", "IDENTIFICATION")


def _json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: Any) -> Path:
    def safe(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(k): safe(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(v) for v in item]
        if isinstance(item, (float, np.floating)) and not math.isfinite(float(item)):
            return None
        if isinstance(item, np.generic):
            return item.item()
        return item
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(value))


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(map(str, values)))


@dataclass(frozen=True)
class StageQualification:
    stage: str
    status: str
    reason: str
    supervision: tuple[str, ...]
    primary_metric: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PCAReconstruction:
    def __init__(self, variance: float = 0.95):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=variance, svd_solver="full")

    def fit(self, x: np.ndarray) -> "PCAReconstruction":
        z = self.scaler.fit_transform(x)
        self.pca.fit(z)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        z = self.scaler.transform(x)
        zh = self.pca.inverse_transform(self.pca.transform(z))
        return np.mean(np.square(z - zh), axis=1)


class IsolationDetector:
    def __init__(self, seed: int):
        self.model = IsolationForest(n_estimators=200, contamination="auto", random_state=seed, n_jobs=1)

    def fit(self, x: np.ndarray) -> "IsolationDetector":
        self.model.fit(x)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        return -np.asarray(self.model.decision_function(x), dtype=float)


def _require_contract(value: Mapping[str, Any], expected: str, name: str) -> None:
    if value.get("contract") != expected:
        raise ValueError(f"{name}.contract must be {expected!r}")


def _validate_device_card(card: Mapping[str, Any]) -> None:
    _require_contract(card, DEVICE_CONTRACT, "device card")
    if not str(card.get("device_id", "")).strip():
        raise ValueError("device card requires device_id")
    sensors = card.get("sensors") or []
    ids = [str(x.get("sensor_id", "")) for x in sensors]
    if any(not x for x in ids) or len(ids) != len(set(ids)):
        raise ValueError("device card sensor_id values must be non-empty and unique")
    for edge in card.get("topology_edges") or []:
        if not str(edge.get("provenance", "")).strip():
            raise ValueError("every device-card topology edge requires provenance")


def _validate_request(request: Mapping[str, Any]) -> None:
    _require_contract(request, REQUEST_CONTRACT, "auto design request")
    stages = [str(x).upper() for x in request.get("stages") or STAGES]
    bad = sorted(set(stages) - set(STAGES))
    if bad:
        raise ValueError(f"unsupported stages: {bad}")
    budget = request.get("budget") or {}
    if int(budget.get("max_candidates", 24)) < 1:
        raise ValueError("budget.max_candidates must be positive")
    if float(budget.get("max_wall_seconds", 7200)) <= 0:
        raise ValueError("budget.max_wall_seconds must be positive")
    for key in ("max_parameters", "max_model_bytes"):
        if key in budget and int(budget[key]) < 1:
            raise ValueError(f"budget.{key} must be positive")
    if "max_inference_latency_ms" in budget and float(budget["max_inference_latency_ms"]) <= 0:
        raise ValueError("budget.max_inference_latency_ms must be positive")
    selection = request.get("selection") or {}
    for key in ("worst_slice_weight", "missing_sensor_weight"):
        value = float(selection.get(key, 0.0))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"selection.{key} must be in [0,1]")
    for key in ("min_worst_slice_score", "min_missing_sensor_score"):
        if key in selection and not 0.0 <= float(selection[key]) <= 1.0:
            raise ValueError(f"selection.{key} must be in [0,1]")
    if int((request.get("evaluation") or {}).get("min_slice_rows", 3)) < 1:
        raise ValueError("evaluation.min_slice_rows must be positive")
    seeds = (request.get("search") or {}).get("seeds")
    if seeds is not None and not seeds:
        raise ValueError("search.seeds must not be empty")
    robustness = request.get("robustness") or {}
    groups = robustness.get("sensor_groups") or {}
    if not isinstance(groups, Mapping) or any(not str(k).strip() or not v for k, v in groups.items()):
        raise ValueError("robustness.sensor_groups must map non-empty group names to feature lists")
    if int(robustness.get("max_missing_groups", 0)) < 0:
        raise ValueError("robustness.max_missing_groups must be non-negative")


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    _require_contract(manifest, MANIFEST_CONTRACT, "dataset manifest")
    modality = str(manifest.get("modality", "TABULAR")).upper()
    if modality not in {"TABULAR", "TIMESERIES_LONG", "TIMESERIES_NPZ", "MULTIRATE_NPZ"}:
        raise ValueError("manifest.modality must be TABULAR, TIMESERIES_LONG, TIMESERIES_NPZ, or MULTIRATE_NPZ")
    split = manifest.get("split") or {}
    if not split.get("column"):
        raise ValueError("manifest.split.column is required")
    for role in ("train_values", "validation_values"):
        if not split.get(role):
            raise ValueError(f"manifest.split.{role} is required")


def _series_features(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    if x.size < 2 or not np.all(np.isfinite(x)):
        raise ValueError("each time-series group/channel needs >=2 finite samples")
    t = np.arange(x.size, dtype=float)
    slope = float(np.polyfit(t, x, 1)[0])
    spectrum = np.square(np.abs(np.fft.rfft(x - np.mean(x))))
    bands = np.array_split(spectrum[1:], 4)
    result = {
        "mean": float(np.mean(x)), "std": float(np.std(x)), "min": float(np.min(x)),
        "max": float(np.max(x)), "median": float(np.median(x)),
        "rms": float(np.sqrt(np.mean(np.square(x)))), "slope": slope,
        "delta": float(x[-1] - x[0]),
    }
    for i, band in enumerate(bands):
        result[f"fft_band_{i}"] = float(np.mean(band)) if band.size else 0.0
    differences = np.diff(x)
    result["temporal__diff_std"] = float(np.std(differences))
    if np.std(x[:-1]) > 1e-12 and np.std(x[1:]) > 1e-12:
        result["temporal__autocorr_1"] = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    else:
        result["temporal__autocorr_1"] = 0.0
    for i, segment in enumerate(np.array_split(x, min(4, x.size))):
        result[f"temporal__segment_{i}_mean"] = float(np.mean(segment))
        result[f"temporal__segment_{i}_std"] = float(np.std(segment))
    return result


def _load_npz(manifest: Mapping[str, Any], data_path: Path, *, inference: bool = False) -> tuple[pd.DataFrame, list[str]]:
    cfg = manifest.get("timeseries") or {}
    array_key = str(cfg.get("array_key", "X"))
    with np.load(data_path, allow_pickle=False) as z:
        if array_key not in z:
            raise ValueError(f"NPZ lacks array {array_key!r}")
        x = np.asarray(z[array_key], dtype=float)
        if x.ndim != 3:
            raise ValueError("NPZ time-series array must have shape [sample,time,channel]")
        names = list(map(str, cfg.get("channel_names") or []))
        if len(names) != x.shape[2]:
            raise ValueError("timeseries.channel_names must match NPZ channel dimension")
        rows: list[dict[str, Any]] = []
        metadata = cfg.get("metadata_keys") or {}
        for i in range(x.shape[0]):
            row: dict[str, Any] = {}
            for output_name, key in metadata.items():
                if str(key) not in z:
                    if inference:
                        continue
                    raise ValueError(f"NPZ lacks metadata array {key!r}")
                row[str(output_name)] = np.asarray(z[str(key)])[i].item()
            for j, name in enumerate(names):
                for stat, value in _series_features(x[i, :, j]).items():
                    row[f"{name}__{stat}"] = value
            rows.append(row)
    features = [c for c in rows[0] if "__" in c] if rows else []
    return pd.DataFrame(rows), features


def _load_multirate_npz(manifest: Mapping[str, Any], data_path: Path, *, inference: bool = False) -> tuple[pd.DataFrame, list[str]]:
    cfg = manifest.get("timeseries") or {}
    sensor_arrays = {str(k): str(v) for k, v in (cfg.get("sensor_arrays") or {}).items()}
    if not sensor_arrays:
        raise ValueError("MULTIRATE_NPZ requires timeseries.sensor_arrays")
    with np.load(data_path, allow_pickle=False) as z:
        arrays: dict[str, np.ndarray] = {}
        n_samples: int | None = None
        for sensor, key in sensor_arrays.items():
            if key not in z:
                raise ValueError(f"NPZ lacks sensor array {key!r}")
            values = np.asarray(z[key], dtype=float)
            if values.ndim != 2:
                raise ValueError(f"multirate sensor {sensor!r} must have shape [sample,time]")
            if n_samples is not None and values.shape[0] != n_samples:
                raise ValueError("all multirate sensor arrays must have the same sample count")
            n_samples = values.shape[0]
            arrays[sensor] = values
        metadata: dict[str, np.ndarray] = {}
        for output_name, key in (cfg.get("metadata_keys") or {}).items():
            if str(key) not in z:
                if inference:
                    continue
                raise ValueError(f"NPZ lacks metadata array {key!r}")
            values = np.asarray(z[str(key)])
            if len(values) != n_samples:
                raise ValueError(f"metadata {key!r} sample count mismatch")
            metadata[str(output_name)] = values
        rows: list[dict[str, Any]] = []
        for i in range(int(n_samples or 0)):
            row = {name: values[i].item() for name, values in metadata.items()}
            for sensor, values in arrays.items():
                for stat, value in _series_features(values[i]).items():
                    row[f"{sensor}__{stat}"] = value
            rows.append(row)
    features = [c for c in rows[0] if "__" in c] if rows else []
    return pd.DataFrame(rows), features


def _load_frame(manifest: Mapping[str, Any], data_path: str | Path, *, inference: bool = False) -> tuple[pd.DataFrame, list[str]]:
    p = Path(data_path)
    modality = str(manifest.get("modality", "TABULAR")).upper()
    if modality == "TIMESERIES_NPZ":
        return _load_npz(manifest, p, inference=inference)
    if modality == "MULTIRATE_NPZ":
        return _load_multirate_npz(manifest, p, inference=inference)
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    if df.empty:
        raise ValueError("dataset is empty")
    if modality == "TABULAR":
        features = _unique((manifest.get("features") or {}).get("raw_columns") or [])
        if not features:
            raise ValueError("tabular manifest requires features.raw_columns")
        return df, features

    cfg = manifest.get("timeseries") or {}
    group_cols = _unique(cfg.get("group_columns") or manifest.get("identifiers") or [])
    channel_cols = _unique(cfg.get("channel_columns") or [])
    time_col = str(cfg.get("time_column", ""))
    if not group_cols or not channel_cols or not time_col:
        raise ValueError("TIMESERIES_LONG requires group_columns, channel_columns and time_column")
    split_col = str((manifest.get("split") or {}).get("column"))
    labels = _declared_supervision_columns(manifest)
    evaluation_cols = list(map(str, (manifest.get("evaluation") or {}).get("slice_columns") or []))
    metadata_cols = [] if inference else [split_col] + labels + evaluation_cols
    required = group_cols + channel_cols + [time_col] + metadata_cols
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"time-series data missing columns: {missing}")
    rows: list[dict[str, Any]] = []
    feature_names: list[str] = []
    for keys, part in df.sort_values(group_cols + [time_col]).groupby(group_cols, sort=False, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, key_values))
        for meta in _unique(metadata_cols):
            vals = pd.unique(part[meta])
            if len(vals) != 1:
                raise ValueError(f"time-series group has non-constant metadata column {meta!r}")
            row[meta] = vals[0]
        for channel in channel_cols:
            for stat, value in _series_features(part[channel].to_numpy(dtype=float)).items():
                name = f"{channel}__{stat}"
                row[name] = value
                feature_names.append(name)
        rows.append(row)
    return pd.DataFrame(rows), _unique(feature_names)


def _declared_supervision_columns(manifest: Mapping[str, Any]) -> list[str]:
    sup = manifest.get("supervision") or {}
    result: list[str] = []
    for stage in STAGES:
        cfg = sup.get(stage.lower()) or {}
        if cfg.get("label_column"):
            result.append(str(cfg["label_column"]))
        result.extend(map(str, cfg.get("target_columns") or []))
    return _unique(result)


def _compile_mechanism_features(df: pd.DataFrame, card: Mapping[str, Any]) -> tuple[pd.DataFrame, list[str], list[dict[str, Any]]]:
    out = df.copy()
    generated: list[str] = []
    records: list[dict[str, Any]] = []
    for spec in card.get("mechanism_features") or []:
        name = str(spec.get("feature_id", "")).strip()
        op = str(spec.get("operation", "")).lower()
        cols = list(map(str, spec.get("source_columns") or []))
        if not name or name in out.columns:
            raise ValueError(f"invalid or duplicate mechanism feature_id {name!r}")
        missing = [c for c in cols if c not in out.columns]
        if missing:
            raise ValueError(f"mechanism feature {name!r} missing sources: {missing}")
        if op == "difference" and len(cols) == 2:
            value = out[cols[0]] - out[cols[1]]
        elif op == "ratio" and len(cols) == 2:
            eps = float(spec.get("epsilon", 1e-12))
            value = out[cols[0]] / out[cols[1]].where(out[cols[1]].abs() > eps, np.nan)
        elif op == "sum" and cols:
            value = out[cols].sum(axis=1)
        elif op == "product" and cols:
            value = out[cols].prod(axis=1)
        elif op == "linear_residual" and cols:
            coeff = list(map(float, spec.get("coefficients") or []))
            if len(coeff) != len(cols):
                raise ValueError(f"mechanism feature {name!r} coefficient count mismatch")
            value = sum(c * out[col] for c, col in zip(coeff, cols)) - float(spec.get("expected", 0.0))
        else:
            raise ValueError(f"unsupported mechanism operation for {name!r}: {op}")
        out[name] = value
        if not np.all(np.isfinite(pd.to_numeric(out[name], errors="coerce"))):
            raise ValueError(f"mechanism feature {name!r} produced non-finite values")
        generated.append(name)
        records.append({
            "feature_id": name, "operation": op, "source_columns": cols,
            "provenance": spec.get("provenance"),
            "information_claim": "DETERMINISTIC_REPRESENTATION_NOT_NEW_OBSERVATION",
        })
    return out, generated, records


def _role_masks(df: pd.DataFrame, manifest: Mapping[str, Any]) -> dict[str, np.ndarray]:
    split = manifest.get("split") or {}
    col = str(split["column"])
    if col not in df.columns:
        raise ValueError(f"split column {col!r} is missing")
    values = df[col].astype(str).to_numpy()
    masks = {
        role: np.isin(values, np.asarray(list(map(str, split.get(f"{role}_values") or [])), dtype=object))
        for role in ("train", "validation", "calibration", "test")
    }
    assigned = sum(mask.astype(int) for mask in masks.values())
    if np.any(assigned > 1):
        raise ValueError("split roles overlap")
    if not masks["train"].any() or not masks["validation"].any():
        raise ValueError("train and validation roles must be non-empty")
    for group_col in map(str, (manifest.get("leakage_guard") or {}).get("group_columns") or []):
        if group_col not in df.columns:
            raise ValueError(f"leakage guard column {group_col!r} is missing")
        role = np.full(len(df), "UNASSIGNED", dtype=object)
        for key, mask in masks.items():
            role[mask] = key
        cross = pd.DataFrame({"g": df[group_col].astype(str), "r": role}).query("r != 'UNASSIGNED'").groupby("g")["r"].nunique()
        if (cross > 1).any():
            raise ValueError(f"leakage guard violation: {int((cross > 1).sum())} {group_col!r} groups cross roles")
    return masks


def _qualify_stages(df: pd.DataFrame, manifest: Mapping[str, Any], request: Mapping[str, Any], masks: Mapping[str, np.ndarray]) -> list[StageQualification]:
    requested = {str(x).upper() for x in request.get("stages") or STAGES}
    sup = manifest.get("supervision") or {}
    output: list[StageQualification] = []
    for stage in STAGES:
        if stage not in requested:
            output.append(StageQualification(stage, "SKIPPED", "stage not requested", (), ""))
            continue
        cfg = sup.get(stage.lower()) or {}
        metric = str(((request.get("metrics") or {}).get(stage.lower()) or {}).get("primary", ""))
        if stage == "DETECTION":
            col = cfg.get("label_column")
            normal = cfg.get("normal_label")
            if col and str(col) in df.columns and normal is not None:
                yv = df.loc[masks["validation"], str(col)].astype(str)
                if yv.nunique() >= 2:
                    output.append(StageQualification(stage, "QUALIFIED", "normal-only fitting and labelled validation are available", (str(col),), metric or "average_precision"))
                else:
                    output.append(StageQualification(stage, "PORTFOLIO_ONLY", "validation lacks both normal and abnormal labels", (str(col),), metric or "average_precision"))
            else:
                output.append(StageQualification(stage, "PORTFOLIO_ONLY", "no detection labels; only normal-data detector portfolio can be retained", (), metric or "average_precision"))
        elif stage in {"LOCALIZATION", "CLASSIFICATION"}:
            multi_targets = tuple(map(str, cfg.get("target_columns") or [])) if stage == "LOCALIZATION" else ()
            col = cfg.get("label_column")
            missing_targets = [c for c in multi_targets if c not in df.columns]
            if multi_targets and not missing_targets:
                train_y = df.loc[masks["train"], list(multi_targets)].to_numpy(dtype=int)
                valid_values = set(np.unique(train_y).tolist()) <= {0, 1}
                varying = np.ptp(train_y, axis=0) > 0
                if not valid_values or not varying.all():
                    output.append(StageQualification(stage, "UNSUPPORTED", "multi-label localization targets must be binary and vary in training", multi_targets, metric or "macro_f1"))
                else:
                    output.append(StageQualification(stage, "QUALIFIED", "independent binary component labels support concurrent localization", multi_targets, metric or "macro_f1"))
            elif multi_targets:
                output.append(StageQualification(stage, "UNSUPPORTED", f"multi-label target columns unavailable: {missing_targets}", multi_targets, metric or "macro_f1"))
            elif not col or str(col) not in df.columns:
                output.append(StageQualification(stage, "UNSUPPORTED", "required label column is unavailable; pseudo-labels are prohibited", (), metric or "macro_f1"))
            elif df.loc[masks["train"], str(col)].astype(str).nunique() < 2:
                output.append(StageQualification(stage, "UNSUPPORTED", "training split has fewer than two classes", (str(col),), metric or "macro_f1"))
            else:
                unseen = set(df.loc[masks["validation"], str(col)].astype(str)) - set(df.loc[masks["train"], str(col)].astype(str))
                status = "UNSUPPORTED" if unseen else "QUALIFIED"
                reason = f"validation contains unseen classes: {sorted(unseen)}" if unseen else "independent train/validation labels are available"
                output.append(StageQualification(stage, status, reason, (str(col),), metric or "macro_f1"))
        else:
            targets = tuple(map(str, cfg.get("target_columns") or []))
            missing = [c for c in targets if c not in df.columns]
            if not targets or missing:
                output.append(StageQualification(stage, "UNSUPPORTED", f"parameter-change targets unavailable: {missing or 'not declared'}", targets, metric or "normalized_mae"))
            else:
                y = df.loc[masks["train"], list(targets)].to_numpy(dtype=float)
                rank = int(np.linalg.matrix_rank(y - np.mean(y, axis=0, keepdims=True)))
                status = "QUALIFIED" if rank >= len(targets) else "UNSUPPORTED"
                reason = "parameter targets have sufficient observed variation" if status == "QUALIFIED" else f"identification target rank {rank} < {len(targets)}"
                output.append(StageQualification(stage, status, reason, targets, metric or "normalized_mae"))
    return output


def _views(raw: Sequence[str], mechanism: Sequence[str], request: Mapping[str, Any]) -> dict[str, list[str]]:
    result = {"RAW": list(raw)}
    if bool((request.get("representations") or {}).get("temporal_pyramid", False)):
        temporal = [x for x in raw if "__temporal__" in x]
        summary = [x for x in raw if "__temporal__" not in x]
        if temporal and summary:
            result["RAW_SUMMARY"] = summary
            result["RAW_TEMPORAL_PYRAMID"] = temporal
    if mechanism:
        result["MECHANISM"] = list(mechanism)
        result["RAW_PLUS_MECHANISM"] = _unique(list(raw) + list(mechanism))
    return result


def _classifier(family: str, seed: int):
    if family == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))
    if family == "random_forest":
        return RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=1)
    if family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=150, random_state=seed)
    if family == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True, random_state=seed))
    raise ValueError(family)


def _regressor(family: str, seed: int):
    if family == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if family == "random_forest_regressor":
        return RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1)
    raise ValueError(family)


def _families(stage: str, request: Mapping[str, Any]) -> list[str]:
    defaults = {
        "DETECTION": ["pca_reconstruction", "isolation_forest"],
        "LOCALIZATION": ["logistic_regression", "random_forest", "hist_gradient_boosting"],
        "CLASSIFICATION": ["logistic_regression", "random_forest", "hist_gradient_boosting", "mlp"],
        "IDENTIFICATION": ["ridge", "random_forest_regressor"],
    }
    requested = (((request.get("candidates") or {}).get(stage.lower()) or {}).get("families"))
    values = list(map(str, requested or defaults[stage]))
    unknown = sorted(set(values) - set(defaults[stage]))
    if unknown:
        raise ValueError(f"unsupported {stage} families: {unknown}")
    return values


def _metric_detection(y: np.ndarray, score: np.ndarray, normal: str, primary: str) -> float:
    target = (np.asarray(y).astype(str) != str(normal)).astype(int)
    if len(np.unique(target)) < 2:
        return float("nan")
    if primary == "roc_auc":
        return float(roc_auc_score(target, score))
    return float(average_precision_score(target, score))


def _score_candidate(stage: str, model: Any, x: np.ndarray, y: np.ndarray, cfg: Mapping[str, Any], metric: str) -> float:
    if stage == "DETECTION":
        return _metric_detection(y, model.score(x), str(cfg.get("normal_label")), metric)
    if stage in {"LOCALIZATION", "CLASSIFICATION"}:
        if np.asarray(y).ndim == 2:
            return float(f1_score(np.asarray(y, dtype=int), np.asarray(model.predict(x), dtype=int), average="macro", zero_division=0))
        return float(f1_score(y.astype(str), model.predict(x).astype(str), average="macro", zero_division=0))
    pred = np.asarray(model.predict(x), dtype=float)
    truth = np.asarray(y, dtype=float)
    if pred.ndim == 1:
        pred = pred[:, None]
    if truth.ndim == 1:
        truth = truth[:, None]
    scale = np.std(truth, axis=0)
    valid = scale > 1e-12
    if not valid.any():
        return float("nan")
    return float(1.0 / (1.0 + np.mean(np.abs(pred[:, valid] - truth[:, valid]) / scale[valid])))


def _fit_model(stage: str, family: str, seed: int, x: np.ndarray, y: np.ndarray | None, cfg: Mapping[str, Any]):
    if stage == "DETECTION":
        normal = str(cfg.get("normal_label"))
        fit_x = x if y is None else x[np.asarray(y).astype(str) == normal]
        if len(fit_x) < 3:
            raise ValueError("detection requires at least three trusted-normal training rows")
        model = PCAReconstruction() if family == "pca_reconstruction" else IsolationDetector(seed)
        return model.fit(fit_x)
    if stage in {"LOCALIZATION", "CLASSIFICATION"}:
        if stage == "LOCALIZATION" and np.asarray(y).ndim == 2:
            return OneVsRestClassifier(_classifier(family, seed)).fit(x, np.asarray(y, dtype=int))
        return _classifier(family, seed).fit(x, np.asarray(y).astype(str))
    target = np.asarray(y, dtype=float)
    if target.ndim == 2 and target.shape[1] == 1:
        target = target[:, 0]
    return _regressor(family, seed).fit(x, target)


def _model_parameter_estimate(model: Any) -> int:
    """Return an auditable structural-size proxy, not a neural parameter claim."""
    seen: set[int] = set()

    def count(value: Any) -> int:
        if id(value) in seen:
            return 0
        seen.add(id(value))
        total = 0
        tree = getattr(value, "tree_", None)
        if tree is not None:
            total += int(tree.node_count)
        for name in ("coef_", "intercept_", "components_", "mean_", "scale_", "var_", "thresholds_"):
            array = getattr(value, name, None)
            if array is not None:
                total += int(np.asarray(array).size)
        for child in getattr(value, "steps", []):
            total += count(child[1])
        for name in ("estimators_",):
            children = getattr(value, name, None)
            if children is not None:
                total += sum(count(child) for child in np.asarray(children, dtype=object).ravel())
        for name in ("model", "scaler", "pca"):
            child = getattr(value, name, None)
            if child is not None:
                total += count(child)
        return total

    return max(1, count(model))


def _resource_profile(stage: str, model: Any, x: np.ndarray) -> dict[str, Any]:
    stream = io.BytesIO()
    joblib.dump(model, stream)

    def infer() -> None:
        if stage == "DETECTION":
            model.score(x)
        else:
            model.predict(x)

    timings: list[float] = []
    for _ in range(3):
        tick = time.perf_counter()
        infer()
        timings.append((time.perf_counter() - tick) * 1000.0 / max(1, len(x)))
    return {
        "model_bytes": len(stream.getvalue()),
        "parameter_estimate": _model_parameter_estimate(model),
        "parameter_estimate_semantics": "stored numeric coefficients plus tree nodes; cross-family structural-size proxy",
        "inference_latency_ms_per_row": float(np.median(timings)),
        "latency_batch_rows": int(len(x)),
        "latency_repeats": 3,
    }


def _resolve_sensor_groups(
    raw_features: Sequence[str], mechanism_records: Sequence[Mapping[str, Any]], request: Mapping[str, Any]
) -> dict[str, list[str]]:
    configured = (request.get("robustness") or {}).get("sensor_groups") or {}
    result: dict[str, list[str]] = {}
    assigned: dict[str, str] = {}
    for group, patterns in configured.items():
        resolved: list[str] = []
        for pattern in map(str, patterns):
            hits = [c for c in raw_features if c == pattern or c.startswith(f"{pattern}__")]
            if not hits:
                raise ValueError(f"robustness sensor group {group!r} pattern {pattern!r} matches no raw feature")
            resolved.extend(hits)
        base = _unique(resolved)
        for column in base:
            if column in assigned:
                raise ValueError(f"raw feature {column!r} belongs to multiple sensor groups")
            assigned[column] = str(group)
        affected = set(base)
        changed = True
        while changed:
            changed = False
            for rec in mechanism_records:
                if set(map(str, rec.get("source_columns") or [])) & affected and str(rec["feature_id"]) not in affected:
                    affected.add(str(rec["feature_id"])); changed = True
        result[str(group)] = sorted(affected)
    return result


def _slice_scores(
    stage: str, model: Any, x: np.ndarray, y: np.ndarray, cfg: Mapping[str, Any], metric: str,
    metadata: pd.DataFrame, columns: Sequence[str], min_rows: int,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for column in columns:
        for value in sorted(metadata[column].astype(str).unique()):
            mask = metadata[column].astype(str).to_numpy() == value
            if int(mask.sum()) < min_rows:
                continue
            score = _score_candidate(stage, model, x[mask], y[mask], cfg, metric)
            if np.isfinite(score):
                scores[f"{column}={value}"] = float(score)
    return scores


def _evaluate_candidate(
    *, stage: str, candidate_id: str, view: str, family: str, seed: int, rung: str,
    model: Any, x_validation: np.ndarray, y_validation: np.ndarray | None,
    stage_cfg: Mapping[str, Any], metric: str, validation_metadata: pd.DataFrame,
    slice_columns: Sequence[str], sensor_groups: Mapping[str, Sequence[str]],
    feature_columns: Sequence[str], feature_fallbacks: Mapping[str, float], request: Mapping[str, Any],
    fit_seconds: float,
) -> dict[str, Any]:
    score = float("nan") if y_validation is None else _score_candidate(stage, model, x_validation, y_validation, stage_cfg, metric)
    min_rows = int((request.get("evaluation") or {}).get("min_slice_rows", 3))
    slices = {} if y_validation is None else _slice_scores(
        stage, model, x_validation, y_validation, stage_cfg, metric,
        validation_metadata, slice_columns, min_rows,
    )
    sensor_scores: dict[str, float] = {}
    if y_validation is not None:
        for group, affected in sensor_groups.items():
            indices = [i for i, column in enumerate(feature_columns) if column in affected]
            if not indices:
                continue
            stressed = x_validation.copy()
            for index in indices:
                stressed[:, index] = float(feature_fallbacks[feature_columns[index]])
            value = _score_candidate(stage, model, stressed, y_validation, stage_cfg, metric)
            if np.isfinite(value):
                sensor_scores[str(group)] = float(value)
    worst_slice = min(slices.values()) if slices else None
    worst_sensor = min(sensor_scores.values()) if sensor_scores else None
    selection = request.get("selection") or {}
    utility = score
    if np.isfinite(utility) and worst_slice is not None:
        utility -= float(selection.get("worst_slice_weight", 0.0)) * max(0.0, score - worst_slice)
    if np.isfinite(utility) and worst_sensor is not None:
        utility -= float(selection.get("missing_sensor_weight", 0.0)) * max(0.0, score - worst_sensor)
    resources = _resource_profile(stage, model, x_validation)
    budget = request.get("budget") or {}
    violations: list[str] = []
    if resources["model_bytes"] > int(budget.get("max_model_bytes", resources["model_bytes"])):
        violations.append("max_model_bytes")
    if resources["parameter_estimate"] > int(budget.get("max_parameters", resources["parameter_estimate"])):
        violations.append("max_parameters")
    if resources["inference_latency_ms_per_row"] > float(budget.get("max_inference_latency_ms", resources["inference_latency_ms_per_row"])):
        violations.append("max_inference_latency_ms")
    if "min_worst_slice_score" in selection and (worst_slice is None or worst_slice < float(selection["min_worst_slice_score"])):
        violations.append("min_worst_slice_score")
    if "min_missing_sensor_score" in selection and (worst_sensor is None or worst_sensor < float(selection["min_missing_sensor_score"])):
        violations.append("min_missing_sensor_score")
    return {
        "stage": stage, "candidate_id": candidate_id, "view": view, "family": family,
        "seed": seed, "score": score, "selection_score": utility,
        "primary_metric": metric, "rung": rung, "fit_seconds": fit_seconds,
        "slice_scores": slices, "worst_slice_score": worst_slice,
        "missing_sensor_scores": sensor_scores, "worst_missing_sensor_score": worst_sensor,
        "resources": resources, "eligible": not violations, "constraint_violations": violations,
    }


def _stage_target(df: pd.DataFrame, stage: str, cfg: Mapping[str, Any]) -> np.ndarray | None:
    if stage == "DETECTION" and not cfg.get("label_column"):
        return None
    if stage == "LOCALIZATION" and cfg.get("target_columns"):
        return df[list(map(str, cfg["target_columns"]))].to_numpy(dtype=int)
    if stage in {"DETECTION", "LOCALIZATION", "CLASSIFICATION"}:
        return df[str(cfg["label_column"])].astype(str).to_numpy()
    return df[list(map(str, cfg.get("target_columns") or []))].to_numpy(dtype=float)


def _select_stage(stage: str, records: list[dict[str, Any]], practical_margin: float, stability: float, labelled: bool) -> dict[str, Any]:
    by_id: dict[str, list[float]] = {}
    rejected: dict[str, list[str]] = {}
    all_ids = {str(row["candidate_id"]) for row in records}
    for row in records:
        cid = str(row["candidate_id"])
        if not bool(row.get("eligible", True)):
            rejected.setdefault(cid, []).extend(map(str, row.get("constraint_violations") or []))
    for row in records:
        cid = str(row["candidate_id"])
        if cid in rejected:
            continue
        by_id.setdefault(cid, []).append(float(row.get("selection_score", row["score"])))
    medians = {k: float(np.nanmedian(v)) for k, v in by_id.items() if np.isfinite(v).any()}
    if not medians:
        eligible_ids = sorted(by_id)
        if not eligible_ids and all_ids:
            return {"status": "UNSUPPORTED", "selected_candidate_id": None, "retained_candidate_ids": [], "candidate_median_scores": {}, "rejected_candidates": {k: sorted(set(v)) for k, v in rejected.items()}, "reason": "all candidates violated declared deployment or robustness constraints"}
        return {"status": "PORTFOLIO" if stage == "DETECTION" and not labelled else "UNSUPPORTED", "selected_candidate_id": None, "retained_candidate_ids": eligible_ids, "candidate_median_scores": {}, "rejected_candidates": {k: sorted(set(v)) for k, v in rejected.items()}, "reason": "no labelled validation metric available"}
    order = sorted(medians, key=medians.get, reverse=True)
    if len(order) == 1:
        return {"status": "FIXED", "selected_candidate_id": order[0], "retained_candidate_ids": [order[0]], "candidate_median_scores": medians, "rejected_candidates": {k: sorted(set(v)) for k, v in rejected.items()}, "reason": "only one constraint-qualified candidate"}
    best, runner = order[:2]
    wins = [a - b >= practical_margin for a in by_id[best] for b in by_id[runner] if np.isfinite(a) and np.isfinite(b)]
    stable = float(np.mean(wins)) if wins else 0.0
    if medians[best] - medians[runner] >= practical_margin and stable >= stability:
        return {"status": "SELECTED", "selected_candidate_id": best, "retained_candidate_ids": [best], "candidate_median_scores": medians, "rejected_candidates": {k: sorted(set(v)) for k, v in rejected.items()}, "reason": f"constraint-qualified utility margin and restart stability passed ({stable:.3f})"}
    retained = [cid for cid in order if medians[best] - medians[cid] < practical_margin]
    return {"status": "ALTERNATIVES", "selected_candidate_id": None, "retained_candidate_ids": retained, "candidate_median_scores": medians, "rejected_candidates": {k: sorted(set(v)) for k, v in rejected.items()}, "reason": "constraint-qualified validation evidence does not distinguish a stable practical winner"}


def _pipeline_dag(qualifications: Sequence[StageQualification]) -> dict[str, Any]:
    supported = [q.stage for q in qualifications if q.status in {"QUALIFIED", "PORTFOLIO_ONLY"}]
    nodes = [{"stage": s, "conditional_on": "DETECTION:ABNORMAL" if s != "DETECTION" and "DETECTION" in supported else None} for s in supported]
    edges = []
    for a, b in zip(supported, supported[1:]):
        edges.append({"from": a, "to": b, "type": "CONDITIONAL" if a == "DETECTION" else "SERIAL"})
    return {"topology": "CONDITIONAL_CASCADE", "nodes": nodes, "edges": edges, "teacher_forced_evaluation_prohibited": True}


def _mechanism_stress_audit(
    frame: pd.DataFrame, masks: Mapping[str, np.ndarray], manifest: Mapping[str, Any],
    request: Mapping[str, Any], models: Sequence[Mapping[str, Any]], mechanism_features: Sequence[str],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    sup = manifest.get("supervision") or {}
    rng = np.random.default_rng(20260813)
    for rec in models:
        affected = [c for c in rec["feature_columns"] if c in mechanism_features]
        if not affected:
            continue
        stage = str(rec["stage"]); cfg = sup.get(stage.lower()) or {}
        y = _stage_target(frame, stage, cfg)
        if y is None:
            continue
        xtr = frame.loc[masks["train"], rec["feature_columns"]].to_numpy(dtype=float)
        xv = frame.loc[masks["validation"], rec["feature_columns"]].to_numpy(dtype=float)
        model = _fit_model(stage, str(rec["family"]), 20260813, xtr, y[masks["train"]], cfg)
        metric = next((m for m in (request.get("metrics") or {}).get(stage.lower(), {}).values()), None)
        if not metric:
            metric = "average_precision" if stage == "DETECTION" else ("macro_f1" if stage != "IDENTIFICATION" else "normalized_mae")
        baseline = _score_candidate(stage, model, xv, y[masks["validation"]], cfg, str(metric))
        corrupted = xv.copy()
        for column in affected:
            j = rec["feature_columns"].index(column)
            corrupted[:, j] = rng.permutation(corrupted[:, j])
        shuffled = _score_candidate(stage, model, corrupted, y[masks["validation"]], cfg, str(metric))
        records.append({
            "candidate_id": rec["candidate_id"], "stage": stage, "stress": "SHUFFLED_MECHANISM_FEATURES",
            "affected_features": affected, "baseline_score": baseline, "stressed_score": shuffled,
            "score_delta": baseline - shuffled, "selection_uses_stress_result": False,
        })
    return {"contract": "DOR-MECHANISM-STRESS-AUDIT/1.0", "state": "RUN" if records else "NOT_APPLICABLE", "records": records}


def _test_audit(
    frame: pd.DataFrame, masks: Mapping[str, np.ndarray], manifest: Mapping[str, Any],
    bundle_root: Path, models: Sequence[Mapping[str, Any]], alpha: float,
) -> dict[str, Any]:
    if not masks["test"].any():
        return {"contract": "DOR-END-TO-END-TEST-AUDIT/1.0", "state": "NOT_RUN", "reason": "no test split", "selection_uses_test": False, "records": []}
    sup = manifest.get("supervision") or {}
    test_rows = np.flatnonzero(masks["test"])
    gate = np.full(len(test_rows), True)
    detection_states = []
    records: list[dict[str, Any]] = []
    for rec in models:
        if rec["stage"] != "DETECTION" or rec.get("decision_threshold") is None:
            continue
        model = joblib.load(bundle_root / rec["artifact"])
        x = frame.iloc[test_rows][rec["feature_columns"]].to_numpy(dtype=float)
        score = model.score(x); state = score > float(rec["decision_threshold"])
        detection_states.append(state)
        cfg = sup.get("detection") or {}; y = _stage_target(frame, "DETECTION", cfg)
        metric = _metric_detection(y[test_rows], score, str(cfg.get("normal_label")), "average_precision") if y is not None else float("nan")
        records.append({"candidate_id": rec["candidate_id"], "stage": "DETECTION", "average_precision": metric})
    if detection_states:
        gate = np.all(np.stack(detection_states, axis=1), axis=1)
    for rec in models:
        stage = str(rec["stage"])
        if stage == "DETECTION":
            continue
        cfg = sup.get(stage.lower()) or {}; y = _stage_target(frame, stage, cfg)
        if y is None:
            continue
        model = joblib.load(bundle_root / rec["artifact"])
        x = frame.iloc[test_rows][rec["feature_columns"]].to_numpy(dtype=float)
        evaluated = int(gate.sum())
        row: dict[str, Any] = {"candidate_id": rec["candidate_id"], "stage": stage, "test_rows": len(test_rows), "executed_rows": evaluated, "teacher_forced": False}
        if evaluated:
            pred = np.asarray(model.predict(x[gate]))
            truth = np.asarray(y[test_rows][gate])
            if stage in {"LOCALIZATION", "CLASSIFICATION"}:
                row["macro_f1_on_executed"] = float(f1_score(truth, pred, average="macro", zero_division=0))
            else:
                if pred.ndim == 1: pred = pred[:, None]
                if truth.ndim == 1: truth = truth[:, None]
                row["mae_on_executed"] = float(mean_absolute_error(truth, pred))
                radii = rec.get("absolute_error_quantiles") or []
                if len(radii) == truth.shape[1]:
                    radius = np.asarray(radii)[None, :]
                    row["interval_joint_coverage_on_executed"] = float(np.mean(np.all(np.abs(truth - pred) <= radius, axis=1)))
                    row["interval_alpha"] = alpha
        records.append(row)
    return {"contract": "DOR-END-TO-END-TEST-AUDIT/1.0", "state": "RUN", "selection_uses_test": False, "gate_rule": "all retained detectors abnormal", "records": records}


def _audit_markdown(summary: Mapping[str, Any]) -> str:
    qrows = "\n".join(f"| {q['stage']} | {q['status']} | {q['reason']} |" for q in summary["qualification"]["stages"])
    return f"""# Auto Diagnostic Architect 审计报告

## 1. 问题合同与结论

- 设备：`{summary['device_id']}`
- 数据模态：`{summary['data']['modality']}`
- 结论边界：这里只在声明预算、候选集合和 validation 合同内选择；不是全局 NAS 最优证明。
- 数据角色：train 用于拟合，validation 用于选择，calibration/test 不参与选择。

| 阶段 | 资格 | 原因 |
|---|---|---|
{qrows}

## 2. 组件机制卡

### 时序统计/频域/时间金字塔编码

- 一句话机制：将 `[B,T,C]` 的有限窗口压缩为每通道统计量、频谱能量、差分/自相关和分段时间统计。
- 作用轴：时间轴；时间金字塔保留粗粒度次序，但压缩仍不可逆，会丢失精确相位和事件次序。
- 创新类型：数据与表示方案；不提供新事实观测。
- 失败模式：短瞬态、非平稳相位关系和跨窗口依赖可能丢失。

### 机理特征分支

- 一句话机制：用有来源的差分、比值、乘积或线性残差重编码已有观测。
- 作用轴：特征轴；不增加新证据。
- 创新类型：架构组合/表示方案。
- 机制破坏：打乱来源列或移除机理分支后复测。

### 条件级联 DAG

- 一句话机制：检测阶段控制后续定位、分类和辨识的条件执行。
- 风险：上游漏检会截断下游；端到端验收禁止 teacher forcing。
- 创新类型：既有组件的架构组合。

## 3. 公平比较矩阵

所有候选使用相同 train/validation 行、随机种子集合和选择指标。RAW、MECHANISM、RAW_PLUS_MECHANISM 分开登记；候选同时记录最差工况、缺失传感器、模型字节、结构复杂度代理和开发机批量时延。目标设备端到端时延、内存和能耗仍须现场补测。

## 4. 最终架构方案

```json
{json.dumps(summary['pipeline_dag'], ensure_ascii=False, indent=2)}
```

阶段选择：

```json
{json.dumps(summary['stage_selections'], ensure_ascii=False, indent=2)}
```

## 5. 实验计划

必须执行 B0 原始基线、B1 数据/训练修正版、S0 低复杂度、A0 完整方案、A1 移除机理分支、C0 参数匹配、C1 延迟匹配、M0 打乱机理映射、H0 机制假设切片、R0 缺失传感器/工况漂移、V0 多种子。

## 6. 技术报告扫描

本实现没有“新组件”或“首次提出”声明。公开证据扫描覆盖 NASA ADAPT 原始 benchmark、ROCKET/MiniRocket、InceptionTime 与多标签 conformal prediction；实际变化属于成熟统计/机器学习组件的组合、数据表示、校准和工程编排，因此禁止作新颖性结论。来源与适用边界见仓库 `docs/PUBLIC_DATASET_AND_LITERATURE_GAP_AUDIT_CN.md`。

## 7. 证据等级与不可宣称内容

- 数学/代码事实：数据角色隔离、有限候选、确定性机理特征、条件 DAG。
- 实验事实：仅限 `validation_results.json` 中当前数据与种子的结果。
- 不可宣称：全局最优、机理特征等于新证据、定位等于根因、辨识参数天然可解释、测试集支持尚未运行的机制因果叙事。

## 8. 资源使用追踪

| 资源 | 影响 |
|---|---|
| README | 从任务失败与信息缺口出发 |
| core-principles | 数据合同、收益拆分和证据边界 |
| component-ontology | 将新增内容归类为表示、架构组合、学习与系统层 |
| research-protocol | 形成无新组件创新声明 |
| design-validation-protocol | 固化 B0–V0 和停止条件 |
| learning-protocol | 对时序压缩、机理融合与条件路由采用 L3 审计 |
| full-design-principles | 检查 shape、信息损失、梯度、复杂度和部署一致性 |
| component-card | 形成第2节组件卡 |
| comparison-matrix | 形成第3节公平比较 |
| architecture-proposal | 形成第1、4节架构方案 |
| experiment-plan | 形成第5节实验计划 |
| technical-report-scan | 形成第6节技术扫描 |
| component-registry.csv | 生成独立组件登记表 |
| learning log v45 | 只决定 L3 分析深度，不作为有效性证据 |
"""


def _write_component_registry(path: Path) -> None:
    fields = [
        "component_id", "canonical_name", "aliases", "component_family", "architecture_level",
        "parent_component", "nearest_prior_art", "first_public_date", "original_source",
        "computation_delta", "input_output", "action_axis", "interaction_range", "state_rule",
        "inductive_bias", "expressive_boundary", "optimization_effect", "parameter_complexity",
        "compute_complexity", "memory_cache", "hardware_behavior", "strengths", "limitations",
        "failure_modes", "representative_models", "company_reports", "innovation_class",
        "evidence_level", "adoption_level", "source_urls", "last_verified_date", "notes",
    ]
    rows = [
        {"component_id": "DOR-AUTO-TS-FEATURES", "canonical_name": "Time-series statistical/frequency/temporal-pyramid encoding", "component_family": "data_representation", "architecture_level": "data", "parent_component": "window feature extraction", "nearest_prior_art": "statistical condition-monitoring and temporal pyramid features", "computation_delta": "fixed summary, FFT-band, difference, autocorrelation and segment encoding", "input_output": "[B,T,C] -> [B,F]", "action_axis": "time", "interaction_range": "per-window", "inductive_bias": "finite-window summaries with coarse order", "expressive_boundary": "exact phase and event order are lost", "failure_modes": "short transient loss; window sensitivity", "innovation_class": "data_or_representation", "evidence_level": "E1", "adoption_level": "A4", "last_verified_date": "2026-08-13", "notes": "No novelty claim; not MiniRocket"},
        {"component_id": "DOR-AUTO-MECH-VIEW", "canonical_name": "Mechanism feature view", "component_family": "multimodal_fusion", "architecture_level": "data/topology", "parent_component": "engineered feature fusion", "nearest_prior_art": "physics-informed feature engineering", "computation_delta": "raw/mechanism/combined candidate views", "input_output": "[B,F]+[B,M] -> candidate views", "action_axis": "feature", "interaction_range": "per-sample", "inductive_bias": "declared physical relations", "expressive_boundary": "cannot recover unobserved facts", "failure_modes": "wrong provenance; unstable ratio; domain shift", "innovation_class": "architecture_combination", "evidence_level": "E1", "adoption_level": "A3", "last_verified_date": "2026-08-12", "notes": "Deterministic representation, not new evidence"},
        {"component_id": "DOR-AUTO-CASCADE", "canonical_name": "Conditional diagnostic cascade", "component_family": "routing", "architecture_level": "topology", "parent_component": "multi-stage classifier", "nearest_prior_art": "hierarchical/conditional classification", "computation_delta": "detection gates localization/classification/identification", "input_output": "stage outputs -> conditional stage inputs", "action_axis": "stage depth", "interaction_range": "pipeline", "state_rule": "abnormal gate", "inductive_bias": "healthy samples need no downstream diagnosis", "expressive_boundary": "upstream misses truncate downstream", "failure_modes": "error propagation; threshold shift", "innovation_class": "architecture_combination", "evidence_level": "E1", "adoption_level": "A3", "last_verified_date": "2026-08-12", "notes": "End-to-end evaluation required"},
        {"component_id": "DOR-AUTO-MULTILABEL-LOC", "canonical_name": "Calibrated multi-label component localization", "component_family": "prediction_head", "architecture_level": "head/calibration", "parent_component": "one-vs-rest classification", "nearest_prior_art": "multi-label classification and conformal-style threshold calibration", "computation_delta": "one binary head and calibration threshold per component", "input_output": "[B,F] -> component prediction set", "action_axis": "component label", "interaction_range": "per-sample", "inductive_bias": "component faults may co-occur", "expressive_boundary": "empirical calibration is not distribution-shift coverage", "failure_modes": "rare negative labels; label shift", "innovation_class": "architecture_combination", "evidence_level": "E1", "adoption_level": "A3", "source_urls": "https://proceedings.mlr.press/v162/fisch22a.html", "last_verified_date": "2026-08-13", "notes": "No novelty claim"},
        {"component_id": "DOR-AUTO-ID-INTERVAL", "canonical_name": "Split-calibrated identification interval", "component_family": "uncertainty_calibration", "architecture_level": "output", "parent_component": "absolute residual interval", "nearest_prior_art": "split conformal regression", "computation_delta": "per-target held-out residual quantile", "input_output": "point estimate -> lower/upper", "action_axis": "target", "interaction_range": "calibration population", "inductive_bias": "exchangeable calibration and deployment samples", "expressive_boundary": "coverage is not guaranteed under arbitrary shift", "failure_modes": "small calibration set; distribution drift", "innovation_class": "training_or_calibration", "evidence_level": "E1", "adoption_level": "A3", "last_verified_date": "2026-08-13", "notes": "No novelty claim"},
        {"component_id": "DOR-AUTO-CONSTRAINT-GATE", "canonical_name": "Worst-slice, sensor-dropout and resource admission gate", "component_family": "evaluation_and_routing", "architecture_level": "selection/system", "parent_component": "validation model selection", "nearest_prior_art": "robust model selection under deployment constraints", "computation_delta": "aggregate utility penalties plus hard bytes/complexity/latency thresholds", "input_output": "candidate validation/resource records -> eligible candidate set", "action_axis": "condition/sensor/system", "interaction_range": "candidate portfolio", "inductive_bias": "weak deployment regimes and constraints dominate average-only gains", "expressive_boundary": "declared slices and development-host latency only", "failure_modes": "unobserved regimes; timing noise; fallback information loss", "innovation_class": "evaluation_or_system_protocol", "evidence_level": "E1", "adoption_level": "A2", "source_urls": "https://proceedings.neurips.cc/paper_files/paper/2024/hash/c3f3c690b7a99fba16d0efd35cb83b2c-Abstract-Datasets_and_Benchmarks_Track.html;https://ieeexplore.ieee.org/document/11018134", "last_verified_date": "2026-08-13", "notes": "No novelty claim; target hardware acceptance remains external"},
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_auto_design(
    device_card_path: str | Path,
    manifest_path: str | Path,
    data_path: str | Path,
    request_path: str | Path,
    output_dir: str | Path,
    *,
    ontology_path: str | Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    card, manifest, request = _json(device_card_path), _json(manifest_path), _json(request_path)
    _validate_device_card(card); _validate_manifest(manifest); _validate_request(request)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    binding = {
        "device_card_sha256": _sha256(device_card_path), "manifest_sha256": _sha256(manifest_path),
        "data_sha256": _sha256(data_path), "request_sha256": _sha256(request_path),
        "ontology_sha256": _sha256(ontology_path) if ontology_path else None,
    }
    state_path = out / "search_state.json"
    if resume and state_path.exists():
        prior = _json(state_path)
        if prior.get("bindings") != binding:
            raise ValueError("resume rejected: request/data/device bindings changed")
        completed_summary = out / "selected_pipeline.json"
        if prior.get("status") == "COMPLETED" and completed_summary.is_file():
            return _json(completed_summary)

    frame, raw_features = _load_frame(manifest, data_path)
    frame, mechanism_features, mechanism_records = _compile_mechanism_features(frame, card)
    missing = [c for c in raw_features if c not in frame.columns]
    if missing:
        raise ValueError(f"raw features missing after ingestion: {missing}")
    used_features = _unique(raw_features + mechanism_features)
    if not np.all(np.isfinite(frame[used_features].to_numpy(dtype=float))):
        raise ValueError("model features contain missing/non-finite values; implicit imputation is prohibited")
    masks = _role_masks(frame, manifest)
    qualifications = _qualify_stages(frame, manifest, request, masks)
    views = _views(raw_features, mechanism_features, request)
    slice_columns = list(map(str, (manifest.get("evaluation") or {}).get("slice_columns") or []))
    missing_slices = [c for c in slice_columns if c not in frame.columns]
    if missing_slices:
        raise ValueError(f"evaluation slice columns are missing: {missing_slices}")
    raw_feature_fallbacks = {
        column: float(frame.loc[masks["train"], column].median()) for column in raw_features
    }
    feature_fallbacks = {
        column: float(frame.loc[masks["train"], column].median()) for column in used_features
    }
    sensor_groups = _resolve_sensor_groups(raw_features, mechanism_records, request)
    max_candidates = int((request.get("budget") or {}).get("max_candidates", 24))
    max_seconds = float((request.get("budget") or {}).get("max_wall_seconds", 7200))
    seeds = list(map(int, (request.get("search") or {}).get("seeds") or [101, 102, 103]))
    practical_margin = float((request.get("selection") or {}).get("practical_margin", 0.02))
    stability = float((request.get("selection") or {}).get("restart_stability", 0.75))
    model_root = out / "pipeline_bundle" / "models"; model_root.mkdir(parents=True, exist_ok=True)
    if not resume:
        for stale_model in model_root.glob("*.joblib"):
            stale_model.unlink()
    validation_records: list[dict[str, Any]] = []
    stage_selections: dict[str, Any] = {}
    model_index: list[dict[str, Any]] = []
    sup = manifest.get("supervision") or {}
    candidate_count = 0

    _write_json(state_path, {"contract": "DOR-AUTO-SEARCH-STATE/1.0", "status": "RUNNING", "bindings": binding, "completed_candidates": []})
    for q in qualifications:
        if q.status not in {"QUALIFIED", "PORTFOLIO_ONLY"}:
            stage_selections[q.stage] = {"status": q.status, "selected_candidate_id": None, "retained_candidate_ids": [], "reason": q.reason}
            continue
        stage_cfg = sup.get(q.stage.lower()) or {}
        y = _stage_target(frame, q.stage, stage_cfg)
        records: list[dict[str, Any]] = []
        candidate_specs: list[tuple[str, str]] = []
        for view in views:
            for family in _families(q.stage, request):
                candidate_specs.append((view, family))
        remaining = max(0, max_candidates - candidate_count)
        candidate_specs = candidate_specs[:remaining]
        first_seed = seeds[0]
        provisional: dict[str, tuple[float, str, str, bool]] = {}
        for view, family in candidate_specs:
            cid = f"{q.stage.lower()}__{view.lower()}__{family}"
            if time.monotonic() - started > max_seconds:
                break
            xtr = frame.loc[masks["train"], views[view]].to_numpy(dtype=float)
            ytr = None if y is None else y[masks["train"]]
            tick = time.perf_counter()
            model = _fit_model(q.stage, family, first_seed, xtr, ytr, stage_cfg)
            fit_seconds = time.perf_counter() - tick
            xv = frame.loc[masks["validation"], views[view]].to_numpy(dtype=float)
            yv = None if y is None else y[masks["validation"]]
            rec = _evaluate_candidate(
                stage=q.stage, candidate_id=cid, view=view, family=family, seed=first_seed,
                rung="SCREEN", model=model, x_validation=xv, y_validation=yv,
                stage_cfg=stage_cfg, metric=q.primary_metric,
                validation_metadata=frame.loc[masks["validation"]], slice_columns=slice_columns,
                sensor_groups=sensor_groups, feature_columns=views[view],
                feature_fallbacks=feature_fallbacks, request=request, fit_seconds=fit_seconds,
            )
            records.append(rec); validation_records.append(rec)
            provisional[cid] = (float(rec["selection_score"]), view, family, bool(rec["eligible"]))
            candidate_count += 1

        finite_scores = sorted((score for score, _, _, eligible in provisional.values() if eligible and np.isfinite(score)), reverse=True)
        if finite_scores:
            cutoff = finite_scores[min(len(finite_scores) - 1, max(0, math.ceil(len(finite_scores) / 2) - 1))]
            survivors = {cid for cid, (score, _, _, eligible) in provisional.items() if eligible and np.isfinite(score) and score >= cutoff - practical_margin}
        else:
            survivors = {cid for cid, (_, _, _, eligible) in provisional.items() if eligible}
        for cid in sorted(survivors):
            _, view, family, _ = provisional[cid]
            xtr = frame.loc[masks["train"], views[view]].to_numpy(dtype=float)
            ytr = None if y is None else y[masks["train"]]
            for seed in seeds[1:]:
                if time.monotonic() - started > max_seconds:
                    break
                tick = time.perf_counter()
                model = _fit_model(q.stage, family, seed, xtr, ytr, stage_cfg)
                fit_seconds = time.perf_counter() - tick
                xv = frame.loc[masks["validation"], views[view]].to_numpy(dtype=float)
                yv = None if y is None else y[masks["validation"]]
                rec = _evaluate_candidate(
                    stage=q.stage, candidate_id=cid, view=view, family=family, seed=seed,
                    rung="CONFIRM", model=model, x_validation=xv, y_validation=yv,
                    stage_cfg=stage_cfg, metric=q.primary_metric,
                    validation_metadata=frame.loc[masks["validation"]], slice_columns=slice_columns,
                    sensor_groups=sensor_groups, feature_columns=views[view],
                    feature_fallbacks=feature_fallbacks, request=request, fit_seconds=fit_seconds,
                )
                records.append(rec); validation_records.append(rec)
            # Refit a deployment model after selection evidence is measured.  Validation
            # may join final fitting, but calibration/test remain isolated.
            fit_mask = masks["train"] | masks["validation"]
            xf = frame.loc[fit_mask, views[view]].to_numpy(dtype=float)
            yf = None if y is None else y[fit_mask]
            final_seed = int((request.get("search") or {}).get("final_seed", 9001))
            final_model = _fit_model(q.stage, family, final_seed, xf, yf, stage_cfg)
            artifact = model_root / f"{_safe_name(cid)}.joblib"
            joblib.dump(final_model, artifact)
            xv = frame.loc[masks["validation"], views[view]].to_numpy(dtype=float)
            deployment_resources = _resource_profile(q.stage, final_model, xv)
            deployment_resources["artifact_bytes"] = int(artifact.stat().st_size)
            deployment_violations: list[str] = []
            budget = request.get("budget") or {}
            if deployment_resources["artifact_bytes"] > int(budget.get("max_model_bytes", deployment_resources["artifact_bytes"])):
                deployment_violations.append("max_model_bytes")
            if deployment_resources["parameter_estimate"] > int(budget.get("max_parameters", deployment_resources["parameter_estimate"])):
                deployment_violations.append("max_parameters")
            if deployment_resources["inference_latency_ms_per_row"] > float(budget.get("max_inference_latency_ms", deployment_resources["inference_latency_ms_per_row"])):
                deployment_violations.append("max_inference_latency_ms")
            if deployment_violations:
                for row in records:
                    if row["candidate_id"] == cid:
                        row["eligible"] = False
                        row["constraint_violations"] = sorted(set(row["constraint_violations"] + deployment_violations))
            screen = next(row for row in records if row["candidate_id"] == cid)
            model_index.append({"candidate_id": cid, "stage": q.stage, "view": view, "family": family, "feature_columns": views[view], "artifact": str(artifact.relative_to(out / "pipeline_bundle")), "sha256": _sha256(artifact), "screen_resources": screen["resources"], "deployment_resources": deployment_resources})
        labelled = y is not None and q.status == "QUALIFIED"
        stage_selections[q.stage] = _select_stage(q.stage, records, practical_margin, stability, labelled)
        _write_json(state_path, {"contract": "DOR-AUTO-SEARCH-STATE/1.0", "status": "RUNNING", "bindings": binding, "completed_candidates": [m["candidate_id"] for m in model_index]})

    dag = _pipeline_dag(qualifications)
    retained_ids = {
        cid for decision in stage_selections.values()
        for cid in (decision.get("retained_candidate_ids") or [])
    }
    detection_cfg = sup.get("detection") or {}
    detection_target = _stage_target(frame, "DETECTION", detection_cfg) if detection_cfg.get("label_column") else None
    alpha = float((request.get("calibration") or {}).get("alpha", 0.10))
    for rec in model_index:
        if rec["candidate_id"] not in retained_ids or rec["stage"] != "DETECTION":
            continue
        model = joblib.load(out / "pipeline_bundle" / rec["artifact"])
        reference_mask = masks["calibration"].copy()
        reference_role = "calibration"
        if detection_target is not None and reference_mask.any():
            reference_mask &= detection_target.astype(str) == str(detection_cfg.get("normal_label"))
        if not reference_mask.any():
            reference_mask = masks["train"].copy()
            reference_role = "train_normal_fallback"
            if detection_target is not None:
                reference_mask &= detection_target.astype(str) == str(detection_cfg.get("normal_label"))
        scores = model.score(frame.loc[reference_mask, rec["feature_columns"]].to_numpy(dtype=float))
        if len(scores):
            rec["decision_threshold"] = float(np.quantile(scores, 1.0 - alpha, method="higher"))
            rec["threshold_alpha"] = alpha
            rec["threshold_reference_role"] = reference_role
    calibration_mask = masks["calibration"]
    for rec in model_index:
        if rec["candidate_id"] not in retained_ids or not calibration_mask.any():
            continue
        model = joblib.load(out / "pipeline_bundle" / rec["artifact"])
        xcal = frame.loc[calibration_mask, rec["feature_columns"]].to_numpy(dtype=float)
        if rec["stage"] == "LOCALIZATION" and (sup.get("localization") or {}).get("target_columns"):
            targets = list(map(str, sup["localization"]["target_columns"]))
            truth = frame.loc[calibration_mask, targets].to_numpy(dtype=int)
            probabilities = np.asarray(model.predict_proba(xcal), dtype=float)
            thresholds: list[float | None] = []
            for j in range(len(targets)):
                negative = probabilities[truth[:, j] == 0, j]
                thresholds.append(float(np.quantile(negative, 1.0 - alpha, method="higher")) if len(negative) else None)
            rec["multilabel_targets"] = targets
            rec["multilabel_thresholds"] = thresholds
            rec["multilabel_false_positive_alpha"] = alpha
            rec["multilabel_calibration_rows"] = int(calibration_mask.sum())
        elif rec["stage"] == "IDENTIFICATION":
            targets = list(map(str, (sup.get("identification") or {}).get("target_columns") or []))
            truth = frame.loc[calibration_mask, targets].to_numpy(dtype=float)
            pred = np.asarray(model.predict(xcal), dtype=float)
            if pred.ndim == 1:
                pred = pred[:, None]
            rec["interval_targets"] = targets
            rec["absolute_error_quantiles"] = [
                float(np.quantile(np.abs(truth[:, j] - pred[:, j]), 1.0 - alpha, method="higher"))
                for j in range(len(targets))
            ]
            rec["interval_alpha"] = alpha
            rec["interval_calibration_rows"] = int(calibration_mask.sum())
    retained_models: list[dict[str, Any]] = []
    for rec in model_index:
        artifact = out / "pipeline_bundle" / rec["artifact"]
        if rec["candidate_id"] in retained_ids:
            retained_models.append(rec)
        elif artifact.is_file():
            artifact.unlink()
    model_index = retained_models
    bundle = {
        "contract": BUNDLE_CONTRACT, "producer": "DOR Auto Diagnostic Architect",
        "device_id": card["device_id"], "bindings": binding, "modality": manifest.get("modality", "TABULAR"),
        "manifest": manifest, "device_card": card, "pipeline_dag": dag,
        "stage_qualification": [q.to_dict() for q in qualifications], "stage_selections": stage_selections,
        "models": model_index, "raw_feature_columns": raw_features, "mechanism_feature_columns": mechanism_features,
        "raw_feature_fallbacks": raw_feature_fallbacks,
        "robustness": {"sensor_groups": (request.get("robustness") or {}).get("sensor_groups") or {}, "max_missing_groups": int((request.get("robustness") or {}).get("max_missing_groups", 0)), "inference_fallback_scope": "TABULAR raw columns"},
        "prohibited_claims": [
            "finite validation selection is not global NAS optimality",
            "deterministic mechanism features are not new observations",
            "localization or identification output is not automatically root cause",
            "test labels are not architecture-selection inputs",
        ],
    }
    _write_json(out / "pipeline_bundle" / "bundle.json", bundle)
    _write_json(out / "mechanism_stress_audit.json", _mechanism_stress_audit(frame, masks, manifest, request, model_index, mechanism_features))
    _write_json(out / "end_to_end_test_audit.json", _test_audit(frame, masks, manifest, out / "pipeline_bundle", model_index, alpha))
    qualification_payload = {
        "contract": "DOR-AUTO-DESIGN-QUALIFICATION/1.0", "status": "QUALIFIED" if model_index else "UNSUPPORTED",
        "stages": [q.to_dict() for q in qualifications], "role_counts": {k: int(v.sum()) for k, v in masks.items()},
        "leakage_guard_passed": True,
    }
    _write_json(out / "qualification.json", qualification_payload)
    _write_json(out / "mechanism_features.json", {"contract": "DOR-MECHANISM-FEATURES/1.0", "features": mechanism_records})
    _write_json(out / "pipeline_candidates.json", {"contract": "DOR-PIPELINE-CANDIDATES/1.0", "candidate_count": candidate_count, "models": model_index, "pipeline_dag": dag})
    _write_json(out / "validation_results.json", {"contract": "DOR-AUTO-VALIDATION-RESULTS/1.0", "records": validation_records, "selection_uses_test": False})
    summary = {
        "contract": "DOR-AUTO-DESIGN-SUMMARY/1.0", "status": qualification_payload["status"], "device_id": card["device_id"],
        "data": {"modality": manifest.get("modality", "TABULAR"), "rows_after_ingestion": len(frame), "raw_feature_count": len(raw_features), "mechanism_feature_count": len(mechanism_features)},
        "qualification": qualification_payload, "pipeline_dag": dag, "stage_selections": stage_selections,
        "candidate_count": candidate_count, "elapsed_seconds": time.monotonic() - started, "bindings": binding,
    }
    _write_json(out / "selected_pipeline.json", summary)
    (out / "audit_report.md").write_text(_audit_markdown(summary), encoding="utf-8")
    _write_component_registry(out / "component_registry.csv")
    _write_json(state_path, {"contract": "DOR-AUTO-SEARCH-STATE/1.0", "status": "COMPLETED", "bindings": binding, "completed_candidates": [m["candidate_id"] for m in model_index]})
    return summary


def predict_pipeline(bundle_dir: str | Path, data_path: str | Path, output_path: str | Path) -> Path:
    root = Path(bundle_dir)
    bundle = _json(root / "bundle.json")
    _require_contract(bundle, BUNDLE_CONTRACT, "pipeline bundle")
    manifest, card = bundle["manifest"], bundle["device_card"]
    frame, raw = _load_frame(manifest, data_path, inference=True)
    missing_raw = [c for c in bundle.get("raw_feature_columns") or raw if c not in frame.columns]
    missing_groups: set[str] = set()
    group_patterns = (bundle.get("robustness") or {}).get("sensor_groups") or {}
    for column in missing_raw:
        owners = [
            str(group) for group, patterns in group_patterns.items()
            if any(column == str(pattern) or column.startswith(f"{pattern}__") for pattern in patterns)
        ]
        if len(owners) != 1:
            raise ValueError(f"missing raw feature {column!r} is not covered by exactly one declared sensor group")
        missing_groups.add(owners[0])
        fallback = (bundle.get("raw_feature_fallbacks") or {}).get(column)
        if fallback is None:
            raise ValueError(f"missing raw feature {column!r} lacks a training fallback")
        frame[column] = float(fallback)
    maximum = int((bundle.get("robustness") or {}).get("max_missing_groups", 0))
    if len(missing_groups) > maximum:
        raise ValueError(f"missing sensor groups {sorted(missing_groups)} exceed declared maximum {maximum}")
    frame, mechanism, _ = _compile_mechanism_features(frame, card)
    result = pd.DataFrame(index=frame.index)
    result["missing_sensor_group_count"] = len(missing_groups)
    result["missing_sensor_groups"] = "|".join(sorted(missing_groups))
    for col in manifest.get("identifiers") or []:
        if str(col) in frame.columns:
            result[str(col)] = frame[str(col)].values
    by_id = {m["candidate_id"]: m for m in bundle.get("models") or []}
    detection_states: list[np.ndarray] = []
    deferred: list[tuple[str, str, Any, np.ndarray, Mapping[str, Any]]] = []
    for stage, decision in (bundle.get("stage_selections") or {}).items():
        retained = list(decision.get("retained_candidate_ids") or [])
        for cid in retained:
            rec = by_id[cid]
            model = joblib.load(root / rec["artifact"])
            x = frame[rec["feature_columns"]].to_numpy(dtype=float)
            prefix = _safe_name(cid)
            if stage == "DETECTION":
                score = model.score(x)
                result[f"{prefix}__anomaly_score"] = score
                threshold = rec.get("decision_threshold")
                state = np.where(score > float(threshold), "ABNORMAL", "NORMAL") if threshold is not None else np.full(len(score), "UNKNOWN", dtype=object)
                result[f"{prefix}__state"] = state
                detection_states.append(state)
            else:
                deferred.append((stage, prefix, model, x, rec))
    if detection_states:
        stack = np.stack(detection_states, axis=1)
        gate = np.where(np.all(stack == "ABNORMAL", axis=1), "ABNORMAL", np.where(np.all(stack == "NORMAL", axis=1), "NORMAL", "ALTERNATIVES"))
    else:
        gate = np.full(len(frame), "NOT_APPLICABLE", dtype=object)
    result["pipeline_gate"] = gate
    execute_mask = gate != "NORMAL"
    for stage, prefix, model, x, rec in deferred:
        if stage in {"LOCALIZATION", "CLASSIFICATION"}:
            if stage == "LOCALIZATION" and rec.get("multilabel_targets"):
                probabilities = np.asarray(model.predict_proba(x), dtype=float)
                targets = list(rec["multilabel_targets"])
                thresholds = rec.get("multilabel_thresholds") or []
                labels = []
                for j, target in enumerate(targets):
                    threshold = thresholds[j] if j < len(thresholds) else None
                    values = np.full(len(frame), None, dtype=object)
                    if threshold is not None:
                        values[execute_mask] = probabilities[execute_mask, j] > float(threshold)
                    result[f"{prefix}__{_safe_name(target)}__present"] = values
                for i in range(len(frame)):
                    if not execute_mask[i]:
                        labels.append(None)
                    else:
                        labels.append("|".join(targets[j] for j, threshold in enumerate(thresholds) if threshold is not None and probabilities[i, j] > float(threshold)))
                result[f"{prefix}__prediction_set"] = labels
                continue
            pred = model.predict(x).astype(object)
            pred[~execute_mask] = None
            result[f"{prefix}__prediction"] = pred
        else:
            pred = np.asarray(model.predict(x), dtype=float)
            if pred.ndim == 1:
                pred = pred[:, None]
            targets = next(q["supervision"] for q in bundle["stage_qualification"] if q["stage"] == stage)
            for j, target in enumerate(targets):
                values = pred[:, j].copy(); values[~execute_mask] = np.nan
                result[f"{prefix}__{target}"] = values
                errors = rec.get("absolute_error_quantiles") or []
                if j < len(errors):
                    radius = float(errors[j])
                    result[f"{prefix}__{target}__lower"] = values - radius
                    result[f"{prefix}__{target}__upper"] = values + radius
    p = Path(output_path); p.parent.mkdir(parents=True, exist_ok=True); result.to_csv(p, index=False)
    return p
