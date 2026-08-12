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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
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


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    _require_contract(manifest, MANIFEST_CONTRACT, "dataset manifest")
    modality = str(manifest.get("modality", "TABULAR")).upper()
    if modality not in {"TABULAR", "TIMESERIES_LONG", "TIMESERIES_NPZ"}:
        raise ValueError("manifest.modality must be TABULAR, TIMESERIES_LONG, or TIMESERIES_NPZ")
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


def _load_frame(manifest: Mapping[str, Any], data_path: str | Path, *, inference: bool = False) -> tuple[pd.DataFrame, list[str]]:
    p = Path(data_path)
    modality = str(manifest.get("modality", "TABULAR")).upper()
    if modality == "TIMESERIES_NPZ":
        return _load_npz(manifest, p, inference=inference)
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
    metadata_cols = [] if inference else [split_col] + labels
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
            col = cfg.get("label_column")
            if not col or str(col) not in df.columns:
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


def _views(raw: Sequence[str], mechanism: Sequence[str]) -> dict[str, list[str]]:
    result = {"RAW": list(raw)}
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
        return _classifier(family, seed).fit(x, np.asarray(y).astype(str))
    target = np.asarray(y, dtype=float)
    if target.ndim == 2 and target.shape[1] == 1:
        target = target[:, 0]
    return _regressor(family, seed).fit(x, target)


def _stage_target(df: pd.DataFrame, stage: str, cfg: Mapping[str, Any]) -> np.ndarray | None:
    if stage == "DETECTION" and not cfg.get("label_column"):
        return None
    if stage in {"DETECTION", "LOCALIZATION", "CLASSIFICATION"}:
        return df[str(cfg["label_column"])].astype(str).to_numpy()
    return df[list(map(str, cfg.get("target_columns") or []))].to_numpy(dtype=float)


def _select_stage(stage: str, records: list[dict[str, Any]], practical_margin: float, stability: float, labelled: bool) -> dict[str, Any]:
    by_id: dict[str, list[float]] = {}
    for row in records:
        by_id.setdefault(str(row["candidate_id"]), []).append(float(row["score"]))
    medians = {k: float(np.nanmedian(v)) for k, v in by_id.items() if np.isfinite(v).any()}
    if not medians:
        ids = sorted(by_id)
        return {"status": "PORTFOLIO" if stage == "DETECTION" and not labelled else "UNSUPPORTED", "selected_candidate_id": None, "retained_candidate_ids": ids, "candidate_median_scores": {}, "reason": "no labelled validation metric available"}
    order = sorted(medians, key=medians.get, reverse=True)
    if len(order) == 1:
        return {"status": "FIXED", "selected_candidate_id": order[0], "retained_candidate_ids": [order[0]], "candidate_median_scores": medians, "reason": "only one qualified candidate"}
    best, runner = order[:2]
    wins = [a - b >= practical_margin for a in by_id[best] for b in by_id[runner] if np.isfinite(a) and np.isfinite(b)]
    stable = float(np.mean(wins)) if wins else 0.0
    if medians[best] - medians[runner] >= practical_margin and stable >= stability:
        return {"status": "SELECTED", "selected_candidate_id": best, "retained_candidate_ids": [best], "candidate_median_scores": medians, "reason": f"validation margin and restart stability passed ({stable:.3f})"}
    retained = [cid for cid in order if medians[best] - medians[cid] < practical_margin]
    return {"status": "ALTERNATIVES", "selected_candidate_id": None, "retained_candidate_ids": retained, "candidate_median_scores": medians, "reason": "validation evidence does not distinguish a stable practical winner"}


def _pipeline_dag(qualifications: Sequence[StageQualification]) -> dict[str, Any]:
    supported = [q.stage for q in qualifications if q.status in {"QUALIFIED", "PORTFOLIO_ONLY"}]
    nodes = [{"stage": s, "conditional_on": "DETECTION:ABNORMAL" if s != "DETECTION" and "DETECTION" in supported else None} for s in supported]
    edges = []
    for a, b in zip(supported, supported[1:]):
        edges.append({"from": a, "to": b, "type": "CONDITIONAL" if a == "DETECTION" else "SERIAL"})
    return {"topology": "CONDITIONAL_CASCADE", "nodes": nodes, "edges": edges, "teacher_forced_evaluation_prohibited": True}


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

### 时序统计/频域编码

- 一句话机制：将 `[B,T,C]` 的有限窗口压缩为每通道统计量和四段频谱能量。
- 作用轴：时间轴；压缩不可逆，会丢失精确相位和事件次序。
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

所有候选使用相同 train/validation 行、特征视图、随机种子集合和选择指标。RAW、MECHANISM、RAW_PLUS_MECHANISM 分开登记；参数/FLOPs匹配与端到端延迟仍须在具体设备验收中补测。

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

本实现没有“新组件”或“首次提出”声明。实际变化属于成熟统计/机器学习组件的组合、数据表示、训练选择和工程编排；未进行最新技术报告创新性检索，因此禁止作新颖性结论。

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
        {"component_id": "DOR-AUTO-TS-FEATURES", "canonical_name": "Time-series statistical/frequency encoding", "component_family": "data_representation", "architecture_level": "data", "parent_component": "window feature extraction", "nearest_prior_art": "statistical condition-monitoring features", "computation_delta": "fixed summary and FFT-band encoding", "input_output": "[B,T,C] -> [B,F]", "action_axis": "time", "interaction_range": "per-window", "inductive_bias": "stationary finite-window summaries", "expressive_boundary": "phase and exact event order are lost", "failure_modes": "short transient loss; window sensitivity", "innovation_class": "data_or_representation", "evidence_level": "E1", "adoption_level": "A4", "last_verified_date": "2026-08-12", "notes": "No novelty claim"},
        {"component_id": "DOR-AUTO-MECH-VIEW", "canonical_name": "Mechanism feature view", "component_family": "multimodal_fusion", "architecture_level": "data/topology", "parent_component": "engineered feature fusion", "nearest_prior_art": "physics-informed feature engineering", "computation_delta": "raw/mechanism/combined candidate views", "input_output": "[B,F]+[B,M] -> candidate views", "action_axis": "feature", "interaction_range": "per-sample", "inductive_bias": "declared physical relations", "expressive_boundary": "cannot recover unobserved facts", "failure_modes": "wrong provenance; unstable ratio; domain shift", "innovation_class": "architecture_combination", "evidence_level": "E1", "adoption_level": "A3", "last_verified_date": "2026-08-12", "notes": "Deterministic representation, not new evidence"},
        {"component_id": "DOR-AUTO-CASCADE", "canonical_name": "Conditional diagnostic cascade", "component_family": "routing", "architecture_level": "topology", "parent_component": "multi-stage classifier", "nearest_prior_art": "hierarchical/conditional classification", "computation_delta": "detection gates localization/classification/identification", "input_output": "stage outputs -> conditional stage inputs", "action_axis": "stage depth", "interaction_range": "pipeline", "state_rule": "abnormal gate", "inductive_bias": "healthy samples need no downstream diagnosis", "expressive_boundary": "upstream misses truncate downstream", "failure_modes": "error propagation; threshold shift", "innovation_class": "architecture_combination", "evidence_level": "E1", "adoption_level": "A3", "last_verified_date": "2026-08-12", "notes": "End-to-end evaluation required"},
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
    views = _views(raw_features, mechanism_features)
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
        provisional: dict[str, tuple[float, str, str]] = {}
        for view, family in candidate_specs:
            cid = f"{q.stage.lower()}__{view.lower()}__{family}"
            if time.monotonic() - started > max_seconds:
                break
            xtr = frame.loc[masks["train"], views[view]].to_numpy(dtype=float)
            ytr = None if y is None else y[masks["train"]]
            model = _fit_model(q.stage, family, first_seed, xtr, ytr, stage_cfg)
            score = float("nan")
            if y is not None:
                xv = frame.loc[masks["validation"], views[view]].to_numpy(dtype=float)
                score = _score_candidate(q.stage, model, xv, y[masks["validation"]], stage_cfg, q.primary_metric)
            rec = {"stage": q.stage, "candidate_id": cid, "view": view, "family": family, "seed": first_seed, "score": score, "primary_metric": q.primary_metric, "rung": "SCREEN"}
            records.append(rec); validation_records.append(rec)
            provisional[cid] = (score, view, family)
            candidate_count += 1

        finite_scores = sorted((score for score, _, _ in provisional.values() if np.isfinite(score)), reverse=True)
        if finite_scores:
            cutoff = finite_scores[min(len(finite_scores) - 1, max(0, math.ceil(len(finite_scores) / 2) - 1))]
            survivors = {cid for cid, (score, _, _) in provisional.items() if np.isfinite(score) and score >= cutoff - practical_margin}
        else:
            survivors = set(provisional)
        for cid in sorted(survivors):
            _, view, family = provisional[cid]
            xtr = frame.loc[masks["train"], views[view]].to_numpy(dtype=float)
            ytr = None if y is None else y[masks["train"]]
            for seed in seeds[1:]:
                if time.monotonic() - started > max_seconds:
                    break
                model = _fit_model(q.stage, family, seed, xtr, ytr, stage_cfg)
                score = float("nan")
                if y is not None:
                    xv = frame.loc[masks["validation"], views[view]].to_numpy(dtype=float)
                    score = _score_candidate(q.stage, model, xv, y[masks["validation"]], stage_cfg, q.primary_metric)
                rec = {"stage": q.stage, "candidate_id": cid, "view": view, "family": family, "seed": seed, "score": score, "primary_metric": q.primary_metric, "rung": "CONFIRM"}
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
            model_index.append({"candidate_id": cid, "stage": q.stage, "view": view, "family": family, "feature_columns": views[view], "artifact": str(artifact.relative_to(out / "pipeline_bundle")), "sha256": _sha256(artifact)})
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
        "prohibited_claims": [
            "finite validation selection is not global NAS optimality",
            "deterministic mechanism features are not new observations",
            "localization or identification output is not automatically root cause",
            "test labels are not architecture-selection inputs",
        ],
    }
    _write_json(out / "pipeline_bundle" / "bundle.json", bundle)
    qualification_payload = {
        "contract": "DOR-AUTO-DESIGN-QUALIFICATION/1.0", "status": "QUALIFIED" if any(q.status in {"QUALIFIED", "PORTFOLIO_ONLY"} for q in qualifications) else "UNSUPPORTED",
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
    frame, mechanism, _ = _compile_mechanism_features(frame, card)
    result = pd.DataFrame(index=frame.index)
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
    p = Path(output_path); p.parent.mkdir(parents=True, exist_ok=True); result.to_csv(p, index=False)
    return p
