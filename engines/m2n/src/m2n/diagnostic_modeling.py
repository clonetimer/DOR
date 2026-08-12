from __future__ import annotations

"""Task-aware diagnostic model orchestration for M2N.

This module closes a deliberate gap in the original M2N practitioner baseline:
M2N could design scientific-ML mechanism models, but no component owned the
finite competition among ordinary data-driven diagnostic estimators when DOR
had genuine fault labels or anomaly-evaluation labels.

The scope is intentionally bounded:

* FAULT_CLASSIFICATION on numeric tabular features;
* ANOMALY_DETECTION on numeric tabular features, trained from trusted-normal
  data and optionally selected with an independently labelled validation split;
* explicit RAW / MECHANISM / RAW_PLUS_MECHANISM feature views;
* model-family qualification before training;
* validation-only selection, independent calibration/test splits, and refusal;
* score bundles for downstream FEC/DOR, not final fault diagnosis.

It is not a generic AutoML framework and it does not infer fault ontology,
root cause, or PHM policy.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import hashlib
import json
import math
import re
import warnings

import joblib
import numpy as np
import pandas as pd

try:  # imported lazily enough to keep the error actionable for source users
    from sklearn.base import clone
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        log_loss,
        recall_score,
        roc_auc_score,
    )
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import OneClassSVM, SVC
except Exception as exc:  # pragma: no cover - only exercised without optional runtime
    raise RuntimeError(
        "M2N diagnostic model orchestration requires scikit-learn. "
        "Install the toolkit diagnostics dependency before using this module."
    ) from exc


DIAGNOSTIC_TASKS = ("FAULT_CLASSIFICATION", "ANOMALY_DETECTION")
FEATURE_VIEWS = ("RAW", "MECHANISM", "RAW_PLUS_MECHANISM")
CLASSIFIER_FAMILIES = (
    "logistic_regression",
    "rbf_svm",
    "random_forest",
    "hist_gradient_boosting",
    "mlp",
)
DETECTOR_FAMILIES = ("pca_reconstruction", "one_class_svm", "isolation_forest")


@dataclass(frozen=True)
class DiagnosticFamilyDescriptor:
    id: str
    task: str
    model_kind: str
    probability_output: bool
    rationale: str
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_FAMILY_REGISTRY: dict[str, DiagnosticFamilyDescriptor] = {
    "logistic_regression": DiagnosticFamilyDescriptor(
        "logistic_regression", "FAULT_CLASSIFICATION", "CLASSICAL_ML", True,
        "Regularized linear probabilistic baseline; useful when engineered/mechanism features make classes close to linearly separable.",
        ("cannot represent strongly nonlinear class boundaries without nonlinear features",),
    ),
    "rbf_svm": DiagnosticFamilyDescriptor(
        "rbf_svm", "FAULT_CLASSIFICATION", "CLASSICAL_ML", True,
        "Kernel classifier for nonlinear tabular boundaries; admitted only below a configurable sample-count cap.",
        ("training cost scales poorly on very large datasets", "probability output is an internal model calibration, not FEC calibration"),
    ),
    "random_forest": DiagnosticFamilyDescriptor(
        "random_forest", "FAULT_CLASSIFICATION", "CLASSICAL_ML", True,
        "Nonlinear tree ensemble baseline with limited preprocessing assumptions and useful robustness to feature scaling.",
        ("probabilities can be poorly calibrated",),
    ),
    "hist_gradient_boosting": DiagnosticFamilyDescriptor(
        "hist_gradient_boosting", "FAULT_CLASSIFICATION", "CLASSICAL_ML", True,
        "Nonlinear boosting baseline for tabular fault classification and interactions among observed/mechanism features.",
        ("probabilities require downstream calibration before policy use",),
    ),
    "mlp": DiagnosticFamilyDescriptor(
        "mlp", "FAULT_CLASSIFICATION", "NEURAL_BASELINE", True,
        "Small generic neural baseline retained so data, rather than model ideology, can decide whether extra nonlinear capacity is useful.",
        ("not a physics architecture", "optimization can be seed-sensitive",),
    ),
    "pca_reconstruction": DiagnosticFamilyDescriptor(
        "pca_reconstruction", "ANOMALY_DETECTION", "STATISTICAL", False,
        "Normal-subspace reconstruction score; useful when healthy behavior is approximately low-rank in the selected feature view.",
        ("requires at least two informative features", "score threshold is intentionally delegated to FEC"),
    ),
    "one_class_svm": DiagnosticFamilyDescriptor(
        "one_class_svm", "ANOMALY_DETECTION", "CLASSICAL_ML", False,
        "Kernel support estimator trained on trusted-normal data.",
        ("training cost scales poorly on very large datasets", "score threshold is intentionally delegated to FEC"),
    ),
    "isolation_forest": DiagnosticFamilyDescriptor(
        "isolation_forest", "ANOMALY_DETECTION", "CLASSICAL_ML", False,
        "Non-parametric isolation score baseline trained on trusted-normal data.",
        ("ranking is not a calibrated anomaly probability",),
    ),
}


def list_diagnostic_families(task: str | None = None) -> tuple[DiagnosticFamilyDescriptor, ...]:
    if task is None:
        return tuple(_FAMILY_REGISTRY[k] for k in sorted(_FAMILY_REGISTRY))
    task = str(task).upper()
    if task not in DIAGNOSTIC_TASKS:
        raise ValueError(f"unsupported diagnostic task {task!r}")
    return tuple(v for v in _FAMILY_REGISTRY.values() if v.task == task)


@dataclass(frozen=True)
class DiagnosticQualification:
    status: str  # QUALIFIED | CONDITIONALLY_QUALIFIED | UNSUPPORTED
    task: str
    n_rows: int
    split_counts: dict[str, int]
    raw_feature_columns: tuple[str, ...]
    mechanism_feature_columns: tuple[str, ...]
    requested_views: tuple[str, ...]
    class_frame: tuple[str, ...]
    warnings: tuple[str, ...]
    reasons: tuple[str, ...]
    calibration_ready: bool
    test_available: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiagnosticCandidate:
    id: str
    family: str
    view: str
    feature_columns: tuple[str, ...]
    admitted: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiagnosticSelectionPolicy:
    primary_metric: str
    practical_margin: float = 0.02
    restart_stability: float = 0.75
    quality_floor: float | None = None

    def validate(self, task: str) -> "DiagnosticSelectionPolicy":
        allowed = {
            "FAULT_CLASSIFICATION": {"macro_f1", "balanced_accuracy", "min_class_recall"},
            "ANOMALY_DETECTION": {"roc_auc", "average_precision"},
        }[task]
        if self.primary_metric not in allowed:
            raise ValueError(f"primary_metric {self.primary_metric!r} not allowed for {task}: {sorted(allowed)}")
        if self.practical_margin < 0 or self.practical_margin >= 1:
            raise ValueError("practical_margin must be in [0,1)")
        if not 0 <= self.restart_stability <= 1:
            raise ValueError("restart_stability must be in [0,1]")
        if self.quality_floor is not None and not 0 <= self.quality_floor <= 1:
            raise ValueError("quality_floor must be in [0,1]")
        return self


@dataclass(frozen=True)
class DiagnosticSelectionDecision:
    status: str  # FIXED | SELECTED | AMBIGUOUS | PORTFOLIO | UNSUPPORTED
    selected_candidate_id: str | None
    runner_up_candidate_id: str | None
    primary_metric: str
    candidate_median_scores: tuple[tuple[str, float], ...]
    minimum_pairwise_margin: float
    pairwise_practical_win_fraction: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiagnosticDesignSummary:
    status: str
    task: str
    selected_candidate_id: str | None
    output_dir: str
    qualification: DiagnosticQualification
    selection: DiagnosticSelectionDecision
    artifacts: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "task": self.task,
            "selected_candidate_id": self.selected_candidate_id,
            "output_dir": self.output_dir,
            "qualification": self.qualification.to_dict(),
            "selection": self.selection.to_dict(),
            "artifacts": list(self.artifacts),
        }


class PCAReconstructionDetector:
    """Pickle-safe normal-subspace detector with a higher-is-more-anomalous score."""

    def __init__(self, variance_fraction: float = 0.95):
        self.variance_fraction = float(variance_fraction)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.variance_fraction, svd_solver="full")

    def fit(self, X: np.ndarray) -> "PCAReconstructionDetector":
        Z = self.scaler.fit_transform(np.asarray(X, dtype=float))
        self.pca.fit(Z)
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        Z = self.scaler.transform(np.asarray(X, dtype=float))
        Zhat = self.pca.inverse_transform(self.pca.transform(Z))
        return np.mean(np.square(Z - Zhat), axis=1)


class OneClassSVMDetector:
    def __init__(self, *, nu: float = 0.05):
        self.pipeline = make_pipeline(StandardScaler(), OneClassSVM(kernel="rbf", gamma="scale", nu=float(nu)))

    def fit(self, X: np.ndarray) -> "OneClassSVMDetector":
        self.pipeline.fit(np.asarray(X, dtype=float))
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        return -np.asarray(self.pipeline.decision_function(np.asarray(X, dtype=float)), dtype=float).reshape(-1)


class IsolationForestDetector:
    def __init__(self, *, random_state: int):
        self.model = IsolationForest(
            n_estimators=300,
            contamination="auto",
            random_state=int(random_state),
            n_jobs=1,
        )

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        self.model.fit(np.asarray(X, dtype=float))
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        return -np.asarray(self.model.decision_function(np.asarray(X, dtype=float)), dtype=float).reshape(-1)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))


def _unique(items: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(x) for x in items))


def _load_request(request_path: str | Path) -> dict[str, Any]:
    p = Path(request_path)
    d = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(d, dict):
        raise ValueError("diagnostic request root must be an object")
    contract = str(d.get("contract", "M2N-DIAGNOSTIC-MODEL-REQUEST/1.0"))
    if contract != "M2N-DIAGNOSTIC-MODEL-REQUEST/1.0":
        raise ValueError(f"unsupported diagnostic request contract: {contract}")
    d["contract"] = contract
    task = str(d.get("task", "")).upper()
    if task not in DIAGNOSTIC_TASKS:
        raise ValueError(f"task must be one of {DIAGNOSTIC_TASKS}")
    d["task"] = task
    return d


def _read_data(data_path: str | Path) -> pd.DataFrame:
    p = Path(data_path)
    if p.suffix.lower() != ".csv":
        raise ValueError("diagnostic model orchestration currently supports headered CSV input")
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError("diagnostic data CSV is empty")
    if len(set(df.columns)) != len(df.columns):
        raise ValueError("diagnostic data contains duplicate column names")
    return df


def _split_masks(request: Mapping[str, Any], df: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, int], tuple[str, ...]]:
    data = request.get("data") or {}
    col = data.get("split_column")
    reasons: list[str] = []
    if not col or str(col) not in df.columns:
        raise ValueError("explicit data.split_column is required; M2N will not create a random split for fault diagnosis")
    col = str(col)
    mapping = {
        "train": tuple(map(str, data.get("train_values") or ("train",))),
        "validation": tuple(map(str, data.get("validation_values") or ("val", "validation"))),
        "calibration": tuple(map(str, data.get("calibration_values") or ())),
        "test": tuple(map(str, data.get("test_values") or ())),
    }
    s = df[col].astype(str).to_numpy()
    masks = {k: np.isin(s, np.asarray(v, dtype=object)) if v else np.zeros(len(df), dtype=bool) for k, v in mapping.items()}
    counts = {k: int(np.sum(v)) for k, v in masks.items()}
    assigned = np.zeros(len(df), dtype=int)
    for m in masks.values():
        assigned += m.astype(int)
    if np.any(assigned > 1):
        raise ValueError("split values overlap across train/validation/calibration/test roles")
    if np.any(assigned == 0):
        reasons.append(f"{int(np.sum(assigned == 0))} rows are outside declared split roles and will not enter fitting/selection/calibration/test")
    return masks, counts, tuple(reasons)


def _feature_config(request: Mapping[str, Any], df: pd.DataFrame) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    f = request.get("features") or {}
    raw = _unique(f.get("raw_columns") or f.get("columns") or ())
    mech = _unique(f.get("mechanism_columns") or ())
    if not raw and not mech:
        raise ValueError("features.raw_columns and/or features.mechanism_columns must be declared explicitly")
    missing = [c for c in raw + mech if c not in df.columns]
    if missing:
        raise ValueError(f"declared diagnostic feature columns are missing: {sorted(set(missing))}")
    requested = f.get("views")
    if requested:
        views = _unique(str(x).upper() for x in requested)
    else:
        views: tuple[str, ...]
        if raw and mech:
            views = ("RAW", "MECHANISM", "RAW_PLUS_MECHANISM")
        elif raw:
            views = ("RAW",)
        else:
            views = ("MECHANISM",)
    bad = [x for x in views if x not in FEATURE_VIEWS]
    if bad:
        raise ValueError(f"unsupported feature views: {bad}; supported={FEATURE_VIEWS}")
    if "RAW" in views and not raw:
        raise ValueError("RAW view requested but no raw feature columns were declared")
    if "MECHANISM" in views and not mech:
        raise ValueError("MECHANISM view requested but no mechanism feature columns were declared")
    if "RAW_PLUS_MECHANISM" in views and (not raw or not mech):
        raise ValueError("RAW_PLUS_MECHANISM view requires both raw and mechanism feature columns")
    return raw, mech, views


def _view_columns(view: str, raw: Sequence[str], mech: Sequence[str]) -> tuple[str, ...]:
    if view == "RAW":
        return tuple(raw)
    if view == "MECHANISM":
        return tuple(mech)
    if view == "RAW_PLUS_MECHANISM":
        return _unique(tuple(raw) + tuple(mech))
    raise ValueError(view)


def qualify_diagnostic_request(request: Mapping[str, Any], df: pd.DataFrame) -> DiagnosticQualification:
    task = str(request.get("task", "")).upper()
    if task not in DIAGNOSTIC_TASKS:
        return DiagnosticQualification("UNSUPPORTED", task, len(df), {}, (), (), (), (), (), ("unsupported task",), False, False)
    warnings_out: list[str] = []
    reasons: list[str] = []
    try:
        masks, counts, split_notes = _split_masks(request, df)
        reasons.extend(split_notes)
        raw, mech, views = _feature_config(request, df)
    except Exception as e:
        return DiagnosticQualification("UNSUPPORTED", task, len(df), {}, (), (), (), (), (), (str(e),), False, False)

    q = request.get("qualification") or {}
    min_train = int(q.get("min_train_rows", 30))
    min_val = int(q.get("min_validation_rows", 20))
    min_cal = int(q.get("min_calibration_rows", 20))
    if counts["train"] < min_train:
        reasons.append(f"train split has {counts['train']} rows < min_train_rows={min_train}")
    if counts["validation"] < min_val:
        reasons.append(f"validation split has {counts['validation']} rows < min_validation_rows={min_val}")

    reserved = {str((request.get("data") or {}).get("split_column"))}
    sup = request.get("supervision") or {}
    if sup.get("label_column"):
        reserved.add(str(sup["label_column"]))
    if (request.get("data") or {}).get("group_column"):
        reserved.add(str(request["data"]["group_column"]))
    leak = sorted(set(raw + mech) & reserved)
    if leak:
        reasons.append(f"label/split/group leakage: reserved columns were declared as features: {leak}")

    # numeric/finite qualification is intentionally strict: missing-modality handling belongs upstream/FEC.
    for c in _unique(raw + mech):
        if not pd.api.types.is_numeric_dtype(df[c]):
            reasons.append(f"feature {c!r} is non-numeric; current diagnostic runtime supports numeric tabular features only")
            continue
        used = masks["train"] | masks["validation"] | masks["calibration"] | masks["test"]
        a = pd.to_numeric(df.loc[used, c], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(a)):
            reasons.append(f"feature {c!r} contains missing/non-finite values in a declared split; no implicit imputation is allowed")
    if raw and mech and set(raw) & set(mech):
        warnings_out.append(f"raw/mechanism feature declarations overlap: {sorted(set(raw) & set(mech))}; combined view de-duplicates them")

    class_frame: tuple[str, ...] = ()
    calibration_ready = counts["calibration"] >= min_cal
    if task == "FAULT_CLASSIFICATION":
        label_col = sup.get("label_column")
        if not label_col or str(label_col) not in df.columns:
            reasons.append("FAULT_CLASSIFICATION requires supervision.label_column")
        else:
            label_col = str(label_col)
            ytr = df.loc[masks["train"], label_col].astype(str).to_numpy()
            yv = df.loc[masks["validation"], label_col].astype(str).to_numpy()
            class_frame = tuple(pd.unique(ytr).tolist())
            if len(class_frame) < 2:
                reasons.append("classification train split must contain at least two fault classes")
            unseen = sorted(set(yv) - set(class_frame))
            if unseen:
                reasons.append(f"validation contains classes unseen in training: {unseen}")
            missing_val = [c for c in class_frame if int(np.sum(yv == c)) < int(q.get("min_validation_rows_per_class", 2))]
            if missing_val:
                reasons.append(f"validation has insufficient support for train classes: {missing_val}")
            counts_by_class = {c: int(np.sum(ytr == c)) for c in class_frame}
            if counts_by_class:
                mx = max(counts_by_class.values()); mn = min(counts_by_class.values())
                if mx > 0 and mn / mx < 0.10:
                    warnings_out.append(f"severe train class imbalance detected: {counts_by_class}; class-balanced estimators/metrics are used where available")
            if counts["calibration"]:
                yc = set(df.loc[masks["calibration"], label_col].astype(str))
                missing_cal = [c for c in class_frame if c not in yc]
                if missing_cal:
                    calibration_ready = False
                    warnings_out.append(f"calibration split lacks class support for {missing_cal}; FEC class calibration should abstain or collect more labeled calibration data")
    else:
        label_col = sup.get("label_column")
        normal_label = sup.get("normal_label")
        trusted_normal = bool(sup.get("trusted_normal_train", False))
        if label_col:
            if str(label_col) not in df.columns:
                reasons.append(f"declared anomaly label column {label_col!r} is missing")
            elif normal_label is None:
                reasons.append("ANOMALY_DETECTION with labels requires supervision.normal_label")
            else:
                ytr = df.loc[masks["train"], str(label_col)].astype(str).to_numpy()
                nnormal = int(np.sum(ytr == str(normal_label)))
                if nnormal < min_train:
                    reasons.append(f"trusted-normal training subset has {nnormal} rows < min_train_rows={min_train}")
                if np.any(ytr != str(normal_label)):
                    warnings_out.append("non-normal rows exist in the train split; diagnostic detector fitting will use only rows matching normal_label")
        elif not trusted_normal:
            reasons.append("ANOMALY_DETECTION without labels requires supervision.trusted_normal_train=true")
        # No absolute anomaly threshold is fitted here, so calibration_ready means a split exists, not that FEC calibration is valid.

    # A fully constant view is not useful; reject only if every declared column is constant on train.
    informative = False
    for c in _unique(raw + mech):
        if pd.api.types.is_numeric_dtype(df[c]):
            a = pd.to_numeric(df.loc[masks["train"], c], errors="coerce").to_numpy(dtype=float)
            if len(a) and np.all(np.isfinite(a)) and float(np.nanstd(a)) > 0:
                informative = True
                break
    if not informative:
        reasons.append("all declared diagnostic features are constant/non-informative on the train split")

    fatal_markers = (
        "requires", "missing", "leakage", "non-numeric", "non-finite", "insufficient", "unseen", "constant/non-informative", "rows <"
    )
    fatal = [r for r in reasons if any(m in r for m in fatal_markers)]
    status = "UNSUPPORTED" if fatal else ("CONDITIONALLY_QUALIFIED" if warnings_out or not calibration_ready else "QUALIFIED")
    return DiagnosticQualification(
        status, task, len(df), counts, raw, mech, views, class_frame,
        tuple(warnings_out), tuple(reasons), bool(calibration_ready), bool(counts["test"]),
    )


def _candidate_families(request: Mapping[str, Any], task: str) -> tuple[str, ...]:
    pol = request.get("candidate_policy") or {}
    default = CLASSIFIER_FAMILIES if task == "FAULT_CLASSIFICATION" else DETECTOR_FAMILIES
    fam = _unique(pol.get("families") or default)
    unknown = [x for x in fam if x not in _FAMILY_REGISTRY]
    if unknown:
        raise ValueError(f"unknown diagnostic model families: {unknown}")
    wrong = [x for x in fam if _FAMILY_REGISTRY[x].task != task]
    if wrong:
        raise ValueError(f"families do not support task {task}: {wrong}")
    return fam


def build_diagnostic_candidates(request: Mapping[str, Any], q: DiagnosticQualification) -> tuple[DiagnosticCandidate, ...]:
    if q.status == "UNSUPPORTED":
        return ()
    families = _candidate_families(request, q.task)
    pol = request.get("candidate_policy") or {}
    svm_max_rows = int(pol.get("svm_max_train_rows", 30000))
    out: list[DiagnosticCandidate] = []
    for view in q.requested_views:
        cols = _view_columns(view, q.raw_feature_columns, q.mechanism_feature_columns)
        for family in families:
            admitted = True
            reason = _FAMILY_REGISTRY[family].rationale
            if family in ("rbf_svm", "one_class_svm") and q.split_counts.get("train", 0) > svm_max_rows:
                admitted = False
                reason = f"rejected by sample-count guard: train rows exceed svm_max_train_rows={svm_max_rows}"
            if family == "pca_reconstruction" and len(cols) < 2:
                admitted = False
                reason = "PCA reconstruction requires at least two features in the selected view"
            if family == "hist_gradient_boosting" and q.split_counts.get("train", 0) < 40:
                admitted = False
                reason = "hist gradient boosting withheld for very small train split (<40 rows)"
            cid = f"{view.lower()}__{family}"
            out.append(DiagnosticCandidate(cid, family, view, cols, admitted, reason))
    return tuple(out)


def _make_classifier(family: str, seed: int):
    if family == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1200, class_weight="balanced", random_state=int(seed)),
        )
    if family == "rbf_svm":
        return make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", gamma="scale", C=1.0, probability=True, class_weight="balanced", random_state=int(seed)),
        )
    if family == "random_forest":
        return RandomForestClassifier(
            n_estimators=350,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=int(seed),
            n_jobs=1,
        )
    if family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.08,
            l2_regularization=0.1,
            random_state=int(seed),
        )
    if family == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=450,
                early_stopping=False,
                n_iter_no_change=30,
                random_state=int(seed),
            ),
        )
    raise KeyError(family)


def _make_detector(family: str, seed: int):
    if family == "pca_reconstruction":
        return PCAReconstructionDetector()
    if family == "one_class_svm":
        return OneClassSVMDetector(nu=0.05)
    if family == "isolation_forest":
        return IsolationForestDetector(random_state=int(seed))
    raise KeyError(family)


def _aligned_probabilities(model: Any, X: np.ndarray, class_frame: Sequence[str]) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError("selected classifier does not expose predict_proba")
    p = np.asarray(model.predict_proba(X), dtype=float)
    classes = tuple(map(str, getattr(model, "classes_", ())))
    if not classes and hasattr(model, "steps"):
        classes = tuple(map(str, getattr(model.steps[-1][1], "classes_", ())))
    if not classes:
        raise RuntimeError("cannot determine classifier class order")
    idx = [classes.index(str(c)) for c in class_frame]
    out = p[:, idx]
    # Guard small numerical drift only; do not temperature-scale/calibrate here.
    denom = np.sum(out, axis=1, keepdims=True)
    if np.any(denom <= 0) or np.any(~np.isfinite(out)):
        raise RuntimeError("classifier produced invalid class probabilities")
    return out / denom


def _classification_metrics(model: Any, X: np.ndarray, y: np.ndarray, class_frame: Sequence[str]) -> dict[str, float]:
    pred = np.asarray(model.predict(X)).astype(str)
    y = np.asarray(y).astype(str)
    p = _aligned_probabilities(model, X, class_frame)
    labels = list(map(str, class_frame))
    recalls = recall_score(y, pred, labels=labels, average=None, zero_division=0)
    metrics = {
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "min_class_recall": float(np.min(recalls)) if len(recalls) else 0.0,
    }
    try:
        sorted_labels = sorted(labels)
        reorder = [labels.index(c) for c in sorted_labels]
        metrics["log_loss"] = float(log_loss(y, p[:, reorder], labels=sorted_labels))
    except Exception:
        metrics["log_loss"] = float("nan")
    return metrics


def _anomaly_labels(request: Mapping[str, Any], df: pd.DataFrame, mask: np.ndarray) -> np.ndarray | None:
    sup = request.get("supervision") or {}
    label_col = sup.get("label_column")
    normal_label = sup.get("normal_label")
    if not label_col or normal_label is None:
        return None
    y = df.loc[mask, str(label_col)].astype(str).to_numpy()
    b = (y != str(normal_label)).astype(int)
    if len(np.unique(b)) < 2:
        return None
    return b


def _detection_metrics(scores: np.ndarray, binary_labels: np.ndarray) -> dict[str, float]:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    y = np.asarray(binary_labels, dtype=int).reshape(-1)
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
    }


def _selection_policy(request: Mapping[str, Any], task: str) -> DiagnosticSelectionPolicy:
    d = request.get("selection") or {}
    default_metric = "macro_f1" if task == "FAULT_CLASSIFICATION" else "roc_auc"
    qf = d.get("quality_floor")
    return DiagnosticSelectionPolicy(
        primary_metric=str(d.get("primary_metric", default_metric)),
        practical_margin=float(d.get("practical_margin", 0.02)),
        restart_stability=float(d.get("restart_stability", 0.75)),
        quality_floor=None if qf is None else float(qf),
    ).validate(task)


def _select_from_scores(
    score_map: Mapping[str, Sequence[float]],
    policy: DiagnosticSelectionPolicy,
    *,
    no_labels_for_detection: bool = False,
) -> DiagnosticSelectionDecision:
    if no_labels_for_detection:
        med = tuple(sorted(((k, float("nan")) for k in score_map), key=lambda x: x[0]))
        return DiagnosticSelectionDecision(
            "PORTFOLIO", None, None, policy.primary_metric, med, 0.0, 0.0,
            ("No independently labelled anomaly validation split is available; retain the qualified detector set and do not manufacture a winner from healthy-only fit quality.",),
        )
    clean: dict[str, np.ndarray] = {}
    for k, vals in score_map.items():
        a = np.asarray(tuple(vals), dtype=float)
        a = a[np.isfinite(a)]
        if len(a) >= 2:
            clean[str(k)] = a
    if not clean:
        return DiagnosticSelectionDecision(
            "UNSUPPORTED", None, None, policy.primary_metric, (), 0.0, 0.0,
            ("No candidate produced at least two finite validation scores.",),
        )
    med = {k: float(np.median(v)) for k, v in clean.items()}
    order = sorted(med, key=med.get, reverse=True)
    if len(order) == 1:
        best = order[0]
        if policy.quality_floor is not None and med[best] < policy.quality_floor:
            return DiagnosticSelectionDecision(
                "UNSUPPORTED", None, None, policy.primary_metric, ((best, med[best]),), 0.0, 0.0,
                (f"Only one candidate is admissible but its median {policy.primary_metric}={med[best]:.4f} is below quality_floor={policy.quality_floor:.4f}.",),
            )
        return DiagnosticSelectionDecision(
            "FIXED", best, None, policy.primary_metric, ((best, med[best]),), math.inf, 1.0,
            ("Exactly one candidate survived qualification; no relative architecture/model-family claim is made.",),
        )

    best, runner = order[0], order[1]
    min_margin = math.inf
    min_stability = 1.0
    for other in order[1:]:
        diffs = np.asarray([a - b for a in clean[best] for b in clean[other]], dtype=float)
        margin = float(np.median(diffs))
        stability = float(np.mean(diffs > policy.practical_margin))
        min_margin = min(min_margin, margin)
        min_stability = min(min_stability, stability)
    reasons: list[str] = []
    if policy.quality_floor is not None and med[best] < policy.quality_floor:
        reasons.append(f"best median {policy.primary_metric}={med[best]:.4f} is below quality_floor={policy.quality_floor:.4f}")
        status = "UNSUPPORTED"
        selected = None
    elif min_margin > policy.practical_margin and min_stability >= policy.restart_stability:
        status = "SELECTED"
        selected = best
        reasons.append(
            f"best candidate exceeds every alternative by the configured practical margin across stable restart comparisons"
        )
    else:
        status = "AMBIGUOUS"
        selected = None
        reasons.append(
            f"no unique winner: minimum median advantage={min_margin:.4f}, required>{policy.practical_margin:.4f}; pairwise practical-win fraction={min_stability:.3f}, required>={policy.restart_stability:.3f}"
        )
    return DiagnosticSelectionDecision(
        status, selected, runner, policy.primary_metric,
        tuple((k, med[k]) for k in order), float(min_margin), float(min_stability), tuple(reasons),
    )


def _candidate_feature_matrix(df: pd.DataFrame, mask: np.ndarray, cols: Sequence[str]) -> np.ndarray:
    return df.loc[mask, list(cols)].to_numpy(dtype=float)


def _train_normal_mask(request: Mapping[str, Any], df: pd.DataFrame, train_mask: np.ndarray) -> np.ndarray:
    sup = request.get("supervision") or {}
    label_col = sup.get("label_column")
    normal_label = sup.get("normal_label")
    if label_col and normal_label is not None:
        y = df[str(label_col)].astype(str).to_numpy()
        return train_mask & (y == str(normal_label))
    return train_mask.copy()


def _evaluate_candidate(
    request: Mapping[str, Any],
    q: DiagnosticQualification,
    candidate: DiagnosticCandidate,
    df: pd.DataFrame,
    masks: Mapping[str, np.ndarray],
    seed: int,
) -> tuple[Any, dict[str, float]]:
    Xtr = _candidate_feature_matrix(df, masks["train"], candidate.feature_columns)
    Xv = _candidate_feature_matrix(df, masks["validation"], candidate.feature_columns)
    if q.task == "FAULT_CLASSIFICATION":
        label_col = str((request.get("supervision") or {})["label_column"])
        ytr = df.loc[masks["train"], label_col].astype(str).to_numpy()
        yv = df.loc[masks["validation"], label_col].astype(str).to_numpy()
        model = _make_classifier(candidate.family, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(Xtr, ytr)
        return model, _classification_metrics(model, Xv, yv, q.class_frame)
    normal_mask = _train_normal_mask(request, df, masks["train"])
    Xn = _candidate_feature_matrix(df, normal_mask, candidate.feature_columns)
    model = _make_detector(candidate.family, seed).fit(Xn)
    scores = model.anomaly_score(Xv)
    yv = _anomaly_labels(request, df, masks["validation"])
    return model, {} if yv is None else _detection_metrics(scores, yv)


def _fit_final_model(
    request: Mapping[str, Any],
    q: DiagnosticQualification,
    candidate: DiagnosticCandidate,
    df: pd.DataFrame,
    masks: Mapping[str, np.ndarray],
    seed: int,
):
    fit_mask = masks["train"] | masks["validation"]
    X = _candidate_feature_matrix(df, fit_mask, candidate.feature_columns)
    if q.task == "FAULT_CLASSIFICATION":
        label_col = str((request.get("supervision") or {})["label_column"])
        y = df.loc[fit_mask, label_col].astype(str).to_numpy()
        m = _make_classifier(candidate.family, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            m.fit(X, y)
        return m
    normal_mask = _train_normal_mask(request, df, fit_mask)
    Xn = _candidate_feature_matrix(df, normal_mask, candidate.feature_columns)
    return _make_detector(candidate.family, seed).fit(Xn)


def _write_candidate_score_pack(
    *,
    task: str,
    model: Any,
    candidate: DiagnosticCandidate,
    q: DiagnosticQualification,
    request: Mapping[str, Any],
    df: pd.DataFrame,
    mask: np.ndarray,
    path: Path,
    include_labels: bool,
) -> Path:
    X = _candidate_feature_matrix(df, mask, candidate.feature_columns)
    out = pd.DataFrame(index=np.arange(int(np.sum(mask))))
    data_cfg = request.get("data") or {}
    id_cols = _unique(data_cfg.get("id_columns") or ())
    for c in id_cols:
        if c in df.columns:
            out[c] = df.loc[mask, c].to_numpy()
    if task == "FAULT_CLASSIFICATION":
        probs = _aligned_probabilities(model, X, q.class_frame)
        for j, c in enumerate(q.class_frame):
            out[f"score__{c}"] = probs[:, j]
        out["model_prediction"] = np.asarray(model.predict(X)).astype(str)
        if include_labels:
            label_col = str((request.get("supervision") or {})["label_column"])
            out["true_label"] = df.loc[mask, label_col].astype(str).to_numpy()
    else:
        out["anomaly_score"] = model.anomaly_score(X)
        if include_labels and (request.get("supervision") or {}).get("label_column"):
            label_col = str(request["supervision"]["label_column"])
            out["true_label"] = df.loc[mask, label_col].astype(str).to_numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path


def _write_score_contract(
    path: Path,
    *,
    request: Mapping[str, Any],
    q: DiagnosticQualification,
    candidate: DiagnosticCandidate,
    data_sha256: str,
) -> Path:
    task = q.task
    if task == "FAULT_CLASSIFICATION":
        semantics = "Rows contain selected-model class probability scores aligned to class_frame. They are model outputs, not FEC-calibrated prediction sets and not root-cause probabilities."
        downstream = ["FEC class conformal calibration", "DOR class-frame alignment/diagnostic arbitration"]
        prohibited = ["treat score max as final diagnosis confidence", "reinterpret scores as scalar FEC anomaly E", "infer physical root-cause directly from class probability"]
    else:
        semantics = "Rows contain a selected/retained detector anomaly ranking score. Higher means more anomalous for that detector; no operating threshold is frozen by M2N."
        downstream = ["FEC normal-reference calibration", "DOR anomaly-task arbitration"]
        prohibited = ["treat score as calibrated anomaly probability", "derive fault class from scalar anomaly score"]
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "contract": "M2N-DIAGNOSTIC-SCORE-PACK/1.0",
        "task": task,
        "candidate_id": candidate.id,
        "family": candidate.family,
        "feature_view": candidate.view,
        "feature_columns": list(candidate.feature_columns),
        "mechanism_feature_columns": list(q.mechanism_feature_columns),
        "class_frame": list(q.class_frame),
        "semantics": semantics,
        "eligible_downstream": downstream,
        "prohibited_claims": prohibited,
        "data_sha256": data_sha256,
        "request_contract": request.get("contract"),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def run_diagnostic_model_design(
    request_path: str | Path,
    data_path: str | Path,
    output_dir: str | Path,
) -> DiagnosticDesignSummary:
    request = _load_request(request_path)
    df = _read_data(data_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    q = qualify_diagnostic_request(request, df)
    (out / "diagnostic_request.json").write_text(json.dumps(request, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "qualification.json").write_text(json.dumps(q.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    artifacts: list[str] = ["diagnostic_request.json", "qualification.json"]
    if q.status == "UNSUPPORTED":
        decision = DiagnosticSelectionDecision("UNSUPPORTED", None, None, "NOT_EVALUATED", (), 0.0, 0.0, q.reasons)
        (out / "selection.json").write_text(json.dumps(decision.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        artifacts.append("selection.json")
        return DiagnosticDesignSummary("UNSUPPORTED", q.task, None, str(out), q, decision, tuple(artifacts))

    masks, _, _ = _split_masks(request, df)
    candidates = build_diagnostic_candidates(request, q)
    (out / "candidate_registry.json").write_text(
        json.dumps({"contract": "M2N-DIAGNOSTIC-CANDIDATES/1.0", "candidates": [c.to_dict() for c in candidates]}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    artifacts.append("candidate_registry.json")
    admitted = [c for c in candidates if c.admitted]
    if not admitted:
        decision = DiagnosticSelectionDecision("UNSUPPORTED", None, None, "NOT_EVALUATED", (), 0.0, 0.0, ("No diagnostic candidate survived model-family qualification.",))
        (out / "selection.json").write_text(json.dumps(decision.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        artifacts.append("selection.json")
        return DiagnosticDesignSummary("UNSUPPORTED", q.task, None, str(out), q, decision, tuple(artifacts))

    training = request.get("training") or {}
    seeds = tuple(int(x) for x in (training.get("restarts") or (101, 102, 103)))
    if len(seeds) < 2:
        raise ValueError("training.restarts must contain at least two independent seeds")
    policy = _selection_policy(request, q.task)
    candidate_metrics: dict[str, list[dict[str, Any]]] = {}
    score_map: dict[str, list[float]] = {}
    failures: dict[str, list[str]] = {}
    validation_has_anomaly_labels = _anomaly_labels(request, df, masks["validation"]) is not None if q.task == "ANOMALY_DETECTION" else True

    for c in admitted:
        candidate_metrics[c.id] = []
        score_map[c.id] = []
        failures[c.id] = []
        for seed in seeds:
            try:
                _, metrics = _evaluate_candidate(request, q, c, df, masks, seed)
                rec: dict[str, Any] = {"seed": seed, **metrics}
                candidate_metrics[c.id].append(rec)
                if policy.primary_metric in metrics:
                    score_map[c.id].append(float(metrics[policy.primary_metric]))
            except Exception as e:
                failures[c.id].append(f"seed={seed}: {type(e).__name__}: {e}")
    metrics_payload = {
        "contract": "M2N-DIAGNOSTIC-VALIDATION/1.0",
        "task": q.task,
        "primary_metric": policy.primary_metric,
        "selection_uses_split": "validation",
        "calibration_and_test_excluded_from_selection": True,
        "metrics": candidate_metrics,
        "failures": failures,
    }
    (out / "validation_metrics.json").write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    artifacts.append("validation_metrics.json")

    decision = _select_from_scores(
        score_map,
        policy,
        no_labels_for_detection=(q.task == "ANOMALY_DETECTION" and not validation_has_anomaly_labels),
    )
    (out / "selection.json").write_text(json.dumps(decision.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    artifacts.append("selection.json")

    # Freeze only candidates that are genuinely retained by the selection contract.
    # A real-device run exposed that freezing *every* valid candidate under AMBIGUOUS
    # can be both computationally wasteful and semantically wrong: clearly inferior
    # candidates are not part of the practical-equivalence set.
    valid_ids = {c.id for c in admitted if len(candidate_metrics.get(c.id, ())) >= 2}
    retained_ids: list[str] = []
    if decision.status in ("SELECTED", "FIXED") and decision.selected_candidate_id:
        retained_ids = [decision.selected_candidate_id]
    elif decision.status == "AMBIGUOUS":
        finite_scores = [(cid, float(score)) for cid, score in decision.candidate_median_scores if np.isfinite(score) and cid in valid_ids]
        if finite_scores:
            best_score = finite_scores[0][1]
            retained_ids = [cid for cid, score in finite_scores if (best_score - score) <= policy.practical_margin + 1e-12]
    elif decision.status == "PORTFOLIO":
        retained_ids = [c.id for c in admitted if c.id in valid_ids]
    final_candidates = [c for c in admitted if c.id in set(retained_ids)]

    models_dir = out / "models"
    model_index: list[dict[str, Any]] = []
    final_seed = int(training.get("final_seed", 9001))
    for c in final_candidates:
        try:
            model = _fit_final_model(request, q, c, df, masks, final_seed)
            models_dir.mkdir(parents=True, exist_ok=True)
            mp = models_dir / f"{_safe_id(c.id)}.joblib"
            joblib.dump(model, mp)
            model_index.append({
                "candidate_id": c.id,
                "family": c.family,
                "view": c.view,
                "feature_columns": list(c.feature_columns),
                "artifact": str(mp.relative_to(out)),
                "sha256": _sha256(mp),
                "final_fit_split": "train+validation",
                "final_seed": final_seed,
            })
            artifacts.append(str(mp.relative_to(out)))
        except Exception as e:
            failures.setdefault(c.id, []).append(f"final_fit: {type(e).__name__}: {e}")

    # Produce downstream score packs for every retained classifier. This preserves
    # AMBIGUOUS as a first-class result: FEC can calibrate each retained model and
    # DOR can arbitrate agreement/alternatives without forcing a pre-calibration top-1.
    score_artifacts: dict[str, Any] = {}
    for c in final_candidates:
        rec = next((x for x in model_index if x["candidate_id"] == c.id), None)
        if rec is None:
            continue
        model = joblib.load(out / rec["artifact"])
        c_art: dict[str, str] = {}
        base = out / "scores" / _safe_id(c.id)
        if q.split_counts.get("calibration", 0) > 0:
            p = _write_candidate_score_pack(
                task=q.task, model=model, candidate=c, q=q, request=request, df=df,
                mask=masks["calibration"], path=base / "calibration_scores.csv", include_labels=True,
            )
            c_art["calibration"] = str(p.relative_to(out)); artifacts.append(str(p.relative_to(out)))
        if q.split_counts.get("test", 0) > 0:
            p = _write_candidate_score_pack(
                task=q.task, model=model, candidate=c, q=q, request=request, df=df,
                mask=masks["test"], path=base / "test_scores.csv", include_labels=bool((request.get("data") or {}).get("include_test_labels", False)),
            )
            c_art["test"] = str(p.relative_to(out)); artifacts.append(str(p.relative_to(out)))
        cp = _write_score_contract(
            out / "score_contracts" / f"{_safe_id(c.id)}.json", request=request, q=q, candidate=c, data_sha256=_sha256(Path(data_path)),
        )
        c_art["contract"] = str(cp.relative_to(out)); artifacts.append(str(cp.relative_to(out)))
        score_artifacts[c.id] = c_art

    # Backward-compatible aliases for the unique selected/fixed case.
    if decision.status in ("SELECTED", "FIXED") and decision.selected_candidate_id in score_artifacts:
        one = score_artifacts[decision.selected_candidate_id]
        if "calibration" in one:
            srcp = out / one["calibration"]; dstp = out / "scores" / "calibration_scores.csv"
            dstp.parent.mkdir(parents=True, exist_ok=True); dstp.write_bytes(srcp.read_bytes())
            artifacts.append(str(dstp.relative_to(out)))
        if "test" in one:
            srcp = out / one["test"]; dstp = out / "scores" / "test_scores.csv"
            dstp.parent.mkdir(parents=True, exist_ok=True); dstp.write_bytes(srcp.read_bytes())
            artifacts.append(str(dstp.relative_to(out)))
        csrc = out / one["contract"]; cdst = out / "score_contract.json"
        cdst.write_bytes(csrc.read_bytes()); artifacts.append(str(cdst.relative_to(out)))

    bundle = {
        "contract": "M2N-DIAGNOSTIC-MODEL-BUNDLE/1.0",
        "producer": "M2N",
        "intended_consumers": ["FEC", "DOR"],
        "task": q.task,
        "qualification_status": q.status,
        "selection": decision.to_dict(),
        "class_frame": list(q.class_frame),
        "raw_feature_columns": list(q.raw_feature_columns),
        "mechanism_feature_columns": list(q.mechanism_feature_columns),
        "models": model_index,
        "retained_candidate_ids": retained_ids,
        "score_artifacts": score_artifacts,
        "calibration_ready": q.calibration_ready,
        "prohibited_claims": [
            "M2N model selection is not final fault diagnosis",
            "uncalibrated class scores are not DOR root-cause probabilities",
            "anomaly ranking scores are not calibrated anomaly probabilities",
            "mechanism-feature usefulness does not by itself prove physical root cause",
        ],
        "request_sha256": _sha256(Path(request_path)),
        "data_sha256": _sha256(Path(data_path)),
    }
    (out / "diagnostic_model_bundle.json").write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    artifacts.append("diagnostic_model_bundle.json")

    # Refresh validation payload with final-fit failures if any occurred.
    metrics_payload["failures"] = failures
    (out / "validation_metrics.json").write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return DiagnosticDesignSummary(decision.status, q.task, decision.selected_candidate_id, str(out), q, decision, tuple(artifacts))


def predict_diagnostic_design(
    design_dir: str | Path,
    data_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Run a selected diagnostic model, or all retained models if selection was ambiguous.

    This is an inference/export utility. It does not perform FEC calibration or DOR
    arbitration. Classification output columns are candidate-scoped class scores;
    anomaly output columns are candidate-scoped ranking scores.
    """
    root = Path(design_dir)
    bundle = json.loads((root / "diagnostic_model_bundle.json").read_text(encoding="utf-8"))
    if bundle.get("contract") != "M2N-DIAGNOSTIC-MODEL-BUNDLE/1.0":
        raise ValueError("unsupported diagnostic model bundle")
    df = _read_data(data_path)
    out = pd.DataFrame(index=np.arange(len(df)))
    task = str(bundle["task"])
    class_frame = tuple(map(str, bundle.get("class_frame") or ()))
    for rec in bundle.get("models") or []:
        cid = str(rec["candidate_id"])
        cols = tuple(map(str, rec["feature_columns"]))
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"input data missing features required by {cid}: {missing}")
        X = df.loc[:, list(cols)].to_numpy(dtype=float)
        if not np.all(np.isfinite(X)):
            raise ValueError(f"input data for {cid} contains non-finite values; no implicit imputation is allowed")
        model = joblib.load(root / rec["artifact"])
        prefix = _safe_id(cid)
        if task == "FAULT_CLASSIFICATION":
            p = _aligned_probabilities(model, X, class_frame)
            pred = np.asarray(model.predict(X)).astype(str)
            out[f"{prefix}__prediction"] = pred
            for j, c in enumerate(class_frame):
                out[f"{prefix}__score__{c}"] = p[:, j]
        else:
            out[f"{prefix}__anomaly_score"] = model.anomaly_score(X)
    if out.shape[1] == 0:
        raise RuntimeError("diagnostic bundle contains no runnable model artifacts")
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)
    return p
