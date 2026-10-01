#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

CANDIDATES: tuple[dict[str, Any], ...] = (
    {"name": "standardized_energy", "family": "energy"},
    *({"name": f"pca_reconstruction_{n}", "family": "pca", "n_components": n} for n in (2, 3, 5, 8, 12, 16)),
    *({"name": f"knn_distance_{k}", "family": "knn", "k": k} for k in (3, 5, 10, 20)),
    {"name": "ledoit_wolf_mahalanobis", "family": "ledoit_wolf"},
    *({"name": f"isolation_forest_mf_{mf}", "family": "isolation_forest", "max_features": mf} for mf in (0.5, 0.75, 1.0)),
)


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _fit_score(train: pd.DataFrame, frames: list[pd.DataFrame], feature_columns: list[str], candidate: dict[str, Any]) -> list[np.ndarray]:
    tr = train[feature_columns].replace([np.inf, -np.inf], np.nan)
    medians = tr.median()
    trn = tr.fillna(medians).to_numpy(float)
    scaler = StandardScaler().fit(trn)
    matrices = [scaler.transform(f[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(medians).to_numpy(float)) for f in [train, *frames]]
    x_train = matrices[0]
    family = candidate["family"]

    if family == "energy":
        score = lambda x: np.mean(x * x, axis=1)
    elif family == "pca":
        n = min(int(candidate["n_components"]), x_train.shape[1], max(1, x_train.shape[0] - 1))
        model = PCA(n_components=n, random_state=0).fit(x_train)
        score = lambda x: np.mean((x - model.inverse_transform(model.transform(x))) ** 2, axis=1)
    elif family == "knn":
        k = min(int(candidate["k"]), len(x_train))
        model = NearestNeighbors(n_neighbors=k).fit(x_train)
        score = lambda x: model.kneighbors(x, return_distance=True)[0][:, -1]
    elif family == "ledoit_wolf":
        model = LedoitWolf().fit(x_train)
        score = lambda x: model.mahalanobis(x)
    elif family == "isolation_forest":
        model = IsolationForest(
            n_estimators=300,
            contamination="auto",
            max_features=float(candidate["max_features"]),
            random_state=101,
        ).fit(x_train)
        score = lambda x: -model.score_samples(x)
    else:
        raise ValueError(f"unsupported family: {family}")

    return [score(x) for x in matrices]


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    has_both = bool(np.any(y_true == 0) and np.any(y_true == 1))
    return {
        "n": int(len(y_true)),
        "normal_n": int(np.sum(y_true == 0)),
        "anomaly_n": int(np.sum(y_true == 1)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if has_both else None,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def _baseline_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False}
    df = pd.read_csv(path)
    supported = df[df["diagnosis_status"].astype(str).eq("SUPPORTED")].copy()
    y_true = supported["true_label"].astype(int).to_numpy()
    label_col = "diagnosis" if "diagnosis" in supported.columns else "decision"
    y_pred = supported[label_col].astype(str).str.upper().eq("ABNORMAL").astype(int).to_numpy()
    overall = _binary_metrics(y_true, y_pred)
    channels = {}
    for ch, g in supported.groupby("channel"):
        yy = g["true_label"].astype(int).to_numpy()
        pp = g[label_col].astype(str).str.upper().eq("ABNORMAL").astype(int).to_numpy()
        channels[str(ch)] = _binary_metrics(yy, pp)
    return {"available": True, "supported_n": int(len(supported)), "overall": overall, "channels": channels}


def run(data_path: Path, request_path: Path, baseline_path: Path, out_dir: Path, alpha: float = 0.1, min_calibration: int = 20) -> dict[str, Any]:
    data = pd.read_csv(data_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    features = list(request["features"]["raw_columns"])
    missing = [c for c in features if c not in data.columns]
    if missing:
        raise ValueError(f"declared OPSSAT feature columns missing: {missing}")

    prediction_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []

    for channel in sorted(data["channel"].astype(str).unique()):
        d = data[data["channel"].astype(str).eq(channel)].copy()
        train = d[(d["split"] == "train") & (d["fault_label"].astype(int) == 0)]
        val = d[d["split"] == "val"]
        cal = d[(d["split"] == "cal") & (d["fault_label"].astype(int) == 0)]
        test = d[d["split"] == "test"]

        if len(cal) < min_calibration:
            selection_rows.append({
                "channel": channel,
                "status": "ABSTAIN_CALIBRATION_POWER",
                "selected_candidate": None,
                "train_normal_n": int(len(train)), "validation_n": int(len(val)), "calibration_normal_n": int(len(cal)), "test_n": int(len(test)),
                "selection_basis": f"calibration_normal_n={len(cal)} < {min_calibration}",
            })
            for _, row in test.iterrows():
                prediction_rows.append({
                    "segment": row["segment"], "channel": channel, "true_label": int(row["fault_label"]),
                    "status": "ABSTAIN", "selected_candidate": None, "anomaly_score": None, "p_value": None, "prediction": None,
                })
            continue

        ranked = []
        y_val = val["fault_label"].astype(int).to_numpy()
        has_val_pos = bool(np.any(y_val == 1) and np.any(y_val == 0))
        for order, candidate in enumerate(CANDIDATES):
            train_scores, val_scores, cal_scores, test_scores = _fit_score(train, [val, cal, test], features, candidate)
            if has_val_pos:
                ap = float(average_precision_score(y_val, val_scores))
                auc = float(roc_auc_score(y_val, val_scores))
                ranking = (1, ap, auc, -order)
                meta = {"validation_average_precision": ap, "validation_roc_auc": auc, "validation_normal_fpr_at_train_q90": None}
            else:
                train_q90 = float(np.quantile(train_scores, 0.90))
                normal_fpr = float(np.mean(val_scores > train_q90))
                ranking = (0, -normal_fpr, -order, 0)
                meta = {"validation_average_precision": None, "validation_roc_auc": None, "validation_normal_fpr_at_train_q90": normal_fpr}
            ranked.append((ranking, candidate, cal_scores, test_scores, meta))

        ranked.sort(key=lambda x: x[0], reverse=True)
        _, selected, cal_scores, test_scores, selected_meta = ranked[0]
        p_values = (1 + (cal_scores[:, None] >= test_scores[None, :]).sum(axis=0)) / (len(cal_scores) + 1)
        predictions = (p_values <= alpha).astype(int)

        selection_rows.append({
            "channel": channel,
            "status": "SUPPORTED",
            "selected_candidate": selected["name"],
            "candidate_family": selected["family"],
            "train_normal_n": int(len(train)), "validation_n": int(len(val)), "validation_anomaly_n": int(np.sum(y_val == 1)),
            "calibration_normal_n": int(len(cal)), "test_n": int(len(test)),
            "selection_basis": "validation anomaly ranking by average precision then ROC-AUC" if has_val_pos else "no validation anomalies; minimize validation normal exceedance against train-only q90",
            **selected_meta,
        })
        for (_, row), score, pval, pred in zip(test.iterrows(), test_scores, p_values, predictions):
            prediction_rows.append({
                "segment": row["segment"], "channel": channel, "true_label": int(row["fault_label"]),
                "status": "SUPPORTED", "selected_candidate": selected["name"], "anomaly_score": float(score), "p_value": float(pval), "prediction": int(pred),
            })

    pred_df = pd.DataFrame(prediction_rows)
    selection_df = pd.DataFrame(selection_rows)
    supported = pred_df[pred_df["status"].eq("SUPPORTED")].copy()
    overall = _binary_metrics(supported["true_label"].astype(int).to_numpy(), supported["prediction"].astype(int).to_numpy())
    channel_metrics: dict[str, Any] = {}
    for ch, g in supported.groupby("channel"):
        channel_metrics[str(ch)] = _binary_metrics(g["true_label"].astype(int).to_numpy(), g["prediction"].astype(int).to_numpy())

    baseline = _baseline_metrics(baseline_path)
    result = {
        "contract": "DOR-OPSSAT-CHANNELWISE-REMEDIATION/1.0",
        "dataset": str(data_path.relative_to(ROOT)) if data_path.is_relative_to(ROOT) else str(data_path),
        "request": str(request_path.relative_to(ROOT)) if request_path.is_relative_to(ROOT) else str(request_path),
        "alpha": alpha,
        "min_calibration_normal_rows": min_calibration,
        "feature_count": len(features),
        "candidate_count": len(CANDIDATES),
        "supported_n": int(len(supported)),
        "abstain_n": int(len(pred_df) - len(supported)),
        "supported_fraction": float(len(supported) / len(pred_df)),
        "overall": overall,
        "channel_metrics": channel_metrics,
        "baseline_current_release": baseline,
        "delta_vs_baseline": {
            "overall_f1": float(overall["f1"] - baseline["overall"]["f1"]) if baseline.get("available") else None,
            "overall_balanced_accuracy": float(overall["balanced_accuracy"] - baseline["overall"]["balanced_accuracy"]) if baseline.get("available") and baseline["overall"]["balanced_accuracy"] is not None else None,
        },
        "selection_safety": {
            "model_fit_uses_train_normal_only": True,
            "candidate_selection_uses_validation_only": True,
            "threshold_calibration_uses_calibration_normal_only": True,
            "test_labels_used_for_candidate_selection": False,
            "test_labels_used_for_threshold_calibration": False,
            "historical_test_results_were_known_before_this_remediation": True,
            "therefore_this_is_development_remediation_not_a_fresh_blind_benchmark": True,
        },
        "limitations": [
            "The candidate set was introduced after historical test performance was already known, so this run cannot retroactively become a blind qualification result.",
            "Per-channel models improve heterogeneity handling but increase model-management and monitoring burden.",
            "Channels with fewer than min_calibration_normal_rows remain fail-closed.",
        ],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    selection_df.to_csv(out_dir / "candidate_selection.csv", index=False)
    _json_dump(out_dir / "summary.json", result)
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "examples/opssat_ad/opssat_p1_features.csv"))
    p.add_argument("--request", default=str(ROOT / "validation/opssat_ad/m2n_request_effective.json"))
    p.add_argument("--baseline", default=str(ROOT / "validation/opssat_ad/diagnosis.csv"))
    p.add_argument("--out", default=str(ROOT / "validation/production_evidence/opssat_channelwise"))
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--min-calibration", type=int, default=20)
    args = p.parse_args()
    result = run(Path(args.data), Path(args.request), Path(args.baseline), Path(args.out), args.alpha, args.min_calibration)
    print(json.dumps({"state": "OK", "supported_fraction": result["supported_fraction"], "overall": result["overall"], "delta_vs_baseline": result["delta_vs_baseline"], "out": str(Path(args.out).resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
