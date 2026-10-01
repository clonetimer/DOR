#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def _model_candidates() -> dict[str, Any]:
    return {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=4000, random_state=9001)),
        ]),
        "random_forest": RandomForestClassifier(n_estimators=400, random_state=9001, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=9001, max_iter=300),
    }


def _metrics(y: pd.Series | np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def _duplicate_overlap(df: pd.DataFrame, feature_columns: list[str], left_role: str, right_role: str) -> int:
    left = pd.util.hash_pandas_object(df[df["split"].eq(left_role)][feature_columns], index=False)
    right = pd.util.hash_pandas_object(df[df["split"].eq(right_role)][feature_columns], index=False)
    return int(len(set(left.astype("uint64").tolist()) & set(right.astype("uint64").tolist())))


def run(data_path: Path, request_path: Path, out_path: Path, seed: int = 20260930) -> dict[str, Any]:
    df = pd.read_csv(data_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    raw = list(request["features"]["raw_columns"])
    mech = list(request["features"]["mechanism_columns"])
    features = raw + mech
    label = request["supervision"]["label_column"]

    roles = {role: df[df["split"].eq(role)].copy() for role in ("train", "val", "cal", "test")}
    groups = {role: set(frame["split_group"].astype(str)) for role, frame in roles.items()}
    group_overlaps = {
        f"{a}__{b}": sorted(groups[a] & groups[b])
        for i, a in enumerate(groups)
        for b in list(groups)[i + 1 :]
    }
    duplicate_overlaps = {
        f"{a}__{b}": _duplicate_overlap(df, features, a, b)
        for i, a in enumerate(groups)
        for b in list(groups)[i + 1 :]
    }

    x_train, y_train = roles["train"][features], roles["train"][label]
    x_val, y_val = roles["val"][features], roles["val"][label]
    x_test, y_test = roles["test"][features], roles["test"][label]

    val_results: dict[str, dict[str, float]] = {}
    fitted = {}
    for name, model in _model_candidates().items():
        model.fit(x_train, y_train)
        val_results[name] = _metrics(y_val, model.predict(x_val))
        fitted[name] = model
    selected_name = max(val_results, key=lambda n: (val_results[n]["macro_f1"], val_results[n]["accuracy"], n))
    selected = fitted[selected_name]
    test_metrics = _metrics(y_test, selected.predict(x_test))

    nuisance_cols = [c for c in ("cooler", "valve", "pump") if c in df.columns]
    nuisance = Pipeline([
        ("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), nuisance_cols)], remainder="drop")),
        ("model", LogisticRegression(max_iter=3000, random_state=9001)),
    ])
    nuisance.fit(roles["train"][nuisance_cols], y_train)
    nuisance_metrics = _metrics(y_test, nuisance.predict(roles["test"][nuisance_cols]))

    # Same-family view comparison: isolates whether the provisional mechanism view has a material effect.
    hgb_raw = HistGradientBoostingClassifier(random_state=9001, max_iter=300).fit(roles["train"][raw], y_train)
    hgb_full = HistGradientBoostingClassifier(random_state=9001, max_iter=300).fit(x_train, y_train)
    raw_metrics = _metrics(y_test, hgb_raw.predict(roles["test"][raw]))
    full_metrics = _metrics(y_test, hgb_full.predict(x_test))

    rng = np.random.default_rng(seed)
    train_std = x_train.std(axis=0, ddof=0).replace(0, 1.0).to_numpy(float)
    noise_stress = []
    for scale in (0.01, 0.05, 0.10, 0.20):
        noisy = x_test.to_numpy(float) + rng.normal(0.0, scale, size=x_test.shape) * train_std
        pred = selected.predict(pd.DataFrame(noisy, columns=features, index=x_test.index))
        noise_stress.append({"noise_std_fraction": scale, **_metrics(y_test, pred)})

    # Artificial missing-sensor stress. This is not counted as real production evidence.
    sensor_prefixes = sorted({c.split("__", 1)[0] for c in raw if "__" in c})
    missing_sensor = []
    for sensor in sensor_prefixes:
        cols = [c for c in features if c.startswith(sensor + "__")]
        if not cols:
            continue
        masked = x_test.copy()
        masked[cols] = np.nan
        try:
            pred = selected.predict(masked)
            item = {"sensor": sensor, "masked_columns": len(cols), "supported_by_estimator": True, **_metrics(y_test, pred)}
        except Exception as exc:
            item = {"sensor": sensor, "masked_columns": len(cols), "supported_by_estimator": False, "error": f"{type(exc).__name__}: {exc}"}
        missing_sensor.append(item)

    result = {
        "contract": "DOR-HYDRAULIC-PRACTICAL-STRESS/1.0",
        "dataset": str(data_path.relative_to(ROOT)) if data_path.is_relative_to(ROOT) else str(data_path),
        "request": str(request_path.relative_to(ROOT)) if request_path.is_relative_to(ROOT) else str(request_path),
        "row_counts": {k: int(len(v)) for k, v in roles.items()},
        "split_audit": {
            "split_group_overlaps": group_overlaps,
            "all_split_groups_disjoint": all(not v for v in group_overlaps.values()),
            "full_feature_exact_duplicate_overlap_counts": duplicate_overlaps,
            "no_cross_split_exact_feature_duplicates": all(v == 0 for v in duplicate_overlaps.values()),
        },
        "candidate_selection": {
            "validation_results": val_results,
            "selected": selected_name,
            "test_labels_used_for_selection": False,
            "test": test_metrics,
        },
        "nuisance_only_baseline": {"columns": nuisance_cols, "test": nuisance_metrics},
        "mechanism_view": {
            "same_family": "hist_gradient_boosting",
            "raw_test": raw_metrics,
            "raw_plus_mechanism_test": full_metrics,
            "macro_f1_delta": float(full_metrics["macro_f1"] - raw_metrics["macro_f1"]),
            "declared_practical_margin": float(request.get("selection", {}).get("practical_margin", 0.02)),
        },
        "synthetic_noise_stress": noise_stress,
        "synthetic_missing_sensor_stress": missing_sensor,
        "evidence_scope": {
            "real_test_rig_derived_subset": True,
            "full_official_2205_cycle_raw_benchmark": False,
            "new_asset_or_new_session_validation": False,
            "synthetic_stress_is_not_real_distribution_shift_evidence": True,
            "synthetic_missing_sensor_is_not_operational_missing_sensor_qualification": True,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "examples/hydraulic_accumulator/hydraulic_accumulator_features.csv"))
    p.add_argument("--request", default=str(ROOT / "validation/hydraulic_accumulator/m2n_request_effective.json"))
    p.add_argument("--out", default=str(ROOT / "validation/production_evidence/hydraulic_practical_stress.json"))
    args = p.parse_args()
    result = run(Path(args.data), Path(args.request), Path(args.out))
    print(json.dumps({"state": "OK", "selected": result["candidate_selection"]["selected"], "test": result["candidate_selection"]["test"], "nuisance": result["nuisance_only_baseline"]["test"], "mechanism_delta": result["mechanism_view"]["macro_f1_delta"], "out": str(Path(args.out).resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
