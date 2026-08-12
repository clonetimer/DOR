from __future__ import annotations

"""DOR v0.4 context hierarchy and reference-applicability control plane.

DOR does not recalibrate scores itself.  It decides which *declared* context
level is eligible for a row and lets FEC fit/apply an exact reference at that
level.  Coarser-level fallback is therefore explicit, provenance-bound, and
fail-closed.  No hierarchy is discovered from held-out test labels or scores.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable
import json

import pandas as pd


REFERENCE_STATES = {
    "EXACT_SUPPORTED",
    "HIERARCHY_SUPPORTED",
    "POWER_LIMITED",
    "CONTEXT_UNSUPPORTED",
}


@dataclass(frozen=True)
class ContextLevel:
    level_id: str
    columns: tuple[str, ...]
    min_group_size: int
    fallback_allowed: bool
    sharing_basis: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["columns"] = list(self.columns)
        return d


@dataclass(frozen=True)
class ReferenceApplicabilityRow:
    row_id: str
    status: str
    selected_level: str | None
    selected_columns: tuple[str, ...]
    selected_key: tuple[str, ...]
    n_reference: int
    sharing_basis: str
    reason: str

    def __post_init__(self) -> None:
        if self.status not in REFERENCE_STATES:
            raise ValueError(f"invalid reference applicability state: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["selected_columns"] = list(self.selected_columns)
        d["selected_key"] = list(self.selected_key)
        return d


def _as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def normalize_context_levels(manifest: dict[str, Any]) -> tuple[ContextLevel, ...]:
    """Return finest->coarsest declared levels.

    Backward-compatible v0.3 EXACT manifests become a single exact level.
    A coarser level is never admitted unless `fallback_allowed=true` and a
    non-empty `sharing_basis` is declared.  Empty-column global levels are
    allowed, but are not created implicitly.
    """
    context = manifest.get("context") or {}
    calibration = manifest.get("calibration") or {}
    policy = str(context.get("policy", "EXACT")).upper()
    if policy not in {"EXACT", "HIERARCHICAL"}:
        raise ValueError(f"unsupported context policy: {policy}")
    fallback_policy = str(context.get("fallback_policy", "DECLARED_ANCESTOR_ONLY")).upper()
    if fallback_policy != "DECLARED_ANCESTOR_ONLY":
        raise ValueError("DOR v0.4 only permits fallback_policy=DECLARED_ANCESTOR_ONLY")
    default_min = int(calibration.get("min_group_size", 20))
    levels_raw = _as_list(context.get("levels"))
    if not levels_raw:
        cols = tuple(str(x) for x in _as_list(context.get("columns")))
        return (
            ContextLevel(
                level_id="exact" if cols else "global",
                columns=cols,
                min_group_size=default_min,
                fallback_allowed=False,
                sharing_basis="exact declared context" if cols else "global context declared by dataset contract",
            ),
        )

    levels: list[ContextLevel] = []
    seen_ids: set[str] = set()
    seen_columns: set[tuple[str, ...]] = set()
    for i, raw in enumerate(levels_raw):
        if not isinstance(raw, dict):
            raise ValueError("context.levels entries must be objects")
        lid = str(raw.get("level_id") or f"level_{i}")
        if lid in seen_ids:
            raise ValueError(f"duplicate context level_id: {lid}")
        cols = tuple(str(x) for x in _as_list(raw.get("columns")))
        if cols in seen_columns:
            raise ValueError(f"duplicate context level columns: {cols}")
        min_n = int(raw.get("min_group_size", default_min))
        if min_n < 2:
            raise ValueError("context level min_group_size must be >=2")
        fallback = bool(raw.get("fallback_allowed", False))
        basis = str(raw.get("sharing_basis") or "").strip()
        if i == 0:
            # Finest level is the exact declared context and is always eligible
            # on its own key; fallback flag has no meaning here.
            fallback = False
            if not basis:
                basis = "exact declared context"
        elif fallback and not basis:
            raise ValueError(f"coarser context level {lid!r} allows fallback but declares no sharing_basis")
        levels.append(ContextLevel(lid, cols, min_n, fallback, basis))
        seen_ids.add(lid); seen_columns.add(cols)

    # Require monotone coarsening by dropping columns, never inventing new ones.
    if policy == "EXACT" and any(level.fallback_allowed for level in levels[1:]):
        raise ValueError("context policy EXACT cannot admit coarser fallback levels")
    for fine, coarse in zip(levels, levels[1:]):
        if not set(coarse.columns).issubset(set(fine.columns)):
            raise ValueError(
                f"context hierarchy is not nested: {coarse.level_id} columns {coarse.columns} "
                f"must be a subset of finer level {fine.level_id} columns {fine.columns}"
            )
        if len(coarse.columns) >= len(fine.columns):
            raise ValueError("each coarser context level must remove at least one context column")
    return tuple(levels)


def _key_df(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    cols = list(columns)
    if not cols:
        return pd.Series(["__GLOBAL__"] * len(df), index=df.index, dtype=object)
    if len(cols) == 1:
        return df[cols[0]].astype(str)
    return df[cols].astype(str).agg("|".join, axis=1)


def _split_masks(manifest: dict[str, Any], df: pd.DataFrame) -> dict[str, pd.Series]:
    split = manifest.get("split") or {}
    col = str(split.get("column", "split"))
    def mask(name: str) -> pd.Series:
        vals = {str(x) for x in _as_list(split.get(f"{name}_values"))}
        return df[col].astype(str).isin(vals)
    return {k: mask(k) for k in ("train", "validation", "calibration", "test")}


def build_reference_applicability(
    manifest: dict[str, Any],
    data_path: str | Path,
    *,
    id_column: str | None = None,
) -> dict[str, Any]:
    """Compute reference eligibility using only declared context + role counts.

    Test labels and test scores are not read.  The result is deterministic once
    the DatasetManifest and data-role assignment are frozen.
    """
    df = pd.read_csv(data_path)
    levels = normalize_context_levels(manifest)
    masks = _split_masks(manifest, df)
    ids = [str(x) for x in _as_list(manifest.get("identifiers"))]
    if id_column is None:
        id_column = ids[0] if ids else None
    if id_column and id_column not in df.columns:
        raise ValueError(f"reference applicability id_column not found: {id_column}")

    required_cols = sorted(set(c for level in levels for c in level.columns))
    missing = sorted(set(required_cols) - set(df.columns))
    if missing:
        raise ValueError(f"context hierarchy columns missing from data: {missing}")

    cal = df.loc[masks["calibration"]].copy()
    test = df.loc[masks["test"]].copy()

    level_counts: dict[str, dict[str, int]] = {}
    level_supported: dict[str, list[str]] = {}
    for level in levels:
        counts = _key_df(cal, level.columns).value_counts().to_dict()
        counts = {str(k): int(v) for k, v in counts.items()}
        level_counts[level.level_id] = counts
        level_supported[level.level_id] = sorted(k for k, n in counts.items() if n >= level.min_group_size)

    rows: list[ReferenceApplicabilityRow] = []
    status_counts = {s: 0 for s in REFERENCE_STATES}
    selected_level_counts: dict[str, int] = {}
    for pos, (idx, row) in enumerate(test.iterrows()):
        rid = str(row[id_column]) if id_column else str(pos)
        exact_level = levels[0]
        exact_key = str(_key_df(pd.DataFrame([row]), exact_level.columns).iloc[0])
        exact_n = int(level_counts[exact_level.level_id].get(exact_key, 0))
        if exact_n >= exact_level.min_group_size:
            rr = ReferenceApplicabilityRow(
                rid, "EXACT_SUPPORTED", exact_level.level_id, exact_level.columns,
                (exact_key,), exact_n, exact_level.sharing_basis,
                "finest declared context has sufficient independent calibration reference",
            )
        else:
            chosen: ReferenceApplicabilityRow | None = None
            for level in levels[1:]:
                if not level.fallback_allowed:
                    continue
                key = str(_key_df(pd.DataFrame([row]), level.columns).iloc[0])
                n = int(level_counts[level.level_id].get(key, 0))
                if n >= level.min_group_size:
                    chosen = ReferenceApplicabilityRow(
                        rid, "HIERARCHY_SUPPORTED", level.level_id, level.columns,
                        (key,), n, level.sharing_basis,
                        f"exact context has n={exact_n} < {exact_level.min_group_size}; "
                        f"declared ancestor {level.level_id} admitted with n={n}",
                    )
                    break
            if chosen is not None:
                rr = chosen
            elif exact_n > 0:
                rr = ReferenceApplicabilityRow(
                    rid, "POWER_LIMITED", None, (), (), exact_n, "",
                    f"exact context exists but has n={exact_n} < {exact_level.min_group_size}; no declared ancestor has adequate power",
                )
            else:
                # If an allowed ancestor exists but is itself sparse, call this
                # power-limited rather than structurally unsupported.
                sparse_ancestor = None
                for level in levels[1:]:
                    if not level.fallback_allowed:
                        continue
                    key = str(_key_df(pd.DataFrame([row]), level.columns).iloc[0])
                    n = int(level_counts[level.level_id].get(key, 0))
                    if n > 0:
                        sparse_ancestor = (level, key, n); break
                if sparse_ancestor:
                    level, key, n = sparse_ancestor
                    rr = ReferenceApplicabilityRow(
                        rid, "POWER_LIMITED", None, (), (), n, level.sharing_basis,
                        f"exact context is unseen and declared ancestor {level.level_id} has only n={n} < {level.min_group_size}",
                    )
                else:
                    rr = ReferenceApplicabilityRow(
                        rid, "CONTEXT_UNSUPPORTED", None, (), (), 0, "",
                        "no exact reference and no declared admissible ancestor with calibration support",
                    )
        rows.append(rr)
        status_counts[rr.status] += 1
        if rr.selected_level:
            selected_level_counts[rr.selected_level] = selected_level_counts.get(rr.selected_level, 0) + 1

    return {
        "contract": "DOR-REFERENCE-APPLICABILITY/0.4",
        "policy": str((manifest.get("context") or {}).get("policy", "EXACT")).upper(),
        "id_column": id_column,
        "levels": [x.to_dict() for x in levels],
        "calibration_reference_counts": level_counts,
        "supported_keys_by_level": level_supported,
        "test_rows": [x.to_dict() for x in rows],
        "summary": {
            "n_test_rows": len(rows),
            "status_counts": status_counts,
            "selected_level_counts": selected_level_counts,
            "supported_rows": status_counts["EXACT_SUPPORTED"] + status_counts["HIERARCHY_SUPPORTED"],
            "unsupported_rows": status_counts["POWER_LIMITED"] + status_counts["CONTEXT_UNSUPPORTED"],
        },
        "safety": {
            "test_labels_used": False,
            "test_scores_used": False,
            "implicit_pooling": False,
            "coarser_reference_requires_declared_fallback_and_sharing_basis": True,
            "coverage_claim": "FEC validity at a chosen level still depends on exchangeability within that declared reference group; DOR hierarchy admission is not itself a coverage theorem.",
        },
    }


def applicability_dataframe(report: dict[str, Any]) -> pd.DataFrame:
    rows = []
    id_col = report.get("id_column") or "row_id"
    for r in report.get("test_rows") or []:
        rows.append({
            id_col: str(r["row_id"]),
            "dor_reference_status": r["status"],
            "dor_reference_level": r.get("selected_level") or "",
            "dor_reference_columns": json.dumps(r.get("selected_columns") or [], ensure_ascii=False),
            "dor_reference_context": json.dumps(r.get("selected_key") or [], ensure_ascii=False),
            "dor_reference_n": int(r.get("n_reference") or 0),
            "dor_reference_sharing_basis": r.get("sharing_basis") or "",
            "dor_reference_reason": r.get("reason") or "",
        })
    return pd.DataFrame(rows)


def _as_bool_value(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    return str(x).strip().lower() in {"true", "1", "yes"}


def select_hierarchical_fec_output(
    level_outputs: dict[str, str | Path],
    applicability_report: dict[str, Any],
    output_csv: str | Path,
    *,
    id_column: str | None = None,
) -> Path:
    """Select the finest admitted FEC result per row.

    Each level output was calibrated independently by FEC at exactly that level.
    Rows without an admitted reference become explicit unsupported rows.  No
    numerical recalibration occurs here.
    """
    id_column = id_column or applicability_report.get("id_column")
    if not id_column:
        raise ValueError("id_column is required to merge hierarchical FEC outputs")
    app = applicability_dataframe(applicability_report)
    if app.empty:
        raise ValueError("empty reference applicability report")
    frames: dict[str, pd.DataFrame] = {}
    for lid, path in level_outputs.items():
        df = pd.read_csv(path, dtype={id_column: str})
        if id_column not in df.columns:
            raise ValueError(f"FEC output {path} lacks id column {id_column}")
        if df[id_column].duplicated().any():
            raise ValueError(f"FEC output {path} has duplicate IDs")
        frames[str(lid)] = df.set_index(id_column, drop=False)

    out_rows: list[dict[str, Any]] = []
    for _, ar in app.iterrows():
        sid = str(ar[id_column])
        lid = str(ar["dor_reference_level"])
        status = str(ar["dor_reference_status"])
        if lid and lid in frames and sid in frames[lid].index and status in {"EXACT_SUPPORTED", "HIERARCHY_SUPPORTED"}:
            base = frames[lid].loc[sid].to_dict()
            if isinstance(base, pd.DataFrame):
                raise ValueError(f"duplicate ID {sid} in level output {lid}")
        else:
            # Preserve just the row id and construct the standardized unsupported
            # fields expected by DOR arbiters.  Task-specific columns are added
            # conditionally below.
            base = {id_column: sid, "calibration_supported": False}
            any_frame = next(iter(frames.values()))
            for frame in frames.values():
                if sid in frame.index and "true_label" in frame.columns:
                    base["true_label"] = frame.loc[sid, "true_label"]; break
            if "decision" in any_frame.columns:
                base.update({"decision": status, "p_value": "", "calibration_n_reference": int(ar["dor_reference_n"])})
            if "prediction_set" in any_frame.columns:
                base.update({
                    "prediction_set": "[]", "prediction_set_size": 0,
                    "prediction_set_status": "ABSTAIN",
                    "calibration_n_reference": int(ar["dor_reference_n"]),
                })
        base["dor_reference_status"] = status
        base["dor_reference_level"] = lid
        base["dor_reference_columns"] = ar["dor_reference_columns"]
        base["dor_reference_context"] = ar["dor_reference_context"]
        base["dor_reference_n"] = int(ar["dor_reference_n"])
        base["dor_reference_sharing_basis"] = ar["dor_reference_sharing_basis"]
        base["dor_reference_reason"] = ar["dor_reference_reason"]
        if not _as_bool_value(base.get("calibration_supported", False)):
            base["calibration_reason"] = ar["dor_reference_reason"]
        out_rows.append(base)

    out = pd.DataFrame(out_rows)
    op = Path(output_csv); op.parent.mkdir(parents=True, exist_ok=True); out.to_csv(op, index=False)
    return op
