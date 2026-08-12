from __future__ import annotations

"""Persistence/CLI bridge for genuine diagnostic class score packs.

The bridge is deliberately outside Frozen Core.  It consumes an independently
produced class-probability score table (for example M2N's selected diagnostic
classifier), fits the existing mask-conditional conformal extension on a
labelled calibration split, persists that reference, and later emits
prediction sets or row-level abstentions.

It does not reinterpret scalar anomaly Evidence E as class evidence and it does
not perform root-cause reasoning.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence
import hashlib
import json

import numpy as np

from .mask_conditional_class import MaskConditionalClassConformal


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read_csv(path: str | Path):
    import csv
    p = Path(path)
    with p.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty class score CSV: {p}")
    return rows


def _class_frame(rows: Sequence[dict[str, str]]) -> tuple[str, ...]:
    cols = tuple(rows[0].keys())
    frame = tuple(c.split("score__", 1)[1] for c in cols if c.startswith("score__"))
    if len(frame) < 2:
        raise ValueError("class score pack must contain at least two score__<class> columns")
    if len(set(frame)) != len(frame):
        raise ValueError("duplicate class labels in score columns")
    return frame


def _probabilities(rows: Sequence[dict[str, str]], frame: Sequence[str]) -> np.ndarray:
    p = np.asarray([[float(r[f"score__{c}"]) for c in frame] for r in rows], dtype=float)
    if p.ndim != 2 or p.shape[1] < 2 or np.any(~np.isfinite(p)) or np.any(p < 0):
        raise ValueError("class scores must form a finite non-negative matrix")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("class score rows must sum to one; FEC will not silently normalize arbitrary scores")
    return p


def _mask_columns(rows: Sequence[dict[str, str]]) -> tuple[str, ...]:
    return tuple(c for c in rows[0].keys() if c.startswith("mask__"))


def _masks(rows: Sequence[dict[str, str]], mask_columns: Sequence[str]) -> np.ndarray:
    if not mask_columns:
        return np.ones((len(rows), 1), dtype=int)
    a = np.asarray([[int(bool(int(float(r[c])))) for c in mask_columns] for r in rows], dtype=int)
    if np.any(np.sum(a, axis=1) == 0):
        raise ValueError("all-zero modality mask is invalid")
    return a


@dataclass(frozen=True)
class ClassConformalReference:
    contract: str
    class_frame: tuple[str, ...]
    mask_columns: tuple[str, ...]
    alpha: float
    lambda_vacuity: float
    min_group_size: int
    groups: tuple[dict[str, Any], ...]
    calibration_score_sha256: str
    source_score_contract_sha256: str | None
    provenance: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["class_frame"] = list(self.class_frame)
        d["mask_columns"] = list(self.mask_columns)
        d["groups"] = [dict(x) for x in self.groups]
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ClassConformalReference":
        if d.get("contract") != "FEC-CLASS-CONFORMAL-REFERENCE/1.0":
            raise ValueError("unsupported class conformal reference contract")
        return ClassConformalReference(
            contract=d["contract"],
            class_frame=tuple(map(str, d["class_frame"])),
            mask_columns=tuple(map(str, d.get("mask_columns") or ())),
            alpha=float(d["alpha"]),
            lambda_vacuity=float(d.get("lambda_vacuity", 0.0)),
            min_group_size=int(d["min_group_size"]),
            groups=tuple(dict(x) for x in d["groups"]),
            calibration_score_sha256=str(d["calibration_score_sha256"]),
            source_score_contract_sha256=d.get("source_score_contract_sha256"),
            provenance=str(d.get("provenance", "unspecified")),
        )


def fit_class_score_reference(
    calibration_csv: str | Path,
    output_reference: str | Path,
    *,
    alpha: float = 0.10,
    lambda_vacuity: float = 0.0,
    min_group_size: int = 20,
    score_contract: str | Path | None = None,
    provenance: str = "external-selected-classifier",
) -> Path:
    rows = _read_csv(calibration_csv)
    frame = _class_frame(rows)
    if "true_label" not in rows[0]:
        raise ValueError("calibration score pack requires true_label")
    labels = np.asarray([str(r["true_label"]) for r in rows], dtype=object)
    unknown = sorted(set(labels) - set(frame))
    if unknown:
        raise ValueError(f"calibration labels are outside class_frame: {unknown}")
    label_index = {c: i for i, c in enumerate(frame)}
    y = np.asarray([label_index[str(x)] for x in labels], dtype=int)
    p = _probabilities(rows, frame)
    mask_cols = _mask_columns(rows)
    masks = _masks(rows, mask_cols)
    cal = MaskConditionalClassConformal(alpha=alpha, lambda_vacuity=lambda_vacuity, min_group_size=min_group_size).fit(p, y, masks)
    groups = []
    for m in cal.calibrated_masks:
        info = cal.info(m)
        groups.append({"mask": list(info.mask), "n_reference": info.n_reference, "threshold": info.threshold, "alpha": info.alpha})
    if not groups:
        raise ValueError("no modality mask has enough labeled calibration samples; FEC refuses to persist an empty class reference")
    sc_hash = None if score_contract is None else _sha256(Path(score_contract))
    ref = ClassConformalReference(
        "FEC-CLASS-CONFORMAL-REFERENCE/1.0", tuple(frame), tuple(mask_cols), float(alpha), float(lambda_vacuity), int(min_group_size),
        tuple(groups), _sha256(Path(calibration_csv)), sc_hash, str(provenance),
    )
    op = Path(output_reference)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(ref.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return op


def _restore_calibrator(ref: ClassConformalReference) -> MaskConditionalClassConformal:
    cal = MaskConditionalClassConformal(alpha=ref.alpha, lambda_vacuity=ref.lambda_vacuity, min_group_size=ref.min_group_size)
    n_modalities = len(ref.mask_columns) if ref.mask_columns else 1
    cal._n_classes = len(ref.class_frame)
    cal._n_modalities = n_modalities
    cal._thresholds = {tuple(int(x) for x in g["mask"]): float(g["threshold"]) for g in ref.groups}
    cal._counts = {tuple(int(x) for x in g["mask"]): int(g["n_reference"]) for g in ref.groups}
    return cal


def infer_class_prediction_sets(
    score_csv: str | Path,
    reference_json: str | Path,
    output_csv: str | Path,
) -> Path:
    import csv
    rows = _read_csv(score_csv)
    ref = ClassConformalReference.from_dict(json.loads(Path(reference_json).read_text(encoding="utf-8")))
    frame = _class_frame(rows)
    if tuple(frame) != ref.class_frame:
        raise ValueError(f"score class_frame {frame} does not match reference {ref.class_frame}")
    mask_cols = _mask_columns(rows)
    if tuple(mask_cols) != ref.mask_columns:
        raise ValueError(f"score modality-mask columns {mask_cols} do not match reference {ref.mask_columns}")
    p = _probabilities(rows, frame)
    masks = _masks(rows, mask_cols)
    cal = _restore_calibrator(ref)
    out_rows: list[dict[str, Any]] = []
    for i, (row, mask) in enumerate(zip(rows, masks)):
        key = tuple(int(x) for x in mask)
        base = {k: v for k, v in row.items() if not k.startswith("score__") and k != "true_label"}
        top = int(np.argmax(p[i]))
        base.update({"top_class": frame[top], "top_score": float(p[i, top]), "modality_mask": json.dumps(list(key))})
        if not cal.has_mask(key):
            base.update({
                "prediction_set": "[]",
                "prediction_set_size": 0,
                "calibration_supported": False,
                "calibration_reason": f"mask {key} lacks supported labeled calibration",
                "prediction_set_status": "ABSTAIN",
            })
        else:
            idx_set = cal.prediction_sets(p[i:i+1], masks[i:i+1])[0]
            labels = [frame[j] for j in idx_set]
            base.update({
                "prediction_set": json.dumps(labels, ensure_ascii=False),
                "prediction_set_size": len(labels),
                "calibration_supported": True,
                "calibration_reason": "mask-conditioned conformal reference supported",
                "prediction_set_status": "CLASS_SET_SUPPORTED" if labels else "ABSTAIN",
            })
        out_rows.append(base)
    op = Path(output_csv)
    op.parent.mkdir(parents=True, exist_ok=True)
    fields = list(out_rows[0].keys())
    with op.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(out_rows)
    return op
