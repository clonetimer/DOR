from __future__ import annotations

"""Held-out audit for modality-mask conditional class conformal references."""

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math

import numpy as np

from .class_score_bridge import (
    ClassConformalReference,
    _class_frame,
    _mask_columns,
    _masks,
    _probabilities,
    _read_csv,
    _restore_calibrator,
)


@dataclass(frozen=True)
class MaskCoverageAudit:
    mask: tuple[int, ...]
    n_test: int
    n_reference: int
    calibration_supported: bool
    empirical_coverage: float | None
    target_coverage: float
    coverage_shortfall: float | None
    mean_set_size: float | None
    singleton_rate: float | None
    empty_set_rate: float | None
    top_class_accuracy: float | None
    pass_test_support: bool
    pass_coverage: bool


@dataclass(frozen=True)
class MaskCalibrationAuditReport:
    contract: str
    reference_contract: str
    alpha: float
    class_frame: tuple[str, ...]
    mask_columns: tuple[str, ...]
    min_test_group_size: int
    groups: tuple[MaskCoverageAudit, ...]
    supported_fraction: float
    unsupported_rows: int
    all_observed_masks_supported: bool
    all_supported_groups_pass: bool
    claim_boundary: str = (
        "Held-out empirical audit only; finite-sample validity still requires exchangeability "
        "within each modality-mask stratum and is not instance-conditional coverage."
    )

    def to_dict(self) -> dict:
        out = asdict(self)
        out["class_frame"] = list(self.class_frame)
        out["mask_columns"] = list(self.mask_columns)
        out["groups"] = [asdict(x) for x in self.groups]
        return out


def audit_mask_conditional_class_reference(
    score_csv: str | Path,
    reference_json: str | Path,
    *,
    min_test_group_size: int = 20,
    coverage_tolerance: float | None = None,
) -> MaskCalibrationAuditReport:
    if min_test_group_size < 2:
        raise ValueError("min_test_group_size must be >= 2")
    rows = _read_csv(score_csv)
    if "true_label" not in rows[0]:
        raise ValueError("mask calibration audit requires held-out true_label")
    ref = ClassConformalReference.from_dict(json.loads(Path(reference_json).read_text(encoding="utf-8")))
    frame = _class_frame(rows)
    if frame != ref.class_frame:
        raise ValueError(f"score class_frame {frame} does not match reference {ref.class_frame}")
    mask_columns = _mask_columns(rows)
    if mask_columns != ref.mask_columns:
        raise ValueError(f"score modality-mask columns {mask_columns} do not match reference {ref.mask_columns}")
    labels = np.asarray([str(row["true_label"]) for row in rows], dtype=object)
    unknown = sorted(set(labels) - set(frame))
    if unknown:
        raise ValueError(f"held-out labels are outside class_frame: {unknown}")
    label_index = {name: i for i, name in enumerate(frame)}
    y = np.asarray([label_index[str(x)] for x in labels], dtype=int)
    p = _probabilities(rows, frame)
    masks = _masks(rows, mask_columns)
    calibrator = _restore_calibrator(ref)
    reference_counts = {tuple(int(x) for x in group["mask"]): int(group["n_reference"]) for group in ref.groups}
    target = 1.0 - ref.alpha
    audits: list[MaskCoverageAudit] = []
    supported_rows = 0
    unsupported_rows = 0
    for key in sorted({tuple(int(x) for x in row) for row in masks}):
        idx = np.asarray([tuple(int(x) for x in row) == key for row in masks], dtype=bool)
        n = int(np.sum(idx))
        supported = calibrator.has_mask(key)
        if not supported:
            unsupported_rows += n
            audits.append(MaskCoverageAudit(
                key, n, reference_counts.get(key, 0), False, None, target, None,
                None, None, None, None, n >= min_test_group_size, False,
            ))
            continue
        supported_rows += n
        sets = calibrator.prediction_sets(p[idx], masks[idx])
        truth = y[idx]
        coverage = float(np.mean([int(label) in pred for label, pred in zip(truth, sets)]))
        sizes = np.asarray([len(x) for x in sets], dtype=float)
        se = math.sqrt(max(target * (1.0 - target) / n, 0.0))
        tolerance = float(coverage_tolerance) if coverage_tolerance is not None else max(0.02, 2.5 * se)
        audits.append(MaskCoverageAudit(
            key, n, reference_counts.get(key, 0), True, coverage, target,
            max(0.0, target - coverage), float(np.mean(sizes)), float(np.mean(sizes == 1)),
            float(np.mean(sizes == 0)), float(np.mean(np.argmax(p[idx], axis=1) == truth)),
            n >= min_test_group_size, bool(n >= min_test_group_size and coverage >= target - tolerance),
        ))
    total = len(rows)
    observed_supported = all(group.calibration_supported for group in audits)
    supported_groups = [group for group in audits if group.calibration_supported]
    return MaskCalibrationAuditReport(
        "FEC-MASK-CALIBRATION-AUDIT/1.0", ref.contract, ref.alpha, ref.class_frame,
        ref.mask_columns, int(min_test_group_size), tuple(audits),
        float(supported_rows / total), unsupported_rows, observed_supported,
        bool(supported_groups and all(group.pass_test_support and group.pass_coverage for group in supported_groups)),
    )


def write_mask_calibration_audit(
    score_csv: str | Path,
    reference_json: str | Path,
    output_json: str | Path,
    **kwargs,
) -> Path:
    report = audit_mask_conditional_class_reference(score_csv, reference_json, **kwargs)
    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output
