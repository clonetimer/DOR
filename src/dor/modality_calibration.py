from __future__ import annotations

"""DOR control-plane feasibility checks for modality-mask calibration."""

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ModalityMaskCalibrationPlan:
    contract: str
    status: str
    modality_names: tuple[str, ...]
    possible_nonempty_masks: int
    min_group_size: int
    calibration_rows: int
    exhaustive_minimum_rows: int
    observed_group_counts: tuple[tuple[str, int], ...]
    inadequate_observed_groups: tuple[str, ...]
    recommendation: str
    coverage_scope: str = "EXACT_MODALITY_MASK_GROUP_CONDITIONAL"
    claim_boundary: str = "A sample-count plan is not a conformal validity theorem; exchangeability must still hold within each admitted mask."

    def to_dict(self) -> dict:
        out = asdict(self)
        out["modality_names"] = list(self.modality_names)
        out["observed_group_counts"] = {key: value for key, value in self.observed_group_counts}
        out["inadequate_observed_groups"] = list(self.inadequate_observed_groups)
        return out


def assess_modality_mask_calibration(
    modality_names: Sequence[str],
    *,
    calibration_rows: int,
    min_group_size: int = 20,
    observed_group_counts: Mapping[str, int] | None = None,
) -> ModalityMaskCalibrationPlan:
    names = tuple(str(x).strip() for x in modality_names)
    if not names or any(not x for x in names) or len(set(names)) != len(names):
        raise ValueError("modality_names must be unique and non-empty")
    if calibration_rows < 0:
        raise ValueError("calibration_rows must be non-negative")
    if min_group_size < 2:
        raise ValueError("min_group_size must be >= 2")
    possible = (1 << len(names)) - 1
    exhaustive = possible * min_group_size
    counts = tuple(sorted((str(key), int(value)) for key, value in dict(observed_group_counts or {}).items()))
    if any(value < 0 for _, value in counts):
        raise ValueError("observed group counts must be non-negative")
    if counts and sum(value for _, value in counts) > calibration_rows:
        raise ValueError("observed group counts exceed calibration_rows")
    inadequate = tuple(key for key, value in counts if value < min_group_size)

    if counts:
        if inadequate:
            status = "INSUFFICIENT_OBSERVED_MASKS"
            recommendation = "collect labeled calibration data for each inadequate observed mask; unseen masks must abstain"
        else:
            status = "OBSERVED_MASKS_READY"
            recommendation = "calibrate only the declared observed masks and retain unseen-mask abstention"
    elif calibration_rows < exhaustive:
        status = "INSUFFICIENT_FOR_EXHAUSTIVE_MASKS"
        recommendation = "declare deployment-relevant masks, collect per-mask labels, or reduce modality grouping; do not silently pool masks"
    else:
        status = "TOTAL_BUDGET_PLAUSIBLE_DISTRIBUTION_UNKNOWN"
        recommendation = "record actual per-mask counts before calibration; total sample count alone cannot qualify any mask"
    if len(names) >= 5:
        recommendation += "; mask count is combinatorial, and any declared clustering weakens the claim to cluster-conditional coverage"
    return ModalityMaskCalibrationPlan(
        "DOR-MODALITY-MASK-CALIBRATION-PLAN/1.0", status, names, possible,
        int(min_group_size), int(calibration_rows), exhaustive, counts, inadequate,
        recommendation,
    )
