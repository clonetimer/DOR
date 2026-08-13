from dor.modality_calibration import assess_modality_mask_calibration


def test_exhaustive_mask_budget_exposes_combinatorial_sample_need():
    plan = assess_modality_mask_calibration(("imu", "power", "thermal"), calibration_rows=100, min_group_size=20)
    assert plan.possible_nonempty_masks == 7
    assert plan.exhaustive_minimum_rows == 140
    assert plan.status == "INSUFFICIENT_FOR_EXHAUSTIVE_MASKS"


def test_observed_masks_are_qualified_exactly_and_sparse_mask_blocks():
    plan = assess_modality_mask_calibration(
        ("imu", "power"), calibration_rows=60, min_group_size=20,
        observed_group_counts={"11": 35, "10": 19},
    )
    assert plan.status == "INSUFFICIENT_OBSERVED_MASKS"
    assert plan.inadequate_observed_groups == ("10",)
    assert "silently pool" not in plan.recommendation


def test_total_count_never_substitutes_for_group_counts():
    plan = assess_modality_mask_calibration(("imu", "power"), calibration_rows=80, min_group_size=20)
    assert plan.status == "TOTAL_BUDGET_PLAUSIBLE_DISTRIBUTION_UNKNOWN"
    assert "actual per-mask counts" in plan.recommendation
