from __future__ import annotations

from math import ceil

import numpy as np
import pytest

from fec.extensions.class_evidence import ClassEvidenceBatch
from fec.extensions.mask_conditional_class import (
    MaskCalibrationUnavailable,
    MaskConditionalClassConformal,
    fuse_class_evidence_by_mask,
)


def test_missing_modality_is_vacuous_and_stale_values_are_ignored():
    m1 = ClassEvidenceBatch(
        evidence=np.array([[8.0, 1.0], [1.0, 8.0]]),
        class_frame=("normal", "fault"), source="pressure", provenance="p",
    )
    # Deliberately huge contradictory stale evidence.  The first sample marks
    # modality 2 missing, so this row must not influence the result.
    m2 = ClassEvidenceBatch(
        evidence=np.array([[0.0, 1000.0], [5.0, 0.0]]),
        class_frame=("normal", "fault"), source="vibration", provenance="v",
    )
    fused = fuse_class_evidence_by_mask((m1, m2), np.array([[1, 0], [1, 1]]))
    expected_first = m1.opinions()[0].probability
    np.testing.assert_allclose(fused.probabilities[0], expected_first)
    assert fused.per_modality_vacuity[0, 1] == 1.0
    with pytest.raises(ValueError, match="at least one present modality"):
        fuse_class_evidence_by_mask((m1, m2), np.array([[0, 0], [1, 1]]))


def _binary_probs(true_p):
    true_p = np.asarray(true_p, dtype=float)
    return np.column_stack([true_p, 1.0 - true_p])


def _finite_quantile(scores, alpha):
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    rank = min(n, ceil((n + 1) * (1.0 - alpha)))
    return scores[rank - 1]


def test_mask_conditioning_removes_structural_full_partial_coverage_gap_in_controlled_harness():
    alpha = 0.10
    ncal = 300
    ntest = 1000
    full_cal = np.linspace(0.80, 0.99, ncal)
    partial_cal = np.linspace(0.35, 0.65, ncal)
    probs_cal = np.vstack([_binary_probs(full_cal), _binary_probs(partial_cal)])
    labels_cal = np.zeros(2 * ncal, dtype=int)
    masks_cal = np.vstack([
        np.tile([1, 1], (ncal, 1)),
        np.tile([1, 0], (ncal, 1)),
    ])
    cal = MaskConditionalClassConformal(alpha=alpha, min_group_size=50).fit(
        probs_cal, labels_cal, masks_cal
    )

    full_test = np.linspace(0.80, 0.99, ntest)
    partial_test = np.linspace(0.35, 0.65, ntest)
    probs_test = np.vstack([_binary_probs(full_test), _binary_probs(partial_test)])
    labels_test = np.zeros(2 * ntest, dtype=int)
    masks_test = np.vstack([
        np.tile([1, 1], (ntest, 1)),
        np.tile([1, 0], (ntest, 1)),
    ])
    sets = cal.prediction_sets(probs_test, masks_test)
    full_cov = np.mean([0 in s for s in sets[:ntest]])
    partial_cov = np.mean([0 in s for s in sets[ntest:]])
    assert 0.88 <= full_cov <= 0.92
    assert 0.88 <= partial_cov <= 0.92
    assert abs(full_cov - partial_cov) <= 0.02

    # A single marginal threshold over the same calibration scores systematically
    # over-covers the easy full-modality stratum and under-covers the partial one.
    marginal_q = _finite_quantile(1.0 - probs_cal[:, 0], alpha)
    full_marginal = np.mean((1.0 - probs_test[:ntest, 0]) <= marginal_q)
    partial_marginal = np.mean((1.0 - probs_test[ntest:, 0]) <= marginal_q)
    assert abs(full_marginal - partial_marginal) > 0.10


def test_sparse_or_unseen_mask_is_unsupported_not_silently_borrowed():
    probs = np.tile([[0.8, 0.2]], (25, 1))
    labels = np.zeros(25, dtype=int)
    masks = np.tile([1, 0], (25, 1))
    cal = MaskConditionalClassConformal(alpha=0.1, min_group_size=20).fit(probs, labels, masks)
    with pytest.raises(MaskCalibrationUnavailable, match="will not silently borrow"):
        cal.prediction_sets(np.array([[0.7, 0.3]]), np.array([[0, 1]]))
