import fec
from fec.extensions import (
    AssetReferenceScope, OperatingContextIdentity,
    validate_fleet_reference_candidate, attest_fleet_reference_robustness,
    fit_class_score_reference, fit_anomaly_score_reference, fit_context_class_score_reference,
)


def test_dor_integration_surface_is_union_of_fleet_and_score_bridges():
    assert fec.__version__.startswith("2.4.2")
    assert AssetReferenceScope is not None
    assert OperatingContextIdentity is not None
    assert callable(validate_fleet_reference_candidate)
    assert callable(attest_fleet_reference_robustness)
    assert callable(fit_class_score_reference)
    assert callable(fit_anomaly_score_reference)
    assert callable(fit_context_class_score_reference)
