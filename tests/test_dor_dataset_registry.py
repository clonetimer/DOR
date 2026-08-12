from __future__ import annotations

import copy
from pathlib import Path

import pytest

from dor.dataset_registry import build_benchmark_plan, catalog_summary, load_catalog, validate_catalog


CATALOG = Path(__file__).parents[1] / "benchmarks" / "public_datasets.json"


def test_bundled_public_catalog_is_valid_and_traceable():
    catalog = load_catalog(CATALOG)
    summary = catalog_summary(catalog)
    assert summary["dataset_count"] >= 16
    assert all(x["adapter_status"] in {"READY", "DERIVED_ONLY", "PLANNED"} for x in summary["datasets"])


def test_hydraulic_plan_exposes_identification_limit_without_blocking_supported_stages():
    plan = build_benchmark_plan(load_catalog(CATALOG), "uci-hydraulic", ["detection", "localization", "classification", "identification"])
    decisions = {x["stage"]: x["decision"] for x in plan["stages"]}
    assert decisions["DETECTION"] == "ELIGIBLE"
    assert decisions["CLASSIFICATION"] == "ELIGIBLE"
    assert decisions["LOCALIZATION"] == "ELIGIBLE"
    assert decisions["IDENTIFICATION"] == "LIMITED_BENCHMARK"
    assert plan["state"] == "LIMITED"


def test_derived_adapter_never_claims_raw_benchmark_eligibility():
    catalog = load_catalog(CATALOG)
    derived = copy.deepcopy(catalog)
    derived["datasets"][0]["adapter"]["status"] = "DERIVED_ONLY"
    plan = build_benchmark_plan(derived, "uci-hydraulic", ["detection", "classification"])
    assert plan["state"] == "LIMITED"
    assert any("derived feature view" in item for item in plan["limitations"])


def test_catalog_fails_closed_on_duplicate_id():
    catalog = load_catalog(CATALOG)
    broken = copy.deepcopy(catalog)
    broken["datasets"].append(copy.deepcopy(broken["datasets"][0]))
    with pytest.raises(ValueError, match="duplicate dataset_id"):
        validate_catalog(broken)


def test_recent_restricted_metadata_record_is_not_benchmark_eligible():
    plan = build_benchmark_plan(
        load_catalog(CATALOG), "spherical-roller-multidefect-2025-restricted", ["detection"]
    )
    assert plan["state"] == "BLOCKED"
    assert "dataset files are not publicly accessible" in plan["blockers"]
