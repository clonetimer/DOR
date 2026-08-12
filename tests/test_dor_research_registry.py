from __future__ import annotations

import copy
from pathlib import Path

import pytest

from dor.research_registry import load_method_catalog, method_catalog_summary, validate_method_catalog


CATALOG = Path(__file__).parents[1] / "benchmarks" / "recent_methods_2024_2026.json"


def test_recent_method_catalog_is_closed_window_and_auditable():
    catalog = load_method_catalog(CATALOG)
    summary = method_catalog_summary(catalog)
    assert summary["search_window"] == {"from": "2024-08-13", "to": "2026-08-13"}
    assert summary["method_count"] >= 12
    assert summary["status_counts"]["PEER_REVIEWED"] >= 4
    assert all(item["mechanism_delta"] and item["risks"] for item in catalog["methods"])


def test_method_catalog_rejects_out_of_window_claims():
    catalog = load_method_catalog(CATALOG)
    broken = copy.deepcopy(catalog)
    broken["methods"][0]["published_on"] = "2024-01-01"
    with pytest.raises(ValueError, match="outside"):
        validate_method_catalog(broken)
