from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


METHOD_CONTRACT = "DOR-RECENT-INDUSTRIAL-DIAGNOSTIC-METHOD-CATALOG/1.0"
EVIDENCE_LEVELS = {"E0", "E1", "E2", "E3"}
STATUSES = {"PREPRINT", "WORKSHOP_PAPER", "PEER_REVIEWED", "OFFICIAL_CHALLENGE_REPORT"}


def default_method_catalog_path() -> Path:
    repository = Path(__file__).resolve().parents[2] / "benchmarks" / "recent_methods_2024_2026.json"
    installed = Path(sys.prefix) / "share" / "dor" / "recent_methods_2024_2026.json"
    return repository if repository.is_file() else installed


def load_method_catalog(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_method_catalog(value)
    return value


def validate_method_catalog(value: dict[str, Any]) -> None:
    if value.get("contract") != METHOD_CONTRACT:
        raise ValueError(f"method catalog contract must be {METHOD_CONTRACT}")
    window = value.get("search_window") or {}
    if not window.get("from") or not window.get("to"):
        raise ValueError("method catalog requires a closed search window")
    methods = value.get("methods")
    if not isinstance(methods, list) or not methods:
        raise ValueError("method catalog methods must be non-empty")
    ids: set[str] = set()
    for index, item in enumerate(methods):
        prefix = f"methods[{index}]"
        method_id = item.get("method_id")
        if not isinstance(method_id, str) or not method_id or method_id in ids:
            raise ValueError(f"{prefix}.method_id must be non-empty and unique")
        ids.add(method_id)
        if not str(item.get("primary_url", "")).startswith("https://"):
            raise ValueError(f"{prefix}.primary_url must be HTTPS")
        if item.get("status") not in STATUSES:
            raise ValueError(f"{prefix}.status is invalid")
        if item.get("evidence_level") not in EVIDENCE_LEVELS:
            raise ValueError(f"{prefix}.evidence_level is invalid")
        if not item.get("mechanism_delta") or not item.get("dor_decision") or not item.get("risks"):
            raise ValueError(f"{prefix} lacks mechanism, decision or risk fields")
        date = str(item.get("published_on", ""))
        if not (str(window["from"]) <= date <= str(window["to"])):
            raise ValueError(f"{prefix}.published_on is outside the catalog search window")


def method_catalog_summary(value: dict[str, Any]) -> dict[str, Any]:
    validate_method_catalog(value)
    status_counts: dict[str, int] = {}
    decision_counts: dict[str, int] = {}
    for item in value["methods"]:
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1
        decision_counts[item["dor_decision"]] = decision_counts.get(item["dor_decision"], 0) + 1
    return {
        "contract": value["contract"], "search_window": value["search_window"],
        "last_verified_on": value["last_verified_on"], "method_count": len(value["methods"]),
        "status_counts": status_counts, "decision_counts": decision_counts,
    }
