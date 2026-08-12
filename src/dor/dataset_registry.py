from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Iterable


CATALOG_CONTRACT = "DOR-PUBLIC-DIAGNOSTIC-DATASET-CATALOG/1.0"
STAGES = ("DETECTION", "LOCALIZATION", "CLASSIFICATION", "IDENTIFICATION")
EVIDENCE_STATES = {"AVAILABLE", "PARTIAL", "ABSENT"}
IMPLEMENTATION_STATES = {"READY", "PARTIAL", "BLOCKED"}


def default_catalog_path() -> Path:
    repository = Path(__file__).resolve().parents[2] / "benchmarks" / "public_datasets.json"
    installed = Path(sys.prefix) / "share" / "dor" / "public_datasets.json"
    return repository if repository.is_file() else installed


def load_catalog(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_catalog(value)
    return value


def validate_catalog(value: dict[str, Any]) -> None:
    if value.get("contract") != CATALOG_CONTRACT:
        raise ValueError(f"catalog contract must be {CATALOG_CONTRACT}")
    datasets = value.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("catalog.datasets must be a non-empty list")
    ids: set[str] = set()
    for index, item in enumerate(datasets):
        prefix = f"datasets[{index}]"
        dataset_id = item.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError(f"{prefix}.dataset_id must be a non-empty string")
        if dataset_id in ids:
            raise ValueError(f"duplicate dataset_id: {dataset_id}")
        ids.add(dataset_id)
        if not str(item.get("official_landing_page", "")).startswith("https://"):
            raise ValueError(f"{prefix}.official_landing_page must be HTTPS")
        if not item.get("primary_references"):
            raise ValueError(f"{prefix}.primary_references must not be empty")
        stage_evidence = item.get("stage_evidence", {})
        dor_support = item.get("dor_support", {})
        for stage in STAGES:
            evidence = stage_evidence.get(stage)
            implementation = dor_support.get(stage)
            if evidence not in EVIDENCE_STATES:
                raise ValueError(f"{prefix}.stage_evidence.{stage} is invalid")
            if implementation not in IMPLEMENTATION_STATES:
                raise ValueError(f"{prefix}.dor_support.{stage} is invalid")
        split = item.get("validation", {})
        if not split.get("split_unit") or not split.get("required_guard"):
            raise ValueError(f"{prefix}.validation must declare split_unit and required_guard")
        adapter = item.get("adapter", {})
        if adapter.get("status") not in {"READY", "DERIVED_ONLY", "PLANNED"}:
            raise ValueError(f"{prefix}.adapter.status is invalid")


def _normalize_stages(stages: Iterable[str]) -> list[str]:
    result: list[str] = []
    for raw in stages:
        stage = str(raw).upper()
        if stage not in STAGES:
            raise ValueError(f"unsupported diagnostic stage: {raw}")
        if stage not in result:
            result.append(stage)
    if not result:
        raise ValueError("at least one stage is required")
    return result


def build_benchmark_plan(
    catalog: dict[str, Any], dataset_id: str, stages: Iterable[str]
) -> dict[str, Any]:
    validate_catalog(catalog)
    requested = _normalize_stages(stages)
    matches = [item for item in catalog["datasets"] if item["dataset_id"] == dataset_id]
    if not matches:
        raise ValueError(f"unknown public dataset: {dataset_id}")
    item = matches[0]
    stage_plan = []
    blockers: list[str] = []
    limitations: list[str] = []
    for stage in requested:
        evidence = item["stage_evidence"][stage]
        implementation = item["dor_support"][stage]
        if evidence == "ABSENT":
            decision = "NOT_EVALUABLE"
            reason = "the public dataset does not provide independent evidence for this stage"
        elif implementation == "BLOCKED":
            decision = "DOR_UPDATE_REQUIRED"
            reason = "dataset evidence exists, but the current DOR implementation cannot consume it safely"
            blockers.append(f"{stage}: implementation gap")
        elif evidence == "PARTIAL" or implementation == "PARTIAL":
            decision = "LIMITED_BENCHMARK"
            reason = "run only with the catalogued scope limitation; do not generalize beyond it"
        else:
            decision = "ELIGIBLE"
            reason = "dataset evidence and current DOR capability are aligned"
        stage_plan.append({
            "stage": stage,
            "dataset_evidence": evidence,
            "dor_support": implementation,
            "decision": decision,
            "reason": reason,
        })
    if item["adapter"]["status"] == "PLANNED":
        blockers.append("dataset adapter is not implemented")
    elif item["adapter"]["status"] == "DERIVED_ONLY":
        limitations.append("only a derived feature view is integrated; raw-signal claims are prohibited")
    state = "BLOCKED" if blockers else (
        "LIMITED" if limitations or any(x["decision"] != "ELIGIBLE" for x in stage_plan) else "ELIGIBLE"
    )
    return {
        "contract": "DOR-PUBLIC-BENCHMARK-PLAN/1.0",
        "state": state,
        "dataset_id": dataset_id,
        "official_landing_page": item["official_landing_page"],
        "license": item["access"]["license"],
        "adapter": item["adapter"],
        "validation": item["validation"],
        "stages": stage_plan,
        "blockers": blockers,
        "limitations": limitations,
        "prohibited_claims": item["prohibited_claims"],
        "catalog_verified_on": catalog["last_verified_on"],
    }


def catalog_summary(catalog: dict[str, Any]) -> dict[str, Any]:
    validate_catalog(catalog)
    return {
        "contract": catalog["contract"],
        "last_verified_on": catalog["last_verified_on"],
        "dataset_count": len(catalog["datasets"]),
        "datasets": [
            {
                "dataset_id": item["dataset_id"],
                "name": item["name"],
                "device_domain": item["device_domain"],
                "adapter_status": item["adapter"]["status"],
            }
            for item in catalog["datasets"]
        ],
    }
