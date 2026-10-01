from __future__ import annotations

"""Fail-closed intake validation for externally acquired benchmark artifacts."""

from hashlib import md5, sha256
from pathlib import Path
from typing import Any
import json
import zipfile

CONTRACT = "DOR-PUBLIC-SOURCE-INTAKE/1.0"
RESULT_CONTRACT = "DOR-PUBLIC-SOURCE-INTAKE-RESULT/1.0"


def _digest(path: Path, algorithm: str) -> str:
    h = sha256() if algorithm == "sha256" else md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_public_source_intake(spec: dict[str, Any], artifact_path: str | Path) -> dict[str, Any]:
    errors: list[str] = []
    if spec.get("contract") != CONTRACT:
        errors.append(f"contract must be {CONTRACT}")
    path = Path(artifact_path)
    if not path.is_file():
        errors.append("artifact is missing")
        return {"contract": RESULT_CONTRACT, "status": "BLOCKED", "artifact": str(path), "errors": errors}

    expected_name = spec.get("expected_filename")
    if expected_name and path.name != expected_name:
        errors.append(f"filename mismatch: expected {expected_name!r}, got {path.name!r}")
    expected_size = spec.get("expected_size_bytes")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        errors.append(f"size mismatch: expected {int(expected_size)}, got {path.stat().st_size}")

    hashes = {"sha256": _digest(path, "sha256"), "md5": _digest(path, "md5")}
    expected_hashes = spec.get("expected_hashes") or {}
    for alg in ("sha256", "md5"):
        expected = expected_hashes.get(alg)
        if expected and hashes[alg].lower() != str(expected).lower():
            errors.append(f"{alg} mismatch")

    archive = spec.get("zip_requirements") or {}
    member_results: list[dict[str, Any]] = []
    if archive:
        if not zipfile.is_zipfile(path):
            errors.append("artifact must be a ZIP archive")
        else:
            with zipfile.ZipFile(path) as zf:
                basenames: dict[str, int] = {}
                for name in zf.namelist():
                    base = name.rsplit("/", 1)[-1]
                    if base:
                        basenames[base.lower()] = basenames.get(base.lower(), 0) + 1
                for member in archive.get("required_members", []):
                    count = basenames.get(str(member).lower(), 0)
                    ok = count == 1
                    member_results.append({"member": member, "count": count, "pass": ok})
                    if not ok:
                        errors.append(f"required ZIP member {member!r} count must be 1, got {count}")

    official_url = str(spec.get("official_url", ""))
    if not official_url.startswith("https://"):
        errors.append("official_url must be https://")

    return {
        "contract": RESULT_CONTRACT,
        "status": "PASS" if not errors else "BLOCKED",
        "artifact": str(path),
        "dataset_id": spec.get("dataset_id"),
        "official_url": official_url,
        "size_bytes": path.stat().st_size,
        "hashes": hashes,
        "zip_member_checks": member_results,
        "errors": errors,
        "provenance": {
            "local_bytes_hash_bound": True,
            "download_channel_attestation": spec.get("retrieval_attestation") or "UNDECLARED",
            "note": "local structure/hash validation does not independently prove that bytes were downloaded from the declared URL unless retrieval provenance is externally recorded",
        },
    }


def validate_public_source_intake_from_files(spec_path: str | Path, artifact_path: str | Path) -> dict[str, Any]:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    return validate_public_source_intake(spec, artifact_path)
