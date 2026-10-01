import hashlib
import json
import zipfile

from dor.source_intake import validate_public_source_intake


def test_zip_source_intake_binds_hash_and_required_members(tmp_path):
    path = tmp_path / "official.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("profile.txt", "1\t2\n")
        zf.writestr("PS1.txt", "3\t4\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    spec = {
        "contract": "DOR-PUBLIC-SOURCE-INTAKE/1.0",
        "dataset_id": "fixture",
        "official_url": "https://example.test/official.zip",
        "expected_filename": "official.zip",
        "expected_hashes": {"sha256": digest},
        "zip_requirements": {"required_members": ["profile.txt", "PS1.txt"]},
    }
    result = validate_public_source_intake(spec, path)
    assert result["status"] == "PASS"
    assert result["hashes"]["sha256"] == digest


def test_source_intake_fails_closed_on_missing_member(tmp_path):
    path = tmp_path / "official.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("profile.txt", "x")
    spec = {
        "contract": "DOR-PUBLIC-SOURCE-INTAKE/1.0",
        "dataset_id": "fixture",
        "official_url": "https://example.test/official.zip",
        "expected_filename": "official.zip",
        "zip_requirements": {"required_members": ["profile.txt", "PS1.txt"]},
    }
    result = validate_public_source_intake(spec, path)
    assert result["status"] == "BLOCKED"
    assert any("PS1.txt" in x for x in result["errors"])
