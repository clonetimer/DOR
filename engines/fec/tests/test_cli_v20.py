from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data"
CONFIG = ROOT / "configs" / "demo_machine.json"


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "fec.cli", *map(str, args)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_minimal_cli_runs(tmp_path: Path):
    out = tmp_path / "minimal.csv"
    r = _run(
        "run", "--config", CONFIG,
        "--normal-leaf", DATA / "normal_leaf.csv",
        "--normal-root", DATA / "normal_root.csv",
        "--input", DATA / "inference.csv",
        "--output", out,
    )
    assert r.returncode == 0, r.stderr
    assert out.exists()
    assert len(out.read_text(encoding="utf-8").splitlines()) == 41


def test_production_cli_restart_is_exact(tmp_path: Path):
    state = tmp_path / "state.json"
    out1 = tmp_path / "first.csv"
    out2 = tmp_path / "restart.csv"
    first = _run(
        "production", "--config", CONFIG,
        "--state", state,
        "--key", "default",
        "--input", DATA / "inference.csv",
        "--output", out1,
        "--trusted-claims", "demo_machine:healthy",
        "--commission-leaf", DATA / "normal_leaf.csv",
        "--commission-root", DATA / "normal_root.csv",
        "--commission-sentinel", DATA / "trusted_sentinel.csv",
        "--bundle-id", "cli-demo-v1",
        "--leaf-data-id", "leaf-run",
        "--root-data-id", "root-run",
        "--sentinel-data-id", "sentinel-run",
    )
    assert first.returncode == 0, first.stderr
    assert state.exists()
    second = _run(
        "production", "--config", CONFIG,
        "--state", state,
        "--key", "default",
        "--input", DATA / "inference.csv",
        "--output", out2,
    )
    assert second.returncode == 0, second.stderr
    assert out1.read_bytes() == out2.read_bytes()
    verified = _run("verify-state", "--state", state)
    assert verified.returncode == 0, verified.stderr
    assert '"state": "VALID"' in verified.stdout


def test_production_cli_blocks_provider_fingerprint_mismatch_after_restart(tmp_path: Path):
    import json
    state = tmp_path / "state.json"
    out1 = tmp_path / "first.csv"
    first = _run(
        "production", "--config", CONFIG,
        "--state", state,
        "--key", "default",
        "--input", DATA / "inference.csv",
        "--output", out1,
        "--trusted-claims", "demo_machine:healthy",
        "--commission-leaf", DATA / "normal_leaf.csv",
        "--commission-root", DATA / "normal_root.csv",
        "--commission-sentinel", DATA / "trusted_sentinel.csv",
        "--bundle-id", "cli-demo-v21",
        "--leaf-data-id", "leaf-run",
        "--root-data-id", "root-run",
        "--sentinel-data-id", "sentinel-run",
    )
    assert first.returncode == 0, first.stderr

    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    cfg["reference_contract"]["provider_fingerprints"] = ["sha256:" + "0" * 64]
    changed = tmp_path / "changed_config.json"
    changed.write_text(json.dumps(cfg), encoding="utf-8")
    blocked = _run(
        "production", "--config", changed,
        "--state", state,
        "--key", "default",
        "--input", DATA / "inference.csv",
        "--output", tmp_path / "blocked.csv",
    )
    assert blocked.returncode == 3
    assert '"state": "REFERENCE_UNSUPPORTED"' in blocked.stdout
    assert "provider_fingerprints" in blocked.stdout
