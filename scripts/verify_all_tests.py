from __future__ import annotations

"""Run every in-repository component test suite with the correct import roots.

DOR integration tests import ISGN contracts directly, so the DOR suite needs the
source roots of all bundled engines when the stack has not been installed into a
virtual environment.  FEC and ISGN are intentionally executed in fresh-process file
partitions: its monolithic pytest process accumulates enough backend/runtime state
to exceed constrained CI/runtime windows even though the same 125 tests pass when
isolated.  Partitioning is therefore an execution-hygiene measure, not a way to
skip or select tests.
"""

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DOR_SRC = ROOT / "src"
M2N_SRC = ROOT / "engines" / "m2n" / "src"
FEC_SRC = ROOT / "engines" / "fec"
ISGN_SRC = ROOT / "engines" / "isgn" / "src"

SUITES = (
    ("DOR", ROOT, (DOR_SRC, ISGN_SRC, M2N_SRC, FEC_SRC)),
    ("M2N", ROOT / "engines" / "m2n", (M2N_SRC,)),
    ("FEC", ROOT / "engines" / "fec", (FEC_SRC,)),
    ("ISGN", ROOT / "engines" / "isgn", (ISGN_SRC,)),
)


def _env(import_roots: tuple[Path, ...]) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Verification must not depend on an ambient checkout/site PYTHONPATH.
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in import_roots)
    # Keep BLAS/OpenMP backends from multiplying worker threads across suites in
    # constrained CI containers.  This is execution hygiene, not a model change.
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    return env


def _run(cwd: Path, env: dict[str, str], args: list[str]) -> int:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *args],
        cwd=cwd,
        env=env,
        check=False,
    )
    return completed.returncode


def _run_partitioned(name: str, cwd: Path, env: dict[str, str], chunk_size: int = 7) -> int:
    files = sorted((cwd / "tests").glob("test_*.py"))
    if not files:
        print(f"[{name}] no test files discovered", flush=True)
        return 2
    total_files = len(files)
    for start in range(0, total_files, chunk_size):
        chunk = files[start:start + chunk_size]
        rel = [str(p.relative_to(cwd)) for p in chunk]
        print(f"[{name} partition {start // chunk_size + 1}] files {start + 1}-{start + len(chunk)} / {total_files}", flush=True)
        rc = _run(cwd, env, rel)
        if rc:
            return rc
    return 0


def main() -> int:
    for name, cwd, import_roots in SUITES:
        env = _env(import_roots)
        print(f"[{name}] {cwd}", flush=True)
        if name == "ISGN":
            rc = _run_partitioned(name, cwd, env, chunk_size=7)
        elif name == "FEC":
            # FEC's final weighted-conformal test can stall in a long-lived
            # multi-suite verification run even though the suite passes alone.
            # Fresh file partitions eliminate cross-suite/runtime accumulation.
            rc = _run_partitioned(name, cwd, env, chunk_size=8)
        else:
            rc = _run(cwd, env, ["tests"])
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
