from __future__ import annotations

"""Run every in-repository component test suite with the correct import root."""

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SUITES = (
    ("DOR", ROOT, ROOT / "src"),
    ("M2N", ROOT / "engines" / "m2n", ROOT / "engines" / "m2n" / "src"),
    ("FEC", ROOT / "engines" / "fec", ROOT / "engines" / "fec"),
    ("ISGN", ROOT / "engines" / "isgn", ROOT / "engines" / "isgn" / "src"),
)


def main() -> int:
    for name, cwd, import_root in SUITES:
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(import_root) if not existing else os.pathsep.join((str(import_root), existing))
        print(f"[{name}] {cwd}", flush=True)
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
            cwd=cwd, env=env, check=False,
        )
        if completed.returncode:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
