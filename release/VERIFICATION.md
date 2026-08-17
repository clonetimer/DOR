# DOR 1.0.3+timer.1 R5-derived Integration Verification

2026-08-18 fresh checks:

- DOR 107 passed, 1 optional skipped; strict JSON Schema test independently passed in the system environment;
- ISGN 125 passed; M2N 95 passed; FEC 118 passed;
- total 445 passed, 1 optional skipped;
- four-wheel isolated import/API/CLI smoke passed;
- `scripts/verify_stack.py` passed, including the Active Discovery advisory schema and release hygiene;
- local editable metadata matches DOR 1.0.3+timer.1 and ISGN 2.4.0+timer.1.

Active Discovery remains planning-only. It does not authorize diagnosis, root-cause claims, mechanism certification, experiment execution, or live actuation.

## Historical v0.6 lifecycle note

This package is a development prototype, not a v0.6 closure release.

Required checks before packaging:

1. `python -m pytest -q tests` -> PASS
2. `python scripts/verify_dor06_rootcause_proto.py` -> PASS
3. Build wheel and verify CLI import/version/root-cause command
4. SHA-256 manifest verification after final ZIP extraction

NASA source-constrained fixture is not a real ADAPT benchmark run.
