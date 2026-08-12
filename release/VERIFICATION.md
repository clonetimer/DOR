# DOR 0.6 DEV PROTO Verification

This package is a development prototype, not a v0.6 closure release.

Required checks before packaging:

1. `python -m pytest -q tests` -> PASS
2. `python scripts/verify_dor06_rootcause_proto.py` -> PASS
3. Build wheel and verify CLI import/version/root-cause command
4. SHA-256 manifest verification after final ZIP extraction

NASA source-constrained fixture is not a real ADAPT benchmark run.
