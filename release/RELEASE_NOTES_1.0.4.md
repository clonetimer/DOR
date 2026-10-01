# DOR 1.0.4+timer.1 Release Notes

## Theme

Production Evidence Closure Phase 1: stop treating software regression or aggregate benchmark scores as production-readiness evidence.

## Added

- `DOR-PRODUCTION-EVIDENCE-GATE/1.0`
- `dor production-evidence-gate`
- OPSSAT per-channel remediation benchmark with strict train/validation/calibration/test role separation
- Hydraulic independent classification/stress replay
- prospective acceptance policy for the next independent dataset/session/asset evaluation
- machine-readable production-evidence input/status artifacts

## Fixed

- clean-source `scripts/verify_all_tests.py` now exposes bundled engine source roots to DOR integration tests
- release packaging no longer ships `.pytest_cache`, `__pycache__`, `.pyc`, or `.pyo`
- `verify_stack.py` now checks production-evidence semantics and the 1.0.4 wheel/schema

## Current evidence

- DOR: 111 passed
- M2N: 95 passed
- FEC: 118 passed
- ISGN: 125 tests collected; monolithic full run not freshly completed in this execution environment
- OPSSAT remediation: F1 0.8250, recall 0.9000, FPR 0.0751 on 523 supported test rows
- Hydraulic independent HGB: macro-F1 0.9706 on the existing derived stable subset

## Deployment status

- `ASSISTED_POC`: SUPPORTED
- `ASSISTED_PRODUCTION`: BLOCKED
- `SAFETY_CRITICAL_AUTONOMY`: BLOCKED

See `docs/PRODUCTION_EVIDENCE_CLOSURE_CN.md` and `release/VERIFICATION.md` for limitations and blocked gates.
