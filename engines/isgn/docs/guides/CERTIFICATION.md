# Guide: Certification and refusal

ISGN distinguishes recovery from trust.

- `COMPLETE`: current data/prior/grammar support the operational mechanism in scope.
- `PRIOR_CONFLICT`: a declared-present relation is unsupported by the fitted mechanism.
- `UNIDENTIFIABLE`: candidate sources cannot be independently distinguished by the current data.
- `GRAMMAR_UNCOVERED`: systematic residual remains above independently calibrated normal coverage.
- `BACKEND_DISAGREEMENT`: independent discovery backends do not agree sufficiently; do not select the lowest-error graph automatically.

For real applications, prefer per-target residual-floor calibration from repeat acquisitions. Scalar benchmark floors are for controlled tests only.
