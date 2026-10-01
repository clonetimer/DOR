# DOR 1.0.5+timer.1 Release Notes

Production Evidence Closure Phase 2 adds executable evidence-integrity gates rather than new diagnostic stages.

## Added

- `dor prospective-evaluate`: exact frozen-policy hash binding, prospective eligibility, subgroup threshold evaluation.
- `dor deployment-ops-evaluate`: continuous-operation vs selected-segment exposure semantics.
- `dor source-intake-check`: local public-artifact hash/size/ZIP-member validation.
- Schemas for the three new contracts.
- UCI Hydraulic and NASA ADAPT source-intake specifications.
- Phase-2 validation artifacts showing current OPSSAT metrics pass the frozen threshold but are not prospectively eligible.

## Safety semantics retained

- ASSISTED_POC: supported.
- ASSISTED_PRODUCTION: blocked.
- SAFETY_CRITICAL_AUTONOMY: blocked.
- Current OPSSAT selected-segment alert-rate calculation is `PROXY_ONLY`, not operational false alarms/hour.
- NASA ADAPT still has no real-run root-cause accuracy claim.
