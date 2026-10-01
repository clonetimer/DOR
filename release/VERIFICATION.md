# DOR 1.0.5+timer.1 Production Evidence Closure Phase 2 — Verification

Date: 2026-09-30

## Release decision

This release closes evidence-integrity and prospective-qualification controls. It is intentionally **not** a production-readiness declaration.

```text
ASSISTED_POC              SUPPORTED
ASSISTED_PRODUCTION       BLOCKED
SAFETY_CRITICAL_AUTONOMY  BLOCKED
```

The machine-readable decision is `release/PRODUCTION_EVIDENCE_STATUS.json`.

## Fresh software regression

| Component | Fresh result | Execution mode |
|---|---:|---|
| DOR | **120 / 120 passed** | standalone full suite |
| M2N | **95 / 95 passed** | standalone full suite |
| FEC | **118 / 118 passed** | standalone full suite |
| ISGN | **125 / 125 passed** | all discovered test files, fresh-process partitions |

ISGN's monolithic single pytest process exceeded the constrained execution window. All 125 tests were therefore run in five fresh-process file partitions (41 + 21 + 15 + 27 + 21); no test file was omitted and no failure occurred. This release records that execution mode explicitly rather than misreporting a monolithic pass.

Machine summary: `validation/production_evidence/phase2/regression_summary.json`.

## DOR wheel/install smoke

The DOR 1.0.5 wheel and the three bundled engine wheels were installed together into an isolated target directory and `dor stack-info` reported:

```text
DOR   1.0.5+timer.1
ISGN  2.4.0+timer.1
M2N   2.16.1+timer.1
FEC   2.4.2+timer.1
```

The three new CLI surfaces also completed help/import smoke checks:

- `dor prospective-evaluate`
- `dor deployment-ops-evaluate`
- `dor source-intake-check`

Captured stack output: `validation/production_evidence/phase2/wheel_stack_info.txt`.

## Phase-2 prospective gate

The existing frozen policy `release/PROSPECTIVE_ACCEPTANCE_POLICY_NEXT.json` is bound by canonical SHA-256 in `release/PROSPECTIVE_POLICY_SHA256.txt`.

The current OPSSAT channel-wise remediation passes **16 / 16** frozen metric checks, but the machine result is still:

```text
status = INELIGIBLE
production_gate_credit = false
```

because historical test results were already known before the remediation and this is not a post-freeze new session/asset. This prevents good post-hoc metrics from being re-labelled as prospective production evidence.

Result: `validation/production_evidence/phase2/current_opssat_prospective_result.json`.

## Phase-2 deployment-operations semantics

For the OPSSAT benchmark's selected supported nominal segments:

- false-positive segments: **31**
- selected nominal segment exposure: **28.3575 h**
- calculated proxy: **1.093185 false-positive segments per selected nominal segment-hour**

The machine result is deliberately:

```text
status = PROXY_ONLY
production_gate_credit = false
```

because expert-selected benchmark segments are not continuous mission exposure. The value must not be reported as production false alarms/hour.

Result: `validation/production_evidence/phase2/opssat_selected_segment_ops_result.json`.

## Public-source intake contracts

Release source-intake specs now exist for:

- UCI Hydraulic Systems dataset archive;
- NASA ADAPT `dataset_text.zip`;
- NASA ADAPT `Experiment_Log.xls`;
- NASA ADAPT `AdaptHTMLdiagrams.zip`.

The runtime used for this release could not resolve/download the external official binary artifacts, so no source-intake PASS and no full raw replay are claimed. The specs provide deterministic local filename/size/hash/ZIP-member checks once the official artifacts are supplied.

## Stack/package verification

`scripts/verify_stack.py`: **PASS — 45 / 45 checks**.

It verifies wheel presence, schemas, production-evidence semantics, prospective fail-closed behavior, selected-segment proxy semantics, source-intake specs, inherited real-domain contracts, and release hygiene.

Machine output: `release/STACK_VERIFY.json`.

## Still blocked

The following evidence remains required before `ASSISTED_PRODUCTION` can be considered:

- a post-freeze independent new session / new asset / new mission-period benchmark;
- complete official UCI Hydraulic 2,205-cycle raw replay;
- real distribution shift (sensor/calibration/maintenance/time) qualification;
- operational missing-sensor qualification and recalibration policy;
- continuous-operation false alarms per time and detection-delay measurements;
- sustained CPU/RAM/latency/cold-start measurements and operator workload;
- NASA ADAPT real fault-injection telemetry + experiment log + topology replay for root-cause evaluation;
- system-specific hazard analysis.

Safety-critical autonomous closed-loop control additionally remains blocked by closed-loop safety evidence.
