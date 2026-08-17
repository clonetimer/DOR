# Active Mechanism Discovery — ISGN 2.4.0 Stable

> 集成说明：下文性能数字是 R5 提供方材料中的声明，并非本仓库独立复现。本地采用版还增加了有限值、维度、预算与参数门控；跨入 DOR 时会进一步剥离原始实验/充分性载荷并保持不可执行。

`isgn.active_discovery` is an **opt-in Stable extension** to the existing ISGN mechanism-recovery and closed-loop information planner. It does not replace `recover_*`, completion certification, the Observation Gate, or consensus fusion.

Its task is narrower:

> Given a declared mechanism-structure/parameter portfolio, solver-side forward simulations, an experiment budget and explicit safety eligibility, which admissible experiment should be acquired next to reduce **mechanism-structure** uncertainty per unit cost, and when must the system refuse a unique mechanism interpretation?

## Stable contract

```text
ISGN Information Gap
      ↓
candidate structures + continuous parameter particles   (proposal ≠ evidence)
      ↓
solver-side simulations
      ↓
repeat-aware simulation likelihood bank
      ↓
structure + parameter SMC posterior
      ↓
VoI / cost ranking under experiment admission policy
      ↓
new observation
      ↓
predictive adequacy + evidence stability
      ↓
identifiability decision
      ↓
UNIQUE_SUPPORTED | ALTERNATIVES | UNIDENTIFIABLE |
HYPOTHESIS_CLASS_INADEQUATE
```

`UNIQUE_SUPPORTED` is still only an **Active Discovery advisory state**. It never bypasses the normal ISGN mechanism certification/differential-evidence path.

## Truth quarantine

Active Discovery has no hidden-truth or test-label input. Experiment design, posterior updating, predictive adequacy, evidence stability and identifiability are all pre-truth operations. Held-out truth may be loaded only by an external evaluator after the pre-truth artifact is SHA-frozen.

## Repeat-aware SBI

`ParticleSimulationBank` is keyed by `(particle_id, experiment_id, repeats)`. Missing repeat semantics fail closed with `KeyError`; ISGN does not assume every observable variance scales as `1/repeats`.

## Observational equivalence

A high posterior mass is insufficient for unique identification. `assess_structure_identifiability` evaluates predictive separation under the declared safe experiment contract. Predictively equivalent structures yield `UNIDENTIFIABLE`; multiple credible structures with a remaining discriminating experiment yield `ALTERNATIVES`.

## Evidence-stability veto

`assess_evidence_stability` independently recomputes the evidence posterior without SMC resampling/rejuvenation and performs leave-one-intervention-out replay. A unique advisory identification can be vetoed when the online top is inconsistent with the robust replay or is too sensitive to removal of a single intervention. This guard is truth-free.

## M-open behavior

`admit_proposal_particles` assigns only bounded proposal prior mass. A proposal from an LLM, symbolic generator or library is never evidence. If the admitted hypothesis space is explicitly incomplete, unique identification is vetoed with `HYPOTHESIS_CLASS_INADEQUATE` until unresolved candidates are admitted and replay-scored. Stable therefore supports **calibrated abstention**, not universal open-ended mechanism recovery.

## Experiment safety and execution scope

`ExperimentExecutionPolicy` is fail-closed:

- `ADVISORY_ONLY` / `OFFLINE_REPLAY` candidates may be ranked subject to budget and `safety_eligible`;
- `LIVE` candidates are blocked by default;
- enabling live ranking requires explicit policy authorization and, by default, `safety_evidence_id`;
- an optional approved-live whitelist can further constrain candidates;
- this module **never executes an actuator**.

ISGN 2.4.0 Stable therefore covers advisory/offline experiment design. Autonomous live experimentation is outside the Stable claim.

## Stable evidence boundary

- deterministic NeuronBench: VoI benefit at small experiment budgets;
- stochastic Fox–Lu partial observations: repeat-aware simulation likelihood and structure/parameter particles;
- continuous-parameter NeuronBench: 18-episode identifiability/evidence-stability audit with **0 false unique** and 18/18 calibrated-safe outcomes;
- stochastic M-open: 12/12 safe abstention when hypothesis coverage remained incomplete; this is not a universal-recovery claim;
- whole-spacecraft PHM offline replay: 68 held-out episodes, VoI budget-1 mean standardized held-out feature MSE 0.01605 vs 1.17391 for fixed inactive-first, with 0 false unique.

The spacecraft result is high-fidelity simulation replay, not flight-data or autonomous-actuation validation.
