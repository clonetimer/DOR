# Application-stage readiness decision

## Decision

**YES — ISGN Mechanism Recovery Core remains in the application-driven update stage, with bounded scope.**

Toolkit 2.2 is the second application-triggered Core update after the 2.1 precedent. The primary problem remains unchanged: partial mechanism knowledge + operational data -> recover missing mechanism evidence or refuse.

## Why 2.2 does not return the project to open-ended development

1. New equipment is still attempted with a frozen release first. SKAB and the real rotary-actuator case did not trigger Core changes because their failures were attributable to excitation, missing boundary variables, calibration and identifiability.
2. The 2.2 change was opened only after a new real hydraulic case reproduced a within-cycle mode-mixing gap on the released 2.1 wheel.
3. The fix is device-neutral: no ZeMA sensor name, threshold or hydraulic component is encoded in `isgn.hybrid`.
4. Existing solvers remain the source of truth. The hybrid layer only validates observed mode provenance, masks switch neighborhoods and projects stationary dwell data back into ordinary 2.x contracts.
5. Fault/health labels are rejected as mechanism partitions by default.
6. The trigger case has a negative boundary: the no-flow mode is explicitly blocked for degenerate target variation instead of being forced to pass.
7. All 95 current toolkit tests and 23 frozen-v1 research regressions pass after the change (118/118 total collected tests).

## Real trigger evidence

On ten fully nominal/stable ZeMA hydraulic cycles, the simple `PS2 ~ PS1` relation has median held-out test NRMSE about 0.560 when modes are mixed. An observed FS1 partition plus switch-neighborhood exclusion reduces the same relation to about 0.038 in the flow-active dwell mode across multiple thresholds and protection margins. Toolkit 2.2 reproduces the improvement and blocks the near-zero-variation no-flow mode.

## Boundaries

- This release does not perform latent mode discovery.
- A threshold partition is an explicit application rule, not a learned causal state.
- Mode-conditioned relation evidence is not automatically certified; `discover_regime_algebraic_relations` returns `certification_allowed=False`.
- Dynamic switched-system recovery is supported through transition-free `PublicDataset` dwell-window projection; no new hybrid ODE solver is claimed.
- MetroPT-3 is a preferred next operational validation case, but it is not part of the 2.2 execution evidence.
- Official PySINDy and external SymMatika remain optional as before.

## Practical meaning

A new equipment case should first use the released Toolkit unchanged. Core changes remain justified only by a traceable, reusable gap with old-vs-new evidence and regression protection.
