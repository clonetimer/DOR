# Changelog

## 2.2.0 — Observed-regime / hybrid mechanism routing

- Added `RegimePartition` for sample/interval-level operating modes and switch-neighborhood masks.
- Added deterministic threshold partitioning from observed channels with optional hysteresis.
- Added algebraic regime projection and fixed-window dynamic regime projection that reuse existing 2.x solvers.
- Added mode-conditioned algebraic recovery with explicit `INSUFFICIENT_MODE_SUPPORT` and `DEGENERATE_TARGET_VARIATION` outcomes.
- Rejects fault/health-label partitions for mechanism discovery by default.
- Triggered by a frozen-2.1 real ZeMA hydraulic application where a simple cross-pressure relation was corrupted by within-cycle mode mixing; post-fix mode-conditioned held-out NRMSE is about 0.038 versus about 0.528 for the non-transition global baseline.
- Added hybrid positive/refusal regressions and preserved all current/frozen regressions.
- Does not claim latent-mode learning or a new universal hybrid ODE solver.

## 2.1.0 — Application-driven mechanism-recovery update

- Added physics-aware search reduction: dimensional monomial screening and held-out pairwise coordinate simplification.
- Added trajectory-level partial-physics recovery that learns unexplained dynamics from whole trajectories and distills the residual into finite mechanism terms.
- Added `reduce-physics` and `recover-missing-trajectory` CLI paths.
- Added AlgebraicDataset NPZ persistence.
- Added Mechanism Recovery Benchmark v2 with 12 release cases, including sparse/noisy dynamic recovery and real UCI hydraulic coordinate simplification.
- Updated application-readiness evidence and update policy.
- Preserved the 2.0 core definition and all earlier refusal semantics.

## 2.0.0 — Mechanism Recovery application baseline

- Re-centered ISGN on partial-physics mechanism recovery: identify, complete, refuse.
- Added `recover_missing_terms` API and `recover-missing` CLI for fixed-known-physics residual recovery.
- Added Mechanism Recovery Benchmark with eight positive/refusal/real-data cases.
- Reorganized user documentation around mechanism recovery rather than Atlas/Closed Loop.
- Kept Atlas, architecture compiler, PHM and closed-loop planning as auxiliary modules for compatibility and downstream use.
- Declared the transition to an application-driven update policy with explicit bounded-scope readiness criteria.
