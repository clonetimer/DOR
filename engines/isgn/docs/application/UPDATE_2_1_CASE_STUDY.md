# Toolkit 2.1 application-driven update case study

## Trigger

Toolkit 2.0 had two bounded shortcomings relevant to its stated task:

1. finite candidate recovery can waste search capacity on physically redundant coordinates;
2. `recover-missing` differentiates observations, so sparse/noisy trajectories can create spurious residual terms even when partial physics is correct.

## Research ideas absorbed

- AI Feynman: simplify a symbolic problem with physics-inspired structure before expensive search.
- Universal Differential Equations: keep known dynamics explicit, learn only the unknown interaction from trajectory mismatch, then recover a symbolic representation.

Neither external algorithm was copied or claimed as implemented.

## Changes

- `isgn.physics_reducer`: dimensional candidate screening and held-out pairwise coordinate reduction.
- `isgn.trajectory_recovery`: fixed-known-equation multiple shooting, neural unexplained-vector-field learning, and sparse held-out distillation.

## Proof of benefit

See `research/evidence/mechanism_recovery_v2/benchmark.json` and `docs/release/RELEASE_VERIFICATION.md`.

This case study is the template for future application-driven Core updates.
