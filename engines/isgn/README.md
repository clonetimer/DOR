# ISGN Toolkit 2.2

**ISGN** is a practical toolkit for one core task:

> Given partial industrial mechanism knowledge and operational data, recover the missing mechanism terms, graph edges, constitutive relations, and parameters; refuse a conclusion when the evidence is insufficient or inconsistent.

The core workflow remains intentionally small:

```text
partial mechanism + data + finite mechanism grammar
                    ↓
       physics-aware search reduction (optional)
                    ↓
        isolate unexplained mechanism
          ↙                    ↘
 derivative residual       trajectory residual
          ↓                    ↓
        mechanism recovery / symbolic distillation
                    ↓
       candidate operational mechanism
                    ↓
               certification
          ↙          ↓          ↘
      COMPLETE   UNIDENTIFIABLE  PRIOR/GRAMMAR conflict
```

Toolkit 2.2 remains in the **application-driven update** stage. Toolkit 2.1 added physics-aware reduction and trajectory-residual recovery; Toolkit 2.2 adds a third bounded capability exposed by a new real hydraulic application: observed intra-cycle operating regimes.

The current reusable recovery paths are:

- physics-inspired dimensional and symmetry reduction before mechanism search;
- trajectory-level recovery of unknown dynamics when numerical differentiation is unreliable;
- observed-regime projection for switched/hybrid operating modes, with transition exclusion and refusal of fault-label partitions by default.

SINDy-like sparse regression, optional external PySINDy/SymMatika, universal approximators, symbolic distillation, and finite industrial mechanism families remain tools under the ISGN workflow rather than separate ISGN claims.

## Start here

1. Read `docs/getting_started/QUICKSTART.md`.
2. If the candidate symbolic problem is large, use `reduce-physics` first.
3. For a known model with missing terms and usable derivatives, use `recover-missing`.
4. For sparse/noisy trajectories where differentiation is unreliable, use `recover-missing-trajectory`.
5. For unknown graph edges, use `recover` / `complete`.
6. Use `certify` before treating a recovered model as trusted.
7. Run `benchmarks/mechanism_recovery_v2/run.py` to reproduce the release benchmark.

## Core vs auxiliary

Core: partial-physics mechanism recovery, physics-aware search reduction, trajectory residual isolation, partial-graph completion, constitutive relation discovery, backend fusion, completion certification.

Auxiliary: Observation Gate, PHM evidence, Mechanism Atlas, architecture compiler, evidence router, and closed-loop information planning. They are retained because they are useful in applications, but they do not define the research core.

## Release status

Toolkit 2.0 remains the first application baseline. Toolkit 2.1 and 2.2 demonstrate the intended **application-driven update process**: identify a reproducible general gap, add a device-neutral capability, benchmark old vs new behavior, preserve refusal semantics, and re-run all frozen regressions. See `docs/application/UPDATE_POLICY.md`, `docs/application/UPDATE_2_1_CASE_STUDY.md`, and `docs/application/UPDATE_2_2_CASE_STUDY.md`.

## 2.2 observed-regime / hybrid support

Toolkit 2.2 can represent intra-cycle operating modes with `RegimePartition`, exclude switching neighborhoods, and route each stationary dwell mode through the existing mechanism solvers. Fault labels are not eligible regime sources by default. See `docs/HYBRID_REGIME_LAYER.md`.
