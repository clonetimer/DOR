# Toolkit 2.2 application-driven update case study

## Trigger

Toolkit 2.1 was first applied unchanged to new real equipment. SKAB and a multi-actuator hydraulic dataset produced bounded refusals/identifiability limits and did **not** justify a Core update. The next application used the ZeMA hydraulic condition-monitoring test rig, a real physical hydraulic system with a repeated 60 s working cycle.

The release-trigger experiment used only the ten cycles for which the component conditions were fully nominal and the dataset stability flag indicated stable operation:

- cooler = 100%;
- valve = 100%;
- pump leakage class = 0;
- accumulator pre-charge = 130 bar;
- stable flag = 0.

No fault class was used to partition or fit the mechanism.

## Frozen 2.1 baseline

The input Toolkit was the released 2.1 wheel with SHA-256:

`2320e953b1fea1f38287c290991d22d73ab59e34f6cf09100032560d217394eb`

The application aligned PS1/PS2 pressure (100 Hz) to FS1 volume flow (10 Hz) by block averaging pressure to 10 Hz. Five cycle-wise folds were used; no sample from a held-out cycle entered fitting.

A deliberately small relation family was tested: `PS2 = a*PS1 + b`.

With all within-cycle modes mixed, median held-out test NRMSE was about `0.5601`.

An observed-process partition was then defined *outside Core* from FS1. Across thresholds 4/5/6 L/min and switch exclusion margins 0.3/0.5/0.7 s, the flow-active dwell mode produced:

- median held-out test NRMSE: `0.0380`;
- median improvement factor over the unpartitioned simple relation: `14.76x`;
- recovered PS1 coefficient range: `1.0072 .. 1.0087`;
- intercept range: `-27.36 .. -27.11 bar`.

The near-zero-flow mode had very little PS2 variation and should not be forced into a stationary relation. A more flexible global Toolkit-2.1 quadratic model could reduce test error to roughly `0.14-0.15`, but it did so by stitching the modes with `PS1^2`, `PS1*FS1`, and `FS1^2` terms rather than representing the operating-mode boundary.

## Gap classification

This is a Core data-contract/routing gap, not a device parser gap:

- `AlgebraicDataset.regime` and `PublicDataset.regime` are dataset-level strings in Toolkit 2.1;
- the same trajectory/cycle cannot declare different stationary operating modes;
- transition samples cannot be explicitly excluded from a mode mechanism;
- cycle-wise validation alone can therefore accept a repeated cross-mode polynomial without testing mode invariance.

## Toolkit 2.2 change

Toolkit 2.2 adds an observed-regime layer, not a new universal hybrid solver:

- `RegimePartition`: sample/interval labels plus transition mask and source role;
- `threshold_regime_partition`: deterministic observed-channel partition with optional hysteresis and transition margin;
- `project_algebraic_regime`: project a stationary dwell mode back to `AlgebraicDataset`;
- `project_public_regime_windows`: extract fixed-length transition-free dwell windows for existing dynamic recovery;
- `discover_regime_algebraic_relations`: route the existing 2.x algebraic solver per mode and explicitly block unsupported/degenerate modes.

Mechanism-eligible partition sources are observed controls, observed process measurements, declared schedules, and external operating context. Fault labels are rejected by default.

## Real post-fix result

Running Toolkit 2.2 on the same five-fold real-data protocol produced:

- median global non-transition baseline NRMSE: `0.5281`;
- median flow-active mode test NRMSE: `0.0380`;
- median improvement factor: `13.91x`;
- the no-flow mode was returned as `DEGENERATE_TARGET_VARIATION`, rather than receiving a forced mechanism.

The output remains candidate relation evidence and sets `certification_allowed=False`. The update does **not** claim that the fitted PS2/PS1 line is a universal hydraulic law.

## Generalization boundary

MetroPT-3, an operational metro-train APU dataset with analogue pressure/current/temperature plus digital compressor/valve/controller states, is the next preferred external validation system for this observed-regime contract. It was identified during this update but was not used as execution evidence because the full official data file was not available in the local execution sandbox.
