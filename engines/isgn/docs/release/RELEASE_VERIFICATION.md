# Release verification — Toolkit 2.2.0

## Release decision

**PASS — application-driven observed-regime update.**

Toolkit 2.2 preserves the 2.1 mechanism-recovery solvers and adds a device-neutral regime-routing layer for intra-cycle switched/hybrid operation.

## Real application trigger

The trigger was the ZeMA hydraulic condition-monitoring test rig. The released Toolkit 2.1 wheel was installed into an isolated application directory and used unchanged.

Input 2.1 wheel SHA-256:

`2320e953b1fea1f38287c290991d22d73ab59e34f6cf09100032560d217394eb`

External hydraulic archive SHA-256:

`24128aad2ee45eea7e6b63ebbd9992cdf25d0483a2cebefbfc13bc69079af1f2`

Only ten fully nominal/stable cycles were used for mechanism fitting. No fault label was used as a mode partition.

Frozen-2.1 five-fold / threshold-margin sensitivity baseline:

- simple unpartitioned `PS2 ~ PS1` median test NRMSE: `0.56009`;
- flow-active dwell-mode median test NRMSE after external observed-FS1 partition: `0.03796`;
- median improvement factor: `14.76x`;
- flow-active slope range: `1.00721 .. 1.00873`;
- flow-active intercept range: `-27.36 .. -27.11 bar`.

A flexible global degree-2 model reduces error to roughly `0.14-0.15` but consistently selects a cross-mode polynomial (`PS1^2`, `PS1*FS1`, `FS1^2`). This is predictive stitching, not an explicit operating-mode representation.

## 2.2 post-fix real replay

The built 2.2 wheel was installed into a fresh target directory and the same real protocol was repeated through the public hybrid API.

- global non-transition baseline median test NRMSE: `0.52809`;
- flow-active mode median test NRMSE: `0.03796`;
- median improvement factor: `13.91x`;
- no-flow mode status: `DEGENERATE_TARGET_VARIATION`;
- top-level mode recovery status: `PARTIAL_HYBRID_RELATION_EVIDENCE`;
- `certification_allowed = false`.

The relation remains candidate evidence and is not declared a universal hydraulic law.

## New hybrid regression

`tests/test_hybrid_regimes.py` verifies:

1. two observed modes recover two distinct synthetic relations and beat the stationary global baseline;
2. threshold partitions mask switching neighborhoods;
3. fault-label partitions are rejected by default;
4. a low-variation mode is explicitly blocked;
5. dynamic `PublicDataset` trajectories can be projected into transition-free fixed dwell windows for the existing dynamic solvers.

## Full regression

Pytest collects `118` tests in the release tree:

```text
95 current toolkit tests
23 frozen-v1 research regressions
-------------------------------
118 total
```

Executed in release shards:

```text
31 passed
16 passed
24 passed
24 passed
23 frozen-v1 passed
-------------------
118 passed / 0 failed
```

The same 95 current tests plus the 23 frozen regressions were also run with `PYTHONPATH` pointed at the isolated wheel installation; all passed.

## Source syntax

```text
52 / 52 source Python files AST PASS
```

## Wheel

```text
isgn_toolkit-2.2.0-py3-none-any.whl
SHA-256 95250a1a1e2c348162604d972af9e37e99387ee6b94f61dbcc8bc0058fa2a3f7
```

Source/wheel comparison:

```text
52 source Python files
52 wheel Python files
0 missing
0 extra
0 mismatch
PASS
```

Wheel-only smoke confirms:

- version `2.2.0`;
- import root is the isolated installed wheel target;
- hybrid public API imports successfully;
- hybrid unit tests: `5 / 5 PASS`;
- real ZeMA replay reproduces the expected mode-conditioned result.

## Core-change boundary

Toolkit 2.2 does not add latent-mode learning, a generic switched-system ODE optimizer, or a device-specific ZeMA rule. It adds only observed-regime provenance, transition masking and projection/routing into existing solvers. MetroPT-3 is retained as a preferred future operational validation system; it was not used as local execution evidence for this release.
