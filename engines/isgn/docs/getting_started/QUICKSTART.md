# Quickstart

## 1. Environment check

```bash
isgn doctor
isgn backends
```

Optional `pysindy` and external SymMatika may be unavailable. The toolkit must report that explicitly; do not interpret `reference-sindy` as official PySINDy.

## 2. Partial physics with usable derivatives

Suppose

```text
omega_dot = -3*sin(theta) + u + [missing]
```

```bash
isgn recover-missing \
  --train train.npz --val val.npz \
  --target omega \
  --known-terms '{"sin(theta)": -3.0, "u": 1.0}' \
  --candidates '["theta","omega","cos(theta)","square(theta)"]' \
  --out missing.json
```

## 3. Partial physics with sparse/noisy trajectories

If differentiation is unreliable but all state equations have a known part, use whole-trajectory fitting:

```bash
isgn recover-missing-trajectory \
  --train train.npz --val val.npz \
  --known-terms '{"theta":{"omega":1.0},"omega":{"sin(theta)":-3.0,"u":1.0}}' \
  --unknown-targets omega \
  --candidates '{"omega":["omega","theta","cos(theta)","square(theta)"]}' \
  --out missing_trajectory.json
```

The neural residual is temporary. The result to inspect is `distilled_terms` / `distilled_coefficients` and its held-out trajectory error.

## 4. Reduce a large algebraic search before discovery

Create/save `AlgebraicDataset` NPZ files, then:

```bash
isgn reduce-physics \
  --train train_table.npz --val val_table.npz \
  --inputs p1,p2,q --target P \
  --max-order 2 --max-terms 2 \
  --out reduction.json
```

Use a proposed reduction only as a new search coordinate. It is not a certificate.

## 5. Partial graph completion

```bash
isgn complete \
  --train train.npz --val val.npz \
  --known-present known_present.json \
  --known-absent known_absent.json \
  --backend reference-sindy \
  --backend-options '{"normalize_columns": true}' \
  --model mechanism.pt --summary recovery.json
```

## 6. Certification

```bash
isgn certify --model mechanism.pt --data validation.npz \
  --coverage-calibration coverage.json --out certificate.json
```

Possible decisions include `COMPLETE`, `PRIOR_CONFLICT`, `GRAMMAR_UNCOVERED`, and `UNIDENTIFIABLE`.

## 7. Release benchmark

```bash
python benchmarks/mechanism_recovery_v2/run.py \
  --uci-hydraulic /path/to/condition+monitoring+of+hydraulic+systems.zip \
  --out results/mechanism_recovery_v2/benchmark.json
```
