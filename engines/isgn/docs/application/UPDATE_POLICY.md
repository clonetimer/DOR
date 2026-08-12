# Application-driven update policy

Toolkit 2.0 froze the first Mechanism Recovery application baseline. Toolkit 2.1 is the first Core feature update executed under this policy.

When an application exposes a new issue:

1. **Reproduce** it with raw/traceable application data and a minimal case.
2. **Classify** the gap: observation/data contract, coordinate/state choice, identifiability, mechanism grammar, discovery backend, certification, or device adapter.
3. **Do not change Core by default.** Device parsing, naming, sensor metadata and vendor-specific behavior belong in adapters/configuration.
4. **Generalization test.** A Core change requires evidence that the gap is a reusable mechanism/method issue, not one device's special case.
5. **Absorb the principle, not the paper implementation.** External research may motivate a capability, but ISGN should implement only the narrow reusable contract needed by mechanism recovery unless the external tool is intentionally integrated as an optional backend.
6. **Add a regression before/with the fix.** A new general mechanism family or recovery path should have at least one controlled positive case and one negative/ambiguity case when applicable.
7. **Compare old vs new behavior.** An update must demonstrate a concrete benefit (recovery, simplification, robustness, or correct refusal), not merely a new module name.
8. **Re-run Mechanism Recovery Benchmark and frozen regression.** No update may silently degrade earlier recovery/refusal behavior.
9. **Document the boundary.** If the issue remains unsupported, say so; do not tune a threshold until a single application passes.

## 2.1 precedent

Toolkit 2.1 followed this sequence:

```text
application/research gap
  -> derivative residual unstable + search space unnecessarily large
  -> identify reusable ideas from UDE / AI Feynman
  -> implement device-neutral trajectory residual + physics reducer
  -> old-vs-new MRB2 cases
  -> real UCI simplification
  -> 113-test full regression
  -> release
```

Recommended release semantics:

- `2.1.x`: bug fixes, adapters and documentation for the 2.1 contracts;
- later `2.x`: backward-compatible reusable recovery paths, mechanism families or optional backends that pass the update policy;
- next major: only when public mechanism-recovery contracts or certificate semantics must change incompatibly.

## 2.2 precedent

Toolkit 2.2 followed the same policy more strictly:

```text
new real equipment applications
  -> two bounded no-update cases (excitation / identifiability)
  -> ZeMA real hydraulic within-cycle mode-mixing case on frozen 2.1 wheel
  -> external observed-FS1 partition prototype
  -> robustness across 5 cycle folds x 3 thresholds x 3 transition margins
  -> classify gap as sample/interval-level regime contract
  -> implement device-neutral observed-regime router
  -> explicit fault-label rejection + transition exclusion + degenerate-mode refusal
  -> real post-fix replay
  -> full current + frozen regression
  -> release
```

The important precedent is that a low predictive error is not sufficient evidence of one stationary mechanism when a fixed operating schedule repeats. Mode invariance and partition provenance must be represented explicitly when the application provides observable operating modes.
