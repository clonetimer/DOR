# Data contract

Dynamic data use `PublicDataset`:

```text
X: [trajectory, time, state]
U: [trajectory, time-1, input]
dt: fixed positive sampling interval
schema.states / schema.inputs: names, dimensions, roles, optional sensor metadata
```

Algebraic/constitutive data use `AlgebraicDataset`:

```text
values: [sample, variable]
sample_id: [sample]
schema.variables
```

Train/validation/test splits should be made at the trajectory, experiment, cycle, or operating-run level to avoid temporal leakage.
