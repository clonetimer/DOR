# Core Python API

```python
from isgn import (
    PublicDataset,
    recover_missing_terms,
    recover_graph,
    complete_graph,
    build_consensus_model,
    certify_completion,
)
```

## `recover_missing_terms`

Use when a trusted equation part is known and the unknown mechanism can be expressed in a finite candidate term library.

## `recover_graph` / `complete_graph`

Use for unknown or partially known dynamic graph topology. Backends are interchangeable behind the common `RecoveryResult` contract.

## `build_consensus_model`

Use when two independent discovery backends are available. A disagreement is evidence and should not be hidden by selecting the lowest validation error.

## `certify_completion`

Use after recovery to distinguish `COMPLETE`, prior conflict, insufficient identifiability, or Grammar coverage failure.

See docstrings for full arguments. The historical `native` solver and downstream PHM/Atlas modules remain available but are not required for the core mechanism-recovery workflow.
