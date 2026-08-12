# Trajectory-level missing dynamics recovery

Use `recover-missing-trajectory` when:

- a trusted partial differential/ordinary equation is available;
- the missing part is expected to depend on declared states/inputs;
- state trajectories are observed;
- numerical derivatives are too noisy or sparse for reliable residual regression.

The method performs multiple shooting from observed initial windows. Known terms remain fixed. A small tanh MLP learns only the unexplained vector field. The MLP is then sampled and distilled into a small declared mechanism basis using held-out trajectories.

The universal approximator is temporary and should not be interpreted as the mechanism. The candidate mechanism is the distilled expression, and it must still be independently validated/certified.

Current scope: fixed-rate ODE-like `PublicDataset`; no claim of SciML UDE parity, stiff adjoints, DAEs, SDEs or delays.
