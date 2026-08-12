# NASA ADAPT published-source fixture — provenance

This directory is a **partial semantic fixture** for DOR v0.5 contract development. It is not a bundled or complete copy of NASA ADAPT telemetry.

Primary sources used to constrain the fixture:

1. NASA Open Data Portal — ADAPT Dataset  
   https://data.nasa.gov/dataset/adapt-dataset  
   Supports the dataset/testbed scope, 128 sensors, 2 Hz acquisition, controlled repeatable physical/software fault insertion, and the official resources (`dataset_text.zip`, `Experiment_Log.xls`, `AdaptHTMLdiagrams.zip`).

2. NASA NTRS — Benchmarking Diagnostic Algorithms on an Electrical Power System Testbed  
   https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20110011109.pdf  
   Supports the ADAPT/DX benchmarking fault taxonomy and published sensor/fault examples.

3. NASA NTRS — Efficient Probabilistic Diagnostics for Electrical Power Systems  
   https://ntrs.nasa.gov/citations/20110012876  
   Supports published diagnostic examples including sensor and relay fault modes and simultaneous-fault cases.

4. NASA Open Data / NTRS — Diagnosing Faults in Electrical Power Systems of Spacecraft and Aircraft  
   https://data.nasa.gov/dataset/diagnosing-faults-in-electrical-power-systems-of-spacecraft-and-aircraft  
   Supports component-oriented diagnostic modeling of the ADAPT electrical power system.

## Reproducibility note

The source fixture contains only semantics that could be directly supported by the sources above. It deliberately does **not** claim to reproduce the full ADAPT component graph, all fault modes, or full telemetry behavior. Those are v0.6 acquisition/validation requirements.
