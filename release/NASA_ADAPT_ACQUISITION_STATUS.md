# NASA ADAPT acquisition status for DOR v0.6

Purpose: prevent the v0.6 software fixture from being confused with a completed empirical ADAPT benchmark.

## Official resources identified

NASA Open Data exposes:

- `dataset_text.zip` — tab-delimited telemetry runs;
- `dataset_matlab.zip`;
- `AdaptHTMLdiagrams.zip` — EPS diagrams;
- `Experiment_Log.xls`;
- ADAPT documentation/PDF resources.

## Local state in this development session

- `Experiment_Log.xls`: acquired from the official NASA resource endpoint outside the release tree;
- `dataset_text.zip`: not locally acquired because the available container downloader rejects ZIP content and shell networking is unavailable;
- `AdaptHTMLdiagrams.zip`: same limitation;
- therefore no full real-run ADAPT root-cause benchmark is included in this release.

## Consequence

`examples/nasa_adapt_v06_source_constrained_fixture/` is only a source-constrained software-contract fixture. Its timestamps/evidence are synthetic. It must not be reported as empirical ADAPT root-cause performance.
