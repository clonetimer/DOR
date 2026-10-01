# Phase 2 external data acquisition status

## UCI Hydraulic 447

Official source:
`https://archive.ics.uci.edu/static/public/447/condition%2Bmonitoring%2Bof%2Bhydraulic%2Bsystems.zip`

Current runtime status: **BLOCKED_BY_NETWORK_RUNTIME**. The execution container cannot resolve external DNS, so official archive bytes were not acquired in this phase. `release/source_intake_specs/UCI_HYDRAULIC_447.json` is ready for externally downloaded bytes and the existing adapter then validates the exact 2,205-cycle matrix shapes.

## NASA ADAPT

Official resources:

- `https://c3.nasa.gov/dashlink/static/media/dataset/dataset_text.zip`
- `https://c3.nasa.gov/dashlink/static/media/dataset/Experiment_Log.xls`
- `https://c3.nasa.gov/dashlink/static/media/dataset/AdaptHTMLdiagrams.zip`

Current runtime status: **BLOCKED_BY_NETWORK_RUNTIME**. Resource identities and legacy listed sizes are frozen in `release/source_intake_specs/`, but no binary resource is claimed acquired or validated in this release.
