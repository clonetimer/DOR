import numpy as np

from isgn.observation_scope import ObservationCandidate, compare_observation_candidates, project_state_scope
from isgn.simulation import PublicDataset, simulate


def _with_schema(ds, schema, X=None, dt=None):
    return PublicDataset(
        X=np.asarray(ds.X if X is None else X, dtype=np.float32),
        U=np.asarray(ds.U, dtype=np.float32),
        trajectory_id=np.asarray(ds.trajectory_id),
        schema=schema,
        dt=float(ds.dt if dt is None else dt),
        regime=ds.regime,
        variant=ds.variant,
    )


def test_project_state_scope_is_value_preserving():
    ds, _ = simulate("motor", 4, T=81, dt=0.05, seed=5)
    scoped = project_state_scope(ds, ["w"])
    assert scoped.X.shape == (4, 81, 1)
    assert np.array_equal(scoped.X[:, :, 0], ds.X[:, :, 1])
    assert np.array_equal(scoped.U, ds.U)
    assert scoped.schema["states"][0]["name"] == "w"


def test_compare_routes_direct_and_sensor_resolution_block():
    ds, _ = simulate("motor", 8, T=121, dt=0.05, seed=7)
    schema = {
        "states": [dict(x) for x in ds.schema["states"]],
        "inputs": [dict(x) for x in ds.schema["inputs"]],
    }
    # i has a normal high-margin sensor; w is declared below a deliberately
    # large measurement floor, so the scope planner must not call it READY.
    for i, state in enumerate(schema["states"]):
        sd = float(np.std(ds.X[:, :, i]))
        state["observation_role"] = "measurement"
        state["sensor"] = {
            "sample_period_s": ds.dt,
            "delay_s": 0.0,
            "noise_std": sd / (20.0 if state["name"] == "i" else 1.0),
            "quantization_step": 0.0,
        }
    prod = _with_schema(ds, schema)
    rep = compare_observation_candidates([ObservationCandidate("prod", prod, True)])
    routes = {x.state: x for x in rep.state_routes}
    assert routes["i"].eligibility in ("READY", "CONDITIONAL")
    assert routes["w"].route == "SENSOR_RESOLUTION_OR_EXCITATION_CHANGE_REQUIRED"
    assert routes["w"].eligibility == "BLOCKED"


def test_compare_uses_truth_only_to_diagnose_not_to_make_production_ready():
    ds, _ = simulate("thermal", 16, T=161, dt=0.05, seed=11)
    schema = {
        "states": [dict(ds.schema["states"][0])],
        "inputs": [dict(x) for x in ds.schema["inputs"]],
    }
    schema["states"][0]["observation_role"] = "measurement"
    schema["states"][0]["sensor"] = {
        "sample_period_s": ds.dt,
        "delay_s": 0.0,
        "noise_std": float(np.std(ds.X[:, :, 0])) / 20.0,
        "quantization_step": 0.0,
    }
    rng = np.random.RandomState(123)
    bad_x = ds.X.astype(float) + rng.randn(*ds.X.shape) * np.std(ds.X) * 0.2
    prod = _with_schema(ds, schema, X=bad_x)
    truth_schema = {
        "states": [{**dict(ds.schema["states"][0]), "observation_role": "truth"}],
        "inputs": [dict(x) for x in ds.schema["inputs"]],
    }
    truth = _with_schema(ds, truth_schema)
    rep = compare_observation_candidates([
        ObservationCandidate("measured", prod, True),
        ObservationCandidate("truth", truth, False),
    ])
    route = rep.state_routes[0]
    assert route.eligibility == "BLOCKED"
    assert route.route == "FILTERED_OR_OBSERVER_FRONTEND_REQUIRED"
    assert route.truth_statuses["truth"] in ("PASS", "WARN")
