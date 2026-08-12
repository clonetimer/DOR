import numpy as np

from isgn.simulation import simulate, PublicDataset
from isgn.observation_gate import assess_observation
from isgn.fusion import build_operational_model
from isgn.consensus import build_consensus_model


def _replace(ds, *, X=None, U=None):
    return PublicDataset(
        X=np.asarray(ds.X if X is None else X, dtype=np.float32),
        U=np.asarray(ds.U if U is None else U, dtype=np.float32),
        trajectory_id=np.asarray(ds.trajectory_id),
        schema=ds.schema,
        dt=ds.dt,
        regime=ds.regime,
        variant=ds.variant,
    )


def test_clean_four_systems_are_qualified():
    for system in ("thermal", "tank", "motor", "msd"):
        ds, _ = simulate(system, 24, T=161, dt=0.05, seed=101, regime="iid", variant="clean")
        rep = assess_observation(ds)
        assert rep.can_run_discovery, (system, rep.to_dict())
        assert rep.status in ("QUALIFIED", "CONDITIONALLY_QUALIFIED")
        assert not rep.blocked_states


def test_unexcited_command_and_collinear_sources_are_blocked():
    thermal, _ = simulate("thermal", 24, T=161, dt=0.05, seed=101)
    U = thermal.U.copy(); U[:,:,0] = 1.5
    r = assess_observation(_replace(thermal, U=U))
    assert r.status == "OBSERVATION_INSUFFICIENT"
    assert any(c.name == "persistent_excitation" and c.status == "FAIL" for c in r.checks)

    tank, _ = simulate("tank", 24, T=161, dt=0.05, seed=101)
    U = tank.U.copy(); U[:,:,1] = 0.5 * U[:,:,0]
    r = assess_observation(_replace(tank, U=U))
    assert r.status == "OBSERVATION_INSUFFICIENT"
    assert any(c.name == "source_identifiability" and c.status == "FAIL" for c in r.checks)


def test_noise_progression_warns_then_blocks_derivative_identification():
    base, _ = simulate("thermal", 32, T=201, dt=0.05, seed=101)
    sd = np.std(base.X, axis=(0,1), keepdims=True)
    rng = np.random.RandomState(9)
    x1 = base.X.astype(float) + rng.randn(*base.X.shape) * sd * 0.01
    r1 = assess_observation(_replace(base, X=x1))
    assert r1.status == "CONDITIONALLY_QUALIFIED"
    assert r1.states[0].status == "WARN"

    rng = np.random.RandomState(9)
    x2 = base.X.astype(float) + rng.randn(*base.X.shape) * sd * 0.02
    r2 = assess_observation(_replace(base, X=x2))
    assert r2.status == "OBSERVATION_INSUFFICIENT"
    assert r2.states[0].status == "FAIL"


def test_high_level_build_and_consensus_stop_before_backend_on_failed_gate(tmp_path):
    ds, _ = simulate("thermal", 16, T=121, dt=0.05, seed=101)
    U = ds.U.copy(); U[:,:,0] = 1.0
    bad = _replace(ds, U=U)
    out = build_operational_model(bad, bad, backend="does-not-exist")
    assert out.recovery is None
    assert out.certificate["status"] == "OBSERVATION_INSUFFICIENT"
    assert out.backend_decision["backend"] == "none"

    co = build_consensus_model(bad, bad, backends=["does-not-exist", "native"])
    assert co.primary is None
    assert co.consensus.status == "OBSERVATION_INSUFFICIENT"
    assert co.candidates == {}


def test_sensor_contract_blocks_noise_dominated_state_variation():
    ds, _ = simulate("thermal", 8, T=121, dt=0.05, seed=101)
    # Keep the same data but declare a sensor whose noise floor exceeds the
    # observed thermal-state variation.  The Gate must not count sensor noise as
    # evidence that the physical state is usefully excited/observable.
    schema = {"states": [dict(ds.schema["states"][0])], "inputs": [dict(x) for x in ds.schema["inputs"]]}
    observed_std = float(np.std(ds.X[:,:,0]))
    schema["states"][0]["sensor"] = {
        "sample_period_s": ds.dt,
        "delay_s": 0.0,
        "noise_std": observed_std,
        "quantization_step": 0.0,
    }
    noisy_contract = PublicDataset(ds.X, ds.U, ds.trajectory_id, schema, ds.dt, ds.regime, ds.variant)
    r = assess_observation(noisy_contract)
    mc = next(c for c in r.checks if c.name == "measurement_chain")
    assert mc.status == "FAIL"
    assert "T" in mc.metrics["resolution_insufficient_states"]
    assert "T" in r.blocked_states
    assert "T" not in r.usable_states
    assert r.status == "OBSERVATION_INSUFFICIENT"


def test_sensor_contract_is_nonintrusive_without_sensor_metadata():
    ds, _ = simulate("tank", 8, T=121, dt=0.05, seed=101)
    r = assess_observation(ds)
    mc = next(c for c in r.checks if c.name == "measurement_chain")
    assert mc.status == "N/A"
    assert r.can_run_discovery
