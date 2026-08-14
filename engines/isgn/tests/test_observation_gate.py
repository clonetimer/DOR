import numpy as np

from isgn.simulation import simulate, PublicDataset
from isgn.observation_gate import assess_observation
from isgn.fusion import build_operational_model
from isgn.consensus import build_consensus_model


def _replace(ds, *, X=None, U=None, sample_time_s=None):
    return PublicDataset(
        X=np.asarray(ds.X if X is None else X, dtype=np.float32),
        U=np.asarray(ds.U if U is None else U, dtype=np.float32),
        trajectory_id=np.asarray(ds.trajectory_id),
        schema=ds.schema,
        dt=ds.dt,
        regime=ds.regime,
        variant=ds.variant,
        sample_time_s=(ds.sample_time_s if sample_time_s is None else sample_time_s),
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


def test_timebase_contract_passes_uniform_time_and_blocks_hidden_gap():
    ds, _ = simulate("thermal", 8, T=121, dt=0.05, seed=101)
    t = np.broadcast_to(np.arange(ds.X.shape[1], dtype=float) * ds.dt, ds.X.shape[:2]).copy()
    good = _replace(ds, sample_time_s=t)
    rg = assess_observation(good)
    tc = next(c for c in rg.checks if c.name == "timebase_integrity")
    assert tc.status == "PASS"
    assert rg.can_run_discovery

    bad_t = t.copy()
    bad_t[:, 60:] += 20.0 * ds.dt
    bad = _replace(ds, sample_time_s=bad_t)
    rb = assess_observation(bad)
    tc = next(c for c in rb.checks if c.name == "timebase_integrity")
    assert tc.status == "FAIL"
    assert tc.metrics["max_interval_ratio"] > 2.0
    assert rb.status == "OBSERVATION_INSUFFICIENT"
    assert rb.can_run_discovery is False


def test_timebase_contract_warns_on_material_but_bounded_jitter():
    ds, _ = simulate("thermal", 8, T=121, dt=0.05, seed=101)
    intervals = np.full(ds.X.shape[1]-1, ds.dt, dtype=float)
    intervals[::10] *= 0.90
    t1 = np.concatenate([[0.0], np.cumsum(intervals)])
    t = np.broadcast_to(t1, ds.X.shape[:2]).copy()
    r = assess_observation(_replace(ds, sample_time_s=t))
    tc = next(c for c in r.checks if c.name == "timebase_integrity")
    assert tc.status == "WARN"
    assert r.status == "CONDITIONALLY_QUALIFIED"
    assert r.can_run_discovery


def test_temporal_sufficiency_passes_known_full_state_and_blocks_missing_state():
    # Full first-order state + inputs: current observation should already explain
    # the derivative, so adding lagged history must not be required.
    full, _ = simulate("motor", 24, T=161, dt=0.05, seed=101, regime="iid", variant="clean")
    rf = assess_observation(full)
    tf = next(c for c in rf.checks if c.name == "temporal_sufficiency")
    assert tf.status == "PASS", tf.to_dict()
    assert rf.can_run_discovery

    # Remove motor speed while retaining the measured current and both declared
    # inputs.  Frozen 2.3rc1 treated this scope as QUALIFIED even though current
    # history acts as a strong proxy for the omitted dynamic state.  The new
    # preflight must refuse the full-state mechanism claim rather than treating
    # lag coefficients as a mechanism.
    schema = {
        "states": [dict(full.schema["states"][0])],
        "inputs": [dict(x) for x in full.schema["inputs"]],
    }
    partial = PublicDataset(
        X=np.asarray(full.X)[:, :, :1],
        U=np.asarray(full.U),
        trajectory_id=np.asarray(full.trajectory_id),
        schema=schema,
        dt=full.dt,
        regime=full.regime,
        variant=full.variant,
    )
    rp = assess_observation(partial)
    tp = next(c for c in rp.checks if c.name == "temporal_sufficiency")
    assert tp.status == "FAIL", tp.to_dict()
    assert "i" in tp.metrics["insufficient_states"]
    assert tp.metrics["history_gain_ratio"]["i"] < 0.40
    assert rp.status == "OBSERVATION_INSUFFICIENT"
    assert rp.can_run_discovery is False
    assert "i" in rp.blocked_states


def test_temporal_sufficiency_does_not_false_reject_full_state_msd():
    ds, _ = simulate("msd", 24, T=161, dt=0.05, seed=303, regime="iid", variant="clean")
    r = assess_observation(ds)
    tc = next(c for c in r.checks if c.name == "temporal_sufficiency")
    assert tc.status == "PASS", tc.to_dict()
    assert r.can_run_discovery


def test_temporal_sufficiency_respects_declared_context_modulation_without_masking_missing_state():
    # Parameter-varying oscillator: the *full* instantaneous Markov state is
    # [x1, x2], while a declared context scales the rotation rate.  An additive
    # X+context probe cannot represent c*X and would falsely attribute the
    # resulting model error to temporal memory.  The context-aware preflight
    # should pass the full state while still rejecting x1-only observation.
    dt = 0.01
    T = 401
    contexts = [0.6, 0.8, 1.0, 1.2] * 3
    X = []
    U = []
    ids = []
    for n, c in enumerate(contexts):
        omega = 2.0 * np.pi * 1.5 * c
        t = np.arange(T, dtype=float) * dt
        phase = 0.37 * n
        X.append(np.stack([np.sin(omega*t + phase), np.cos(omega*t + phase)], axis=-1))
        U.append(np.full((T-1, 1), c, dtype=float))
        ids.append(f"ctx-{n}")
    schema = {
        "states": [{"name": "x1"}, {"name": "x2"}],
        "inputs": [{"name": "speed_context", "role": "context"}],
    }
    full = PublicDataset(np.asarray(X, dtype=np.float32), np.asarray(U, dtype=np.float32),
                         np.asarray(ids), schema, dt, None, None)
    rf = assess_observation(full)
    tf = next(c for c in rf.checks if c.name == "temporal_sufficiency")
    assert tf.status == "PASS", tf.to_dict()
    assert tf.metrics["context_interactions_enabled"] is True
    assert tf.metrics["context_inputs"] == ["speed_context"]
    assert max(tf.metrics["instantaneous_nrmse"].values()) < 0.02
    assert rf.can_run_discovery

    partial_schema = {
        "states": [{"name": "x1"}],
        "inputs": [{"name": "speed_context", "role": "context"}],
    }
    partial = PublicDataset(np.asarray(X, dtype=np.float32)[:, :, :1], np.asarray(U, dtype=np.float32),
                            np.asarray(ids), partial_schema, dt, None, None)
    rp = assess_observation(partial)
    tp = next(c for c in rp.checks if c.name == "temporal_sufficiency")
    assert tp.status == "FAIL", tp.to_dict()
    assert tp.metrics["instantaneous_nrmse"]["x1"] > 0.90
    assert tp.metrics["best_history_nrmse"]["x1"] < 0.10
    assert "x1" in tp.metrics["insufficient_states"]
    assert rp.can_run_discovery is False
