import numpy as np

from isgn.api import RecoveryResult, RecoverySummary, calibrate_completion_coverage, certify_completion
from isgn.backends.base import CanonicalMechanismModel
from isgn.io import from_arrays


def _replicate_oscillator(*, repeats=6, noise=2e-4, seed=0):
    dt=0.01
    t=np.arange(501,dtype=float)*dt
    truth=np.stack([np.sin(t),np.cos(t)],axis=-1)
    rng=np.random.RandomState(seed)
    X=np.stack([truth+rng.normal(scale=noise,size=truth.shape) for _ in range(repeats)],axis=0)
    U=np.zeros((repeats,len(t)-1,0),dtype=float)
    schema={"states":[{"name":"x","dim":"displacement"},{"name":"v","dim":"velocity"}],"inputs":[]}
    return from_arrays(X,U,schema=schema,dt=dt,variant="replicate-calibration")


def _result(ds, coeff, graph):
    ops={t:{s:"Identity" for s in srcs} for t,srcs in graph.items()}
    params={t:{s:None for s in srcs} for t,srcs in graph.items()}
    model=CanonicalMechanismModel(schema=ds.schema,dt=ds.dt,coefficients=coeff,operators=ops,operator_parameters=params)
    summary=RecoverySummary(graph=graph,operators=ops,coefficients=coeff,val_nrmse=0.0,seed=0,
                            constitutive_search=False,backend="test-canonical",operator_parameters=params)
    return RecoveryResult(model=model,summary=summary,schema=ds.schema,dt=ds.dt)


def test_pairwise_replicate_calibration_accepts_covered_mechanism():
    ds=_replicate_oscillator()
    r=_result(ds,{"x":{"v":1.0},"v":{"x":-1.0}},{"x":["v"],"v":["x"]})
    cal=calibrate_completion_coverage(r,[ds],quantile=.95)
    cert=certify_completion(r,ds,coverage_calibration=cal,coverage_threshold=8.0)
    assert cert["status"] == "COMPLETE"
    assert max(v["coverage_ratio"] for v in cert["per_target_coverage"].values()) < 8.0
    assert cal.assumptions["deterministic_model_bias_cancelled_by_pairwise_difference"] is True


def test_per_target_coverage_detects_local_grammar_gap():
    ds=_replicate_oscillator()
    # x equation is covered, v equation is deliberately missing its -x mechanism.
    r=_result(ds,{"x":{"v":1.0},"v":{}},{"x":["v"],"v":[]})
    cal=calibrate_completion_coverage(r,[ds],quantile=.95)
    cert=certify_completion(r,ds,coverage_calibration=cal,coverage_threshold=8.0)
    assert cert["status"] == "GRAMMAR_UNCOVERED"
    assert cert["coverage_target"] == "v"
    assert cert["per_target_coverage"]["x"]["coverage_ratio"] < 8.0
    assert cert["per_target_coverage"]["v"]["coverage_ratio"] >= 8.0


def test_incomplete_target_calibration_refuses_certificate():
    ds=_replicate_oscillator()
    r=_result(ds,{"x":{"v":1.0},"v":{"x":-1.0}},{"x":["v"],"v":["x"]})
    cal=calibrate_completion_coverage(r,[ds],quantile=.95).to_dict()
    del cal["target_floors"]["v"]
    cert=certify_completion(r,ds,coverage_calibration=cal)
    assert cert["status"] == "UNCERTIFIED"
    assert cert["reason"] == "per_target_coverage_calibration_incomplete"
    assert cert["missing_targets"] == ["v"]
