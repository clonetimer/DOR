from isgn.simulation import simulate
from isgn.fallbacks import fit_linear_state_space_fallback, save_fallback, load_fallback


def test_linear_fallback_is_predictive_not_mechanism(tmp_path):
    tr,_=simulate('motor',8,T=70,seed=1); va,_=simulate('motor',5,T=70,seed=2)
    r=fit_linear_state_space_fallback(tr,va)
    assert r.summary.status=='PREDICTIVE_FALLBACK_ONLY'
    assert not r.summary.mechanism_certifiable
    assert r.summary.validation_rollout_nrmse < .05
    p=tmp_path/'fallback.json'; save_fallback(r,p); rr=load_fallback(p)
    assert rr.summary.model_type=='discrete-linear-state-space'
