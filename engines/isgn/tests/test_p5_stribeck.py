import numpy as np

from isgn.api import RecoveryResult, RecoverySummary, save_checkpoint, load_checkpoint, fit_normal_phm, extract_mechanism_features
from isgn.backends.base import CanonicalMechanismModel
from isgn.consensus import operator_parameter_max_rel_diff
from isgn.friction import smooth_stribeck_transform, profile_smooth_stribeck
from isgn.io import from_arrays


def _friction_dataset(*, phase=0.0, ntraj=2, steps=2400, dt=0.002):
    schema={"states":[{"name":"velocity","dim":"velocity"}],
            "inputs":[{"name":"drive","dim":"acceleration","role":"input"}]}
    X=[]; U=[]
    eps=0.05; vs=0.20; rv=0.40; rs=0.55; c=-0.8; drive_c=1.15
    for r in range(ntraj):
        t=np.arange(steps,dtype=float)*dt
        u=(0.9*np.sin(2*np.pi*0.7*t+phase+0.4*r)
           +0.45*np.sin(2*np.pi*1.9*t+0.2*r))
        x=np.zeros(steps+1,dtype=float); x[0]=0.03*(r+1)
        def rhs(v,uk):
            return c*smooth_stribeck_transform(np.asarray(v),{
                "smoothing_velocity":eps,"breakaway_velocity":vs,
                "viscous_ratio":rv,"stribeck_ratio":rs}) + drive_c*uk
        for k in range(steps):
            # Constant input over the interval, RK4 on the state.
            k1=rhs(x[k],u[k]); k2=rhs(x[k]+0.5*dt*k1,u[k])
            k3=rhs(x[k]+0.5*dt*k2,u[k]); k4=rhs(x[k]+dt*k3,u[k])
            x[k+1]=x[k]+dt*(k1+2*k2+2*k3+k4)/6.0
        X.append(x[:,None]); U.append(u[:,None])
    return from_arrays(np.asarray(X),np.asarray(U),schema=schema,dt=dt,variant="smooth-stribeck-synthetic")


def _identity_template(ds):
    graph={"velocity":["velocity","drive"]}
    coeff={"velocity":{"velocity":-1.0,"drive":1.0}}
    ops={"velocity":{"velocity":"Identity","drive":"Identity"}}
    params={"velocity":{"velocity":None,"drive":None}}
    model=CanonicalMechanismModel(schema=ds.schema,dt=ds.dt,coefficients=coeff,operators=ops,operator_parameters=params)
    summ=RecoverySummary(graph=graph,operators=ops,coefficients=coeff,val_nrmse=0.0,seed=7,
                         constitutive_search=False,backend="test-template",operator_parameters=params)
    return RecoveryResult(model,summ,ds.schema,ds.dt)


def test_smooth_stribeck_transform_matches_expanded_basis():
    z=np.linspace(-0.4,0.4,101)
    p={"smoothing_velocity":0.05,"breakaway_velocity":0.2,"viscous_ratio":0.4,"stribeck_ratio":0.55}
    th=np.tanh(z/0.05)
    expected=0.4*z + th + 0.55*np.exp(-(np.abs(z)/0.2)**2)*th
    assert np.allclose(smooth_stribeck_transform(z,p),expected,rtol=1e-12,atol=1e-12)


def test_profile_smooth_stribeck_recovers_reusable_constitutive_family(tmp_path):
    tr=_friction_dataset(phase=0.0)
    va=_friction_dataset(phase=0.73,ntraj=1)
    template=_identity_template(tr)
    rr,profile=profile_smooth_stribeck(
        template,tr,va,target="velocity",source="velocity",
        smoothing_grid=[0.025,0.05,0.1],breakaway_grid=[0.1,0.2,0.4],
        expected_signs={"velocity":-1,"drive":1},ridge=1e-10)
    assert rr.summary.operators["velocity"]["velocity"] == "SmoothStribeck"
    assert profile.smoothing_velocity == 0.05
    assert profile.breakaway_velocity == 0.2
    assert profile.validation_target_nrmse < 0.04
    assert profile.truth_channels_used is False
    assert rr.summary.coefficients["velocity"]["velocity"] < 0
    assert rr.summary.coefficients["velocity"]["drive"] > 0
    # Canonical dict-valued operator parameters must survive checkpoints.
    path=tmp_path/"m.pt"; save_checkpoint(rr,path); loaded=load_checkpoint(path)
    assert loaded.summary.operator_parameters["velocity"]["velocity"]["breakaway_velocity"] == 0.2
    assert np.isfinite(loaded.model.rhs_numpy(va.X[:,:-1],va.U)).all()
    # PHM nominal/refit must use the constitutive transform rather than silently
    # reverting the velocity edge to a linear regressor.
    nom=fit_normal_phm(tr,rr.summary.graph,operators=rr.summary.operators,operator_parameters=rr.summary.operator_parameters)
    feat=extract_mechanism_features(va,nom)
    assert nom.operators["velocity"]["velocity"] == "SmoothStribeck"
    assert abs(nom.coefficients["velocity"]["velocity"] + 0.8) < 0.08
    assert np.isfinite(feat).all()


def test_consensus_parameter_comparison_supports_mapping_parameters():
    ds=_friction_dataset(steps=200)
    a=_identity_template(ds); b=_identity_template(ds)
    p={"smoothing_velocity":0.05,"breakaway_velocity":0.2,"viscous_ratio":0.4,"stribeck_ratio":0.55}
    for r in (a,b):
        r.summary.operators["velocity"]["velocity"]="SmoothStribeck"
        r.summary.operator_parameters["velocity"]["velocity"]=dict(p)
    assert operator_parameter_max_rel_diff(a,b) == 0.0
    b.summary.operator_parameters["velocity"]["velocity"]["breakaway_velocity"]=0.25
    assert 0.19 < operator_parameter_max_rel_diff(a,b) < 0.21


def test_cli_prior_loader_accepts_wrapped_mechanism_prior(tmp_path):
    import json
    from isgn.cli import _load_prior
    p=tmp_path/"prior.json"
    p.write_text(json.dumps({"known_present":{"position":["velocity"]},
                             "known_absent":{"velocity":["command"]},
                             "candidate_edges_after_prior":3}),encoding="utf-8")
    assert _load_prior(p,"known_present") == {"position":["velocity"]}
    assert _load_prior(p,"known_absent") == {"velocity":["command"]}
