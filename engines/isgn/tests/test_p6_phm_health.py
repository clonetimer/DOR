import numpy as np
from isgn.simulation import PublicDataset
from isgn.phm import fit_nominal, anchored_parameter_drift, mechanism_health_evidence, calibrate_phm_health, score_phm_health


def _ds(coeff=2.0, n=4, seed=1, const_u=False):
    rng=np.random.RandomState(seed); dt=.01; T=301
    X=np.zeros((n,T,1),np.float32); U=np.zeros((n,T-1,1),np.float32)
    for k in range(n):
        u=np.zeros(T-1) if const_u else rng.normal(size=T-1)
        U[k,:,0]=u
        for t in range(T-1): X[k,t+1,0]=X[k,t,0]+dt*(coeff*u[t]+0.001*rng.normal())
    schema={'states':[{'name':'x','dimension':'dimensionless','role':'state'}],'inputs':[{'name':'u','dimension':'dimensionless','role':'input'}]}
    return PublicDataset(X,U,np.arange(n),schema,dt,'test','p6')


def test_anchored_drift_recovers_single_coefficient_change():
    normal=_ds(2.0,8,11); nominal=fit_nominal(normal,{'x':['u']})
    fault=_ds(3.0,5,22)
    d,n=anchored_parameter_drift(fault,nominal)
    assert n==['adtheta:x<-u']
    assert np.all(np.isfinite(d))
    assert abs(float(np.median(d))-1.0) < .03


def test_unexcited_edge_is_explicitly_blocked_not_zero_filled():
    train=_ds(2.0,8,33); nominal=fit_nominal(train,{'x':['u']})
    flat=_ds(2.0,4,44,const_u=True)
    d,_=anchored_parameter_drift(flat,nominal)
    assert np.isnan(d).all()
    cal=calibrate_phm_health(flat,nominal)
    assert cal.blocked_features['adtheta:x<-u']=='WINDOW_UNIDENTIFIABLE_OR_NONFINITE'


def test_health_calibration_and_scoring_are_finite_for_eligible_features():
    normal=_ds(2.0,8,55); nominal=fit_nominal(normal,{'x':['u']})
    cal=calibrate_phm_health(normal,nominal)
    fault=_ds(2.5,4,66)
    z,names=score_phm_health(fault,nominal,cal)
    assert names
    assert np.isfinite(z).all()
    j=names.index('adtheta:x<-u')
    assert float(np.median(z[:,j])) > 5.0

def test_slice_time_window_preserves_fixed_rate_contract():
    from isgn.phm import slice_time_window
    ds=_ds(2.0,3,77)
    w=slice_time_window(ds,.10,.40)
    assert w.X.shape[1] == 31
    assert w.U.shape[1] == 30
    assert w.X.shape[0] == 3 and w.dt == ds.dt
