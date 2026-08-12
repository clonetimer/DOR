import numpy as np
from fec.extensions.weighted_conformal import weighted_upper_p, LogisticDensityRatio


def test_weighted_conformal_reduces_known_covariate_shift_miscalibration():
    rng=np.random.default_rng(7)
    mu=1.5
    xs=rng.normal(0,1,10000)
    xt_unlab=rng.normal(mu,1,10000)
    xt_test=rng.normal(mu,1,10000)
    rs=np.sort(xs);rank=np.searchsorted(rs,xt_test,side='left')
    p_un=(1+(len(rs)-rank))/(len(rs)+1)
    ratio=LogisticDensityRatio.fit(xs,xt_unlab,random_state=3)
    p_w=weighted_upper_p(xs,ratio.ratio(xs),xt_test,test_weight=ratio.ratio(xt_test))
    fpr_un=float(np.mean(p_un<=.05));fpr_w=float(np.mean(p_w<=.05))
    assert fpr_un>.35
    assert abs(fpr_w-.05)<.02
