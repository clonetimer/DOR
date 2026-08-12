import numpy as np
from fec.extensions.calibration_audit import audit_group_pvalues


def test_group_audit_reports_group_imbalance():
    # deterministic synthetic p-values: group a near target, group b badly miscalibrated
    a=np.linspace(.001,.999,200)
    b=np.r_[np.full(80,.01),np.linspace(.2,.99,120)]
    r=audit_group_pvalues({'a':a,'b':b},alpha=.05,min_group_size=20)
    assert r.all_groups_adequate
    ga={x.key:x for x in r.groups}
    assert ga['b'].empirical_fpr > ga['a'].empirical_fpr + .2
    assert r.max_gap > .2
