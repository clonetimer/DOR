import numpy as np
from fec.extensions.reference_cleaning import ActiveReferenceCleaner

def test_active_cleaner_removes_only_reviewed_confirmed_outliers():
    s=np.array([0.,1.,2.,3.,4.,5.])
    c=ActiveReferenceCleaner(budget=3); plan=c.propose(s)
    assert plan.candidate_indices.tolist()==[5,4,3]
    res=c.apply(s,plan,confirmed_outlier=[True,False,True])
    assert set(res.removed_indices.tolist())=={5,3}
    assert 4. in res.cleaned_scores
