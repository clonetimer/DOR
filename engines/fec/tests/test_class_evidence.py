import numpy as np, pytest
from fec.extensions.class_evidence import ClassEvidenceBatch,validate_shared_frame

def test_class_evidence_shared_frame():
    a=ClassEvidenceBatch(np.array([[2.,1.],[1.,2.]]),('ok','bad'),'s1','adapter-v1')
    b=ClassEvidenceBatch(np.array([[3.,1.],[1.,3.]]),('ok','bad'),'s2','adapter-v1')
    assert validate_shared_frame([a,b])==('ok','bad')
    assert len(a.opinions())==2

def test_mismatched_frame_rejected():
    a=ClassEvidenceBatch(np.ones((2,2)),('a','b'),'s1','p')
    b=ClassEvidenceBatch(np.ones((2,2)),('b','a'),'s2','p')
    with pytest.raises(ValueError): validate_shared_frame([a,b])
