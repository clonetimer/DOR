import numpy as np
import pytest
from isgn.contracts import AlgebraicDataset
from isgn.tabulated import fit_tabulated_1d, Tabulated1DResult


def _ds(x,y,regime='r',variant='v'):
    return AlgebraicDataset(values=np.column_stack([x,y]),sample_id=np.arange(len(x)),
        schema={'variables':[{'name':'x','dim':'1'},{'name':'y','dim':'1'}]},regime=regime,variant=variant)


def test_tabulated_pchip_completes_generic_nonlinear_constitutive_gap():
    x=np.linspace(0.0,1.0,101); y=np.sqrt(x+0.02)+0.03*np.tanh(14*(x-.62))
    # contiguous holes, not a random-row leakage split
    hold=((x>=.22)&(x<=.30))|((x>=.68)&(x<=.76))
    r=fit_tabulated_1d(_ds(x[~hold],y[~hold],variant='train'),_ds(x[hold],y[hold],variant='val'),input='x',target='y')
    assert r.method == 'pchip'
    assert r.validation_nrmse < .08
    assert r.candidate_only is True and r.certification_allowed is False
    pred=r.predict(x[hold])
    assert np.isfinite(pred).all()
    # serialized payload remains executable after explicit reconstruction
    rr=Tabulated1DResult.from_dict(r.to_dict())
    assert np.allclose(rr.predict(x[hold]),pred)


def test_tabulated_refuses_extrapolation():
    x=np.linspace(.1,.9,41); y=np.sin(x)
    r=fit_tabulated_1d(_ds(x,y),_ds(x[10:20],y[10:20]),input='x',target='y')
    with pytest.raises(ValueError,match='OUT_OF_SUPPORT'):
        r.predict(np.array([.05,.5]))
    loose=r.predict(np.array([.05,.5]),strict=False)
    assert np.isnan(loose[0]) and np.isfinite(loose[1])


def test_tabulated_refuses_context_mismatch():
    x=np.linspace(.1,.9,41); y=x*x
    r=fit_tabulated_1d(_ds(x,y,regime='temperature=25C'),_ds(x[10:20],y[10:20],regime='temperature=25C'),input='x',target='y')
    with pytest.raises(ValueError,match='CONTEXT_MISMATCH'):
        r.predict_dataset(_ds(x[10:20],y[10:20],regime='temperature=35C'))


def test_tabulated_refuses_multivalued_relation():
    x=np.array([0,.1,.2,.3,.4,.5,.5,.6,.7,.8,1.0],float)
    y=np.array([0,.1,.2,.3,.4,.5,.8,.6,.7,.8,1.0],float)
    with pytest.raises(ValueError,match='MULTIVALUED_RELATION'):
        fit_tabulated_1d(_ds(x,y),_ds(np.array([.15,.25,.35]),np.array([.15,.25,.35])),input='x',target='y',duplicate_atol=1e-3)

def test_tabulated_cli_roundtrip(tmp_path):
    import json
    from isgn.io import save_algebraic_npz
    from isgn.cli import main
    x=np.linspace(.0,1.0,41); y=np.sqrt(x+.1)
    hold=(x>.35)&(x<.55)
    tr=_ds(x[~hold],y[~hold],regime='same',variant='train')
    va=_ds(x[hold],y[hold],regime='same',variant='val')
    trp=tmp_path/'tr.npz'; vap=tmp_path/'va.npz'; model=tmp_path/'model.json'; out=tmp_path/'out.json'
    save_algebraic_npz(tr,trp); save_algebraic_npz(va,vap)
    import sys
    old=sys.argv
    try:
        sys.argv=['isgn','fit-tabulated-1d','--train',str(trp),'--val',str(vap),'--input','x','--target','y','--model',str(model),'--out',str(out)]
        main()
    finally:
        sys.argv=old
    payload=json.loads(model.read_text())
    assert payload['method']=='pchip'
    assert payload['support_min']==0.0 and payload['support_max']==1.0
    assert json.loads(out.read_text())['status']=='PASS'
