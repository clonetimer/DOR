from isgn.simulation import simulate
from isgn.fusion import build_operational_model


def test_fusion_resolver_complete_reference_backend():
    tr,_=simulate('tank',8,T=70,seed=1); va,_=simulate('tank',5,T=70,seed=2)
    out=build_operational_model(tr,va,backend='reference-sindy',coverage_floor=1e-4)
    assert out.recovery.summary.backend=='reference-sindy'
    assert out.certificate['status']=='COMPLETE'
    assert out.next_action=='READY_FOR_PHM'
