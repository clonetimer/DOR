from isgn.simulation import simulate
from isgn.consensus import build_consensus_model, graph_jaccard


def test_graph_jaccard_basic():
    assert graph_jaccard({'x':['u']},{'x':['u']})==1.0
    assert graph_jaccard({'x':['u']},{'x':['x']})==0.0


def test_consensus_native_reference_tank():
    tr,_=simulate('tank',12,T=90,seed=11)
    va,_=simulate('tank',8,T=90,seed=12)
    out=build_consensus_model(tr,va,backends=['native','reference-sindy'],epochs=500,refit_epochs=250,
                              coverage_floor=1e-4)
    assert out.consensus.status=='CONSENSUS'
    assert out.consensus.graph_jaccard==1.0
    assert out.consensus.operator_agreement==1.0
    assert out.next_action=='READY_FOR_PHM'


def test_consensus_reports_backend_disagreement():
    tr,_=simulate('motor',10,T=80,seed=21)
    va,_=simulate('motor',6,T=80,seed=22)
    # Force reference backend to omit a real edge; native receives no such restriction.
    # Build separately then use a lightweight fabricated disagreement via known-absent on consensus is not possible per-backend,
    # so this unit test focuses on graph metric while end-to-end disagreement is covered by status logic in a direct report test below.
    from isgn.api import recover_graph, certify_completion
    from isgn.consensus import ConsensusReport, _edge_diff
    a=recover_graph(tr,va,backend='native',epochs=350,refit_epochs=180)
    b=recover_graph(tr,va,backend='reference-sindy',known_absent={'i':['w']})
    assert graph_jaccard(a.summary.graph,b.summary.graph)<1.0
    d=_edge_diff(a,b)
    assert any('w->i' in xs for xs in d.values())


def test_structural_consensus_refit_resolves_backend_fit_certificate_noise():
    tr,_=simulate('thermal',16,T=100,seed=101)
    va,_=simulate('thermal',8,T=100,seed=1101)
    out=build_consensus_model(tr,va,backends=['native','reference-sindy'],seed=101,epochs=500,refit_epochs=250,coverage_floor=1e-3)
    assert out.consensus.status=='CONSENSUS'
    assert out.consensus.graph_exact
    assert out.consensus.primary_backend=='consensus-refit'
    assert out.consensus_certificate['status']=='COMPLETE'
    assert out.next_action=='READY_FOR_PHM'

def test_consensus_rejects_real_low_data_disagreement():
    tr,_=simulate('thermal',2,T=45,seed=1)
    va,_=simulate('thermal',3,T=45,seed=1001)
    out=build_consensus_model(tr,va,backends=['native','reference-sindy'],seed=1,epochs=70,refit_epochs=40,coverage_floor=1e-3)
    assert out.consensus.status=='BACKEND_DISAGREEMENT'
    assert out.consensus.graph_jaccard < 0.8
    assert out.next_action=='REVIEW_BACKEND_DISAGREEMENT'
