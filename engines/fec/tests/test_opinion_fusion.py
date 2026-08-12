import numpy as np
import pytest
from fec.extensions.opinion_fusion import (
    DirichletOpinion, discounted_belief_fusion,
    pairwise_degree_of_conflict, dempster_shafer_pair,
    opinion_from_scalar_anomaly_evidence,
)


def test_opinion_components_sum_to_one():
    o=DirichletOpinion(np.array([4.,1.,0.]))
    assert np.isclose(o.belief.sum()+o.uncertainty,1.0)


def test_dbf_order_invariant():
    ops=[DirichletOpinion(np.array([8.,1.])),DirichletOpinion(np.array([7.,2.])),DirichletOpinion(np.array([1.,8.]))]
    a=discounted_belief_fusion(ops,conflict_power=3)
    b=discounted_belief_fusion(list(reversed(ops)),conflict_power=3)
    assert np.allclose(a.opinion.evidence,b.opinion.evidence,atol=1e-12)


def test_conflicting_source_is_discounted_more():
    a=DirichletOpinion(np.array([12.,1.]))
    b=DirichletOpinion(np.array([10.,1.]))
    c=DirichletOpinion(np.array([1.,12.]))
    r=discounted_belief_fusion([a,b,c])
    # two mutually agreeing sources should retain at least as much support as the odd source
    assert r.source_discount[2] <= max(r.source_discount[0],r.source_discount[1]) + 1e-12
    assert pairwise_degree_of_conflict(a,c) > pairwise_degree_of_conflict(a,b)


def test_dempster_pair_returns_valid_opinion():
    o=dempster_shafer_pair(DirichletOpinion(np.array([4.,1.])),DirichletOpinion(np.array([3.,2.])))
    assert np.all(o.evidence>=0)
    assert np.isclose(o.belief.sum()+o.uncertainty,1.0)


def test_scalar_anomaly_evidence_bridge_is_forbidden():
    with pytest.raises(TypeError):
        opinion_from_scalar_anomaly_evidence(3.2)
