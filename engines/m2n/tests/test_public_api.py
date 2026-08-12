import pytest
import torch
from m2n import (
    Dim, KnowledgeLevel, KnowledgeSource, MechanismSlot, SlotEvidence,
    plan_neural_architecture, ArchitecturePortfolio,
)


def test_posthoc_truth_is_rejected():
    d=Dim()
    slot=MechanismSlot('x',KnowledgeLevel.K1,'x',(d,),d,exact_family='gain',learnable_parameters=('g',),knowledge_source=KnowledgeSource.SIMULATOR_TRUTH_POSTHOC)
    with pytest.raises(ValueError): slot.validate()


def test_blinding_notice_surfaces():
    d=Dim()
    slot=MechanismSlot('x',KnowledgeLevel.K2,'x',(d,),d,finite_candidates=('a','b'),knowledge_source=KnowledgeSource.BLINDED_FOR_VALIDATION)
    p=plan_neural_architecture((slot,),{'x':SlotEvidence(finite_status='PENDING')})
    assert p.planning_status=='READY'
    assert any('BLINDED-FOR-VALIDATION' in r for r in p.reasons)


def test_portfolio_executes():
    class M(torch.nn.Module):
        def __init__(self,g): super().__init__(); self.g=g
        def forward(self,x): return x*self.g
    p=ArchitecturePortfolio({'a':M(1.0),'b':M(2.0)},('gain',))
    o=p(torch.tensor([1.0]))
    assert torch.allclose(o['mean'],torch.tensor([1.5]))
