import torch
from isgn.atlas import MechanismAtlas, MechanismScopeRecord, UncoveredRegion
from isgn.architecture import compile_mechanism_atlas, build_compiled_network


def test_compiler_enforces_certified_receptive_field_by_construction():
    a=MechanismAtlas('demo')
    a.add_scope(MechanismScopeRecord('s',['y1','y2'],observed_drivers=['x1','x2','x3'],
        graph={'y1':['x1','x2'],'y2':['x3']},completion_status='COMPLETE'))
    b=compile_mechanism_atlas(a); m=build_compiled_network(b,hidden_dim=8,depth=1)
    x=torch.randn(6,len(b.input_variables),requires_grad=True); y=m(x)
    y[:,b.output_variables.index('y1')].sum().backward(retain_graph=True)
    ix3=b.input_variables.index('x3')
    assert torch.allclose(x.grad[:,ix3],torch.zeros_like(x.grad[:,ix3]))
    x.grad.zero_(); y[:,b.output_variables.index('y2')].sum().backward()
    ix1=b.input_variables.index('x1')
    assert torch.allclose(x.grad[:,ix1],torch.zeros_like(x.grad[:,ix1]))


def test_compiler_excludes_uncertified_and_keeps_uncovered_as_declaration():
    a=MechanismAtlas('demo')
    a.add_scope(MechanismScopeRecord('ok',['y'],observed_drivers=['x'],graph={'y':['x']},completion_status='COMPLETE'))
    a.add_scope(MechanismScopeRecord('bad',['z'],observed_drivers=['q'],graph={'z':['q']},completion_status='UNCERTIFIED'))
    a.add_uncovered(UncoveredRegion('gap',['u'],'NEW_SCOPE_REQUIRED','not covered'))
    b=compile_mechanism_atlas(a)
    assert [s.scope_id for s in b.scopes]==['ok']
    assert b.excluded_scopes[0]['scope_id']=='bad'
    assert b.uncovered_blocks[0]['active_residual_block'] is False


def test_certified_algebraic_coefficients_compile_to_fixed_operator():
    a=MechanismAtlas('alg')
    a.add_scope(MechanismScopeRecord('energy',['y'],scope_kind='algebraic',observed_drivers=['a','b'],graph={'y':['a','b']},
        completion_status='COMPLETE',metadata={'coefficients':{'a*b':2.0},'intercept':0.5}))
    b=compile_mechanism_atlas(a); m=build_compiled_network(b)
    assert m.policies['y']=='analytic_fixed'
    x=torch.zeros(2,len(b.input_variables)); x[:,b.input_variables.index('a')]=3; x[:,b.input_variables.index('b')]=4
    y=m(x)[:,b.output_variables.index('y')]
    assert torch.allclose(y,torch.tensor([24.5,24.5]))
    assert sum(p.numel() for p in m.targets['y'].parameters())==0


def test_certified_dynamic_canonical_payload_compiles_identity_and_smooth_stribeck_fixed():
    a=MechanismAtlas('dyn')
    a.add_scope(MechanismScopeRecord('m',['x','v'],observed_drivers=['p'],graph={'x':['v'],'v':['v','p']},completion_status='COMPLETE',
      metadata={'mechanism':{
        'coefficients':{'x':{'v':1.0},'v':{'v':-2.0,'p':3.0}},
        'operators':{'x':{'v':'Identity'},'v':{'v':'SmoothStribeck','p':'Identity'}},
        'operator_parameters':{'x':{'v':None},'v':{'v':{'smoothing_velocity':.1,'breakaway_velocity':1.0,'viscous_ratio':0.0,'stribeck_ratio':0.0},'p':None}},
      }}))
    b=compile_mechanism_atlas(a); m=build_compiled_network(b)
    assert m.policies=={'v':'analytic_fixed','x':'analytic_fixed'} or m.policies=={'x':'analytic_fixed','v':'analytic_fixed'}
    inp=torch.zeros(1,len(b.input_variables)); inp[:,b.input_variables.index('v')]=.2; inp[:,b.input_variables.index('p')]=.5
    out=m(inp); ox=out[:,b.output_variables.index('x')]; ov=out[:,b.output_variables.index('v')]
    assert torch.allclose(ox,torch.tensor([.2]),atol=1e-6)
    expected=-2.0*torch.tanh(torch.tensor(.2/.1))+3.0*.5
    assert torch.allclose(ov,expected.reshape(1),atol=1e-6)
    assert sum(p.numel() for p in m.parameters())==0


def test_certified_algebraic_scopes_chain_without_requiring_intermediate_as_external_input():
    a=MechanismAtlas('chain')
    a.add_scope(MechanismScopeRecord('s1',['y'],scope_kind='algebraic',observed_drivers=['a','b'],
        graph={'y':['a','b']},completion_status='COMPLETE',metadata={'coefficients':{'a*b':2.0}}))
    a.add_scope(MechanismScopeRecord('s2',['z'],scope_kind='algebraic',observed_drivers=['y'],
        graph={'z':['y']},completion_status='COMPLETE',metadata={'coefficients':{'y':0.5},'intercept':1.0}))
    b=compile_mechanism_atlas(a)
    assert b.input_variables==['a','b']
    assert b.metadata['execution_order'].index('y') < b.metadata['execution_order'].index('z')
    m=build_compiled_network(b)
    x=torch.tensor([[3.0,4.0]])
    out=m(x)
    assert torch.allclose(out[:,b.output_variables.index('y')],torch.tensor([24.0]))
    assert torch.allclose(out[:,b.output_variables.index('z')],torch.tensor([13.0]))


def test_algebraic_cycle_is_rejected_instead_of_hidden_in_neural_layer():
    a=MechanismAtlas('cycle')
    a.add_scope(MechanismScopeRecord('s1',['y'],scope_kind='algebraic',observed_drivers=['z'],graph={'y':['z']},completion_status='COMPLETE'))
    a.add_scope(MechanismScopeRecord('s2',['z'],scope_kind='algebraic',observed_drivers=['y'],graph={'z':['y']},completion_status='COMPLETE'))
    import pytest
    with pytest.raises(ValueError,match='cyclic algebraic'):
        compile_mechanism_atlas(a)
