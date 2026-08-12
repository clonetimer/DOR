"""v2.2: physics knowledge routing + representation eligibility, without loading heavy runtimes."""
from m2n import build_bias_route_plan, build_representation_plan

spec={
  'physics_biases':[
    {'id':'energy_core','knowledge_form':'EXACT_CONSERVATION','trust':'HIGH'},
    {'id':'degradation_trend','knowledge_form':'MONOTONICITY','trust':'MEDIUM'},
    {'id':'virtual_temperature','knowledge_form':'VIRTUAL_SENSOR','trust':'MEDIUM'},
  ],
  'template':'generic_mechanism_graph',
  'task_semantics':{
    'representation_kind':'CONSERVATIVE_HAMILTONIAN',
    'canonical_qp':True,
    'conservative_core':True,
    'dissipation_separate':True,
  },
  'knowledge':{'slots':[]},
}
print(build_bias_route_plan(spec).to_dict())
print([x.to_dict() for x in build_representation_plan(spec)])
assert build_representation_plan(spec)[0].eligibility.architecture_pattern=='HNN_STYLE'
print('BIAS_REPRESENTATION_POLICY_PASS')
