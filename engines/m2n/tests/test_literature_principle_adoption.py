import csv, json
from pathlib import Path
import numpy as np

from m2n.bias_routing import build_bias_route_plan
from m2n.representation_policy import assess_representation, build_representation_plan
from m2n.exact_physics import audit_exact_physics
from m2n.canonical_ir import Dim, VarSpec, CNode, CanonicalIR
from m2n.goal_design import run_goal_driven_design
from test_p1_p5_pipeline import graph_spec, write_data


def test_bias_routing_does_not_force_all_physics_into_architecture():
    spec={'physics_biases':[
        {'id':'mass_balance','knowledge_form':'EXACT_CONSERVATION','trust':'HIGH'},
        {'id':'degradation_trend','knowledge_form':'MONOTONICITY','trust':'MEDIUM'},
        {'id':'temperature_virtual','knowledge_form':'VIRTUAL_SENSOR','trust':'MEDIUM'},
        {'id':'uncertain_equation','knowledge_form':'EXACT_EQUATION','trust':'LOW'},
    ]}
    p=build_bias_route_plan(spec); routes={x.item_id:x.route for x in p.decisions}
    assert routes['mass_balance']=='INDUCTIVE_HARD'
    assert routes['degradation_trend']=='LEARNING_SOFT'
    assert routes['temperature_virtual']=='OBSERVATIONAL'
    assert routes['uncertain_equation']=='LEARNING_SOFT'


def test_representation_eligibility_routes_by_mathematical_object():
    assert assess_representation('LOCAL_FUNCTION',{}).status=='ELIGIBLE'
    assert assess_representation('CONSERVATIVE_HAMILTONIAN',{'canonical_qp':True,'conservative_core':True,'dissipation_separate':True}).status=='ELIGIBLE'
    assert assess_representation('CONSERVATIVE_HAMILTONIAN',{'canonical_qp':True,'conservative_core':True}).status=='INELIGIBLE'
    assert assess_representation('FUNCTION_TO_FUNCTION_OPERATOR',{'multiple_input_functions':True,'consistent_input_sensors':True,'output_query_coordinate':True}).architecture_pattern=='DEEPONET_STYLE'
    assert assess_representation('REGULAR_GRID_FIELD_OPERATOR',{'field_operator':True,'regular_grid':True,'multiple_instances':True}).architecture_pattern=='FNO_STYLE'


def test_exact_physics_contract_distinguishes_compilation_from_truth():
    D=Dim()
    ir=CanonicalIR((VarSpec('x',D),),(CNode('y','sqrt_pos',('x',),{'sqrt_eps':1e-8}),),('y',),(('y',D),))
    c=audit_exact_physics(ir)
    assert c.physical_truth_claim=='NOT_ESTABLISHED_BY_COMPILATION'
    assert 'y' in c.totalized_or_guarded_node_ids
    assert c.compiled_semantics_exactness.startswith('WITH_RESPECT')


def test_specialized_k3_representation_blocks_native_local_mlp(tmp_path):
    spec=graph_spec(True)
    spec['schema_version']='2.1'
    spec['task_semantics']={'representation_kind':'DIFFERENTIAL_DYNAMICS','continuous_time':True,'solver_compatible':True}
    sp=tmp_path/'s.json'; sp.write_text(json.dumps(spec)); dp=tmp_path/'d.csv'; write_data(dp,True)
    out=tmp_path/'o'; r=run_goal_driven_design(sp,dp,out)
    d=json.loads((out/'design_result.json').read_text())
    assert r.status=='AMBIGUOUS'
    assert d['k3_gate']['status']=='SPECIALIZED_REPRESENTATION_REQUIRED'
    assert d['k3_representation']['eligibility']['architecture_pattern']=='UDE_STYLE'
    assert not list((out/'models').glob('*.pt'))


def test_minimal_augmentation_policy_emits_credit_assignment_audit(tmp_path):
    spec=graph_spec(True); spec['schema_version']='2.1'
    spec['k3_policy']['minimal_augmentation']={'enabled':True,'epochs':35,'fit_slack':0.10,'constraint_weight':150.0,'credit_assignment_cv_warn':0.4}
    sp=tmp_path/'s.json'; sp.write_text(json.dumps(spec)); dp=tmp_path/'d.csv'; write_data(dp,True)
    out=tmp_path/'o'; r=run_goal_driven_design(sp,dp,out); d=json.loads((out/'design_result.json').read_text())
    assert 'k3_training_policy' in d and d['k3_training_policy']['minimal_augmentation_enabled'] is True
    assert 'k3_credit_assignment_audit' in d
    assert d['k3_credit_assignment_audit']['augmentation_nrmse_median'] is not None
    assert (out/'bias_route_plan.json').is_file()
    assert (out/'representation_plan.json').is_file()
    assert (out/'exact_physics_contract.json').is_file()
