from __future__ import annotations

from pathlib import Path
import pandas as pd

from dor.ontology import validate_fault_ontology, validate_mechanism_component_map
from dor.localization import localize_faults, localize_from_files


def _ontology():
    return {
        'contract':'DOR-FAULT-ONTOLOGY/0.5','ontology_id':'unit','domain':'test',
        'components':[
            {'component_id':'R1','component_type':'relay','kind':'ACTUATOR','subsystem':'eps'},
            {'component_id':'S1_HEALTH','component_type':'position_sensor','kind':'SENSOR','subsystem':'eps'},
        ],
        'sensors':[{'sensor_id':'S1','sensor_type':'position_feedback','health_component_id':'S1_HEALTH','observes_components':['R1']}],
        'commands':[{'command_id':'R1_CMD','target_component_id':'R1'}],
        'topology_edges':[{'source':'R1','target':'S1_HEALTH','relation':'FEEDBACK'}],
        'faults':[
            {'fault_id':'R1_STUCK_OPEN','target_component_id':'R1','fault_mode':'stuck_open','category':'DISCRETE','expected_evidence':[
                {'scope_type':'COMMAND','scope_id':'R1_CMD','expected_state':'CLOSED','weight':0.5},
                {'scope_type':'SENSOR','scope_id':'S1','expected_state':'OPEN','weight':1.0},
            ],'contradictions':[{'scope_type':'SENSOR','scope_id':'S1','state':'CLOSED','weight':1.0}]},
            {'fault_id':'R1_STUCK_CLOSED','target_component_id':'R1','fault_mode':'stuck_closed','category':'DISCRETE','expected_evidence':[
                {'scope_type':'COMMAND','scope_id':'R1_CMD','expected_state':'OPEN','weight':0.5},
                {'scope_type':'SENSOR','scope_id':'S1','expected_state':'CLOSED','weight':1.0},
            ],'contradictions':[{'scope_type':'SENSOR','scope_id':'S1','state':'OPEN','weight':1.0}]},
        ],
        'provenance':{'sources':['unit']},
    }


def _ev(*rows):
    out=[]
    for i,(st,sid,state,strength,validity) in enumerate(rows):
        out.append({'evidence_id':f'e{i}','scope_type':st,'scope_id':sid,'state':state,'strength':strength,'validity':validity})
    return out


def test_ontology_rejects_unknown_sensor_reference():
    o=_ontology(); o['faults'][0]['expected_evidence'][1]['scope_id']='MISSING'
    v=validate_fault_ontology(o)
    assert v.state=='FAIL'
    assert any('unknown sensor' in e for e in v.errors)


def test_relay_stuck_open_is_localized_without_root_cause_claim():
    r=localize_faults(_ontology(),_ev(('COMMAND','R1_CMD','CLOSED',1.0,'VALID'),('SENSOR','S1','OPEN',1.0,'VALID')))
    assert r['status']=='SUPPORTED'
    assert r['fault_hypotheses'][0]['fault_id']=='R1_STUCK_OPEN'
    assert r['component_candidates'][0]['component_id']=='R1'
    assert r['safety']['root_cause_claimed'] is False
    assert r['safety']['propagation_inferred'] is False


def test_invalid_evidence_abstains():
    r=localize_faults(_ontology(),_ev(('SENSOR','S1','OPEN',1.0,'CONTEXT_UNSUPPORTED')))
    assert r['status']=='ABSTAIN'


def test_generic_component_abnormality_does_not_identify_fault_mode():
    r=localize_faults(_ontology(),_ev(('COMPONENT','R1','ABNORMAL',1.0,'VALID')))
    assert r['status']=='UNKNOWN'
    assert r['safety']['generic_component_or_mechanism_evidence_cannot_by_itself_identify_fault_mode'] is True


def test_indistinguishable_declared_faults_preserve_alternatives():
    o=_ontology()
    o['components'].append({'component_id':'R2','component_type':'relay','kind':'ACTUATOR','subsystem':'eps'})
    o['faults']=[
        {'fault_id':'F1','target_component_id':'R1','fault_mode':'unknown','category':'COMPONENT','exclusion_group':'g','expected_evidence':[{'scope_type':'SENSOR','scope_id':'S1','expected_state':'ABNORMAL','weight':1.0}]},
        {'fault_id':'F2','target_component_id':'R2','fault_mode':'unknown','category':'COMPONENT','exclusion_group':'g','expected_evidence':[{'scope_type':'SENSOR','scope_id':'S1','expected_state':'ABNORMAL','weight':1.0}]},
    ]
    r=localize_faults(o,_ev(('SENSOR','S1','HIGH',0.9,'VALID')))
    assert r['status']=='ALTERNATIVES'
    assert {x['fault_id'] for x in r['fault_hypotheses']}=={'F1','F2'}


def test_mechanism_map_is_reference_checked_and_mechanism_only_is_scope_evidence():
    o=_ontology()
    m={'contract':'DOR-MECHANISM-COMPONENT-MAP/0.5','map_id':'m','bindings':[{'mechanism_id':'relay_response','component_ids':['R1'],'sensor_ids':['S1'],'provenance':'unit'}]}
    assert validate_mechanism_component_map(m,o).state=='PASS'
    r=localize_faults(o,_ev(('MECHANISM','relay_response','ABNORMAL',1.0,'VALID')),mechanism_map=m)
    assert r['status']=='UNKNOWN'


def test_file_runtime(tmp_path: Path):
    import json
    o=_ontology(); op=tmp_path/'o.json'; op.write_text(json.dumps(o))
    ep=tmp_path/'e.csv'; pd.DataFrame(_ev(('COMMAND','R1_CMD','CLOSED',1.0,'VALID'),('SENSOR','S1','OPEN',1.0,'VALID'))).to_csv(ep,index=False)
    r=localize_from_files(op,ep)
    assert r['status']=='SUPPORTED'

def test_class_evidence_maps_calibrated_fault_class_to_declared_fault():
    o=_ontology()
    o['faults']=[
        {'fault_id':'R1_CLASS_A','target_component_id':'R1','fault_mode':'class_a','category':'COMPONENT','expected_evidence':[{'scope_type':'COMPONENT','scope_id':'R1','expected_state':'CLASS','expected_value':'A','weight':1.0}]},
        {'fault_id':'R1_CLASS_B','target_component_id':'R1','fault_mode':'class_b','category':'COMPONENT','expected_evidence':[{'scope_type':'COMPONENT','scope_id':'R1','expected_state':'CLASS','expected_value':'B','weight':1.0}]},
    ]
    rows=[{'evidence_id':'c','scope_type':'COMPONENT','scope_id':'R1','state':'CLASS','value':'B','strength':1.0,'validity':'VALID'}]
    r=localize_faults(o,rows)
    assert r['status']=='SUPPORTED'
    assert r['fault_hypotheses'][0]['fault_id']=='R1_CLASS_B'


def test_concurrent_faults_on_independent_groups_are_supported_together():
    o=_ontology()
    o['components'].append({'component_id':'R2','component_type':'load','kind':'PHYSICAL','subsystem':'eps'})
    o['faults']=[
        {'fault_id':'F1','target_component_id':'R1','fault_mode':'a','category':'COMPONENT','exclusion_group':'R1_mode','expected_evidence':[{'scope_type':'SENSOR','scope_id':'S1','expected_state':'OPEN','weight':1.0}]},
        {'fault_id':'F2','target_component_id':'R2','fault_mode':'b','category':'COMPONENT','exclusion_group':'R2_mode','expected_evidence':[{'scope_type':'COMPONENT','scope_id':'R2','expected_state':'CLASS','expected_value':'B','weight':1.0}]},
    ]
    rows=[
        {'evidence_id':'e1','scope_type':'SENSOR','scope_id':'S1','state':'OPEN','strength':1.0,'validity':'VALID'},
        {'evidence_id':'e2','scope_type':'COMPONENT','scope_id':'R2','state':'CLASS','value':'B','strength':1.0,'validity':'VALID'},
    ]
    r=localize_faults(o,rows)
    assert r['status']=='SUPPORTED'
    assert set(r['supported_fault_set'])=={'F1','F2'}
    assert r['safety']['concurrent_faults_are_not_relabelled_as_alternatives'] is True


def test_abstain_result_keeps_full_contract_safety_and_policy():
    r=localize_faults(_ontology(),_ev(('SENSOR','S1','OPEN',1.0,'CONTEXT_UNSUPPORTED')))
    assert r['status']=='ABSTAIN'
    assert r['policy']['candidate_set_cap']==32
    assert r['safety']['concurrent_faults_are_not_relabelled_as_alternatives'] is True
    assert r['candidate_fault_sets']==[]


def test_unknown_result_keeps_full_contract_safety_and_policy():
    r=localize_faults(_ontology(),_ev(('COMPONENT','R1','ABNORMAL',1.0,'VALID')))
    assert r['status']=='UNKNOWN'
    assert r['policy']['candidate_set_cap']==32
    assert r['safety']['concurrent_faults_are_not_relabelled_as_alternatives'] is True
