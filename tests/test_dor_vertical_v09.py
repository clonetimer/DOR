from dor.vertical import evaluate_vertical, validate_vertical_registry

def base():
    return {'target_domain':'SATELLITE','requirements':{'DETECTION':{'minimum_evidence_grade':'REAL_IN_ORBIT','require_target_domain':True},'CLASSIFICATION':{'minimum_evidence_grade':'HIGH_FIDELITY_SYSTEM_SIMULATION','require_target_domain':True}},'cases':[]}
def case(cid,task,grade,domain='SATELLITE',state='PASS'):
    return {'case_id':cid,'task':task,'evidence_grade':grade,'domain':domain,'state':state,'provenance':'p'}
def test_full_when_all_target_gates_met():
    r=base(); r['cases']=[case('a','DETECTION','REAL_IN_ORBIT'),case('b','CLASSIFICATION','HIGH_FIDELITY_SYSTEM_SIMULATION')]; assert evaluate_vertical(r)['status']=='FULLY_VALIDATED'
def test_cross_domain_hardware_does_not_satisfy_satellite_gate():
    r=base(); r['cases']=[case('a','DETECTION','REAL_IN_ORBIT'),case('b','CLASSIFICATION','REAL_HARDWARE_TESTBED','AEROSPACE_TESTBED')]; o=evaluate_vertical(r); assert o['status']=='PARTIALLY_VALIDATED' and 'CLASSIFICATION' in o['missing_gates']
def test_source_anchored_mechanism_cannot_upgrade_to_high_fidelity():
    r=base(); r['cases']=[case('a','DETECTION','REAL_IN_ORBIT'),case('b','CLASSIFICATION','SOURCE_ANCHORED_SYNTHETIC_MECHANISM')]; assert 'CLASSIFICATION' in evaluate_vertical(r)['missing_gates']
def test_partial_case_does_not_satisfy_gate():
    r=base(); r['cases']=[case('a','DETECTION','REAL_IN_ORBIT',state='PARTIAL')]; assert not evaluate_vertical(r)['task_summary']['DETECTION']['meets_requirement']
def test_invalid_grade_blocks():
    r=base(); r['cases']=[{'case_id':'x','task':'DETECTION','evidence_grade':'MAGIC','domain':'SATELLITE','state':'PASS','provenance':'p'}]; assert evaluate_vertical(r)['status']=='BLOCKED'
def test_missing_target_domain_invalid(): assert validate_vertical_registry({'cases':[]})
