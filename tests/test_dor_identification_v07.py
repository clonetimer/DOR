from dor.identification import identify_fault_changes, validate_isgn_identification_evidence


def ont(category='DISCRETE', mode='failed_off'):
    return {
      'contract':'DOR-FAULT-ONTOLOGY/0.5','ontology_id':'t','domain':'test',
      'components':[{'component_id':'A','kind':'PHYSICAL'}], 'sensors':[], 'commands':[], 'topology_edges':[],
      'faults':[{'fault_id':'A__f','target_component_id':'A','fault_mode':mode,'category':category,'exclusion_group':'A__g','expected_evidence':[{'scope_type':'COMPONENT','scope_id':'A','expected_state':'FAILED_OFF' if mode=='failed_off' else 'ABNORMAL','weight':1.0}]}],
      'provenance':{'sources':['unit']}
    }

def loc():
    return {'contract':'DOR-LOCALIZATION-RESULT/0.5','status':'SUPPORTED','fault_hypotheses':[{'fault_id':'A__f','matched_evidence':['e1']} ]}

def root(status='SUPPORTED'):
    return {'contract':'DOR-ROOT-CAUSE-RESULT/0.6','status':status,'root_cause_set':['A__f'] if status=='SUPPORTED' else []}

def evidence(): return [{'evidence_id':'e1','validity':'VALID'}]

def isgn(status='SUPPORTED', cert='COMPLETE', validity='VALID', changes=None):
    return {'contract':'DOR-ISGN-IDENTIFICATION-EVIDENCE/0.7','status':status,'baseline_certificate_status':cert,'window_evidence_validity':validity,'changes':changes if changes is not None else [{'component_id':'A','change_type':'PARAMETER_DRIFT','parameter_name':'k','baseline_value':1.0,'delta':0.2,'relative_delta':0.2,'identifiable':True,'supported':True,'evidence_id':'isgn:k','provenance':'ISGN anchored drift'}]}

def test_discrete_mode_does_not_require_isgn():
    r=identify_fault_changes(ont(),loc(),evidence(),root_cause=root())
    assert r['status']=='SUPPORTED' and r['isgn_admission']['required'] is False
    assert r['identifications'][0]['identified_change']['mode']=='failed_off'

def test_generic_state_fault_does_not_become_exact_mode():
    r=identify_fault_changes(ont(mode='state_fault'),loc(),evidence(),root_cause=root())
    assert r['status']=='UNKNOWN'
    assert r['identifications'][0]['identified_change']['exact_mode_identified'] is False

def test_parametric_requires_isgn():
    r=identify_fault_changes(ont(category='PARAMETRIC',mode='parameter_drift'),loc(),evidence(),root_cause=root())
    assert r['status']=='UNKNOWN' and r['isgn_admission']['required'] is True

def test_parametric_supported_only_with_complete_valid_identifiable_isgn():
    r=identify_fault_changes(ont(category='PARAMETRIC',mode='parameter_drift'),loc(),evidence(),root_cause=root(),isgn_identification=isgn())
    assert r['status']=='SUPPORTED'
    assert r['identifications'][0]['basis']=='ISGN_CERTIFIED_DIFFERENTIAL'

def test_invalid_isgn_certificate_forces_abstain():
    r=identify_fault_changes(ont(category='PARAMETRIC',mode='parameter_drift'),loc(),evidence(),root_cause=root(),isgn_identification=isgn(cert='UNCERTIFIED'))
    assert r['status']=='ABSTAIN'

def test_invalid_window_evidence_forces_abstain():
    r=identify_fault_changes(ont(category='PARAMETRIC',mode='parameter_drift'),loc(),evidence(),root_cause=root(),isgn_identification=isgn(validity='REFERENCE_UNSUPPORTED'))
    assert r['status']=='ABSTAIN'

def test_unidentifiable_change_remains_unknown():
    ch=[{'component_id':'A','change_type':'PARAMETER_DRIFT','parameter_name':'k','identifiable':False,'supported':False,'evidence_id':'x','provenance':'test'}]
    r=identify_fault_changes(ont(category='PARAMETRIC',mode='parameter_drift'),loc(),evidence(),root_cause=root(),isgn_identification=isgn(status='UNIDENTIFIABLE',changes=ch))
    assert r['status']=='UNKNOWN'

def test_multiple_identifiable_changes_preserve_alternatives():
    ch=[{'component_id':'A','change_type':'PARAMETER_DRIFT','parameter_name':'k1','identifiable':True,'supported':True,'evidence_id':'x1','provenance':'test'}, {'component_id':'A','change_type':'PARAMETER_DRIFT','parameter_name':'k2','identifiable':True,'supported':True,'evidence_id':'x2','provenance':'test'}]
    r=identify_fault_changes(ont(category='PARAMETRIC',mode='parameter_drift'),loc(),evidence(),root_cause=root(),isgn_identification=isgn(status='ALTERNATIVES',changes=ch))
    assert r['status']=='ALTERNATIVES'
    assert len(r['identifications'][0]['identified_change']['alternatives'])==2

def test_root_unknown_does_not_block_local_fault_identification_or_imply_root():
    r=identify_fault_changes(ont(),loc(),evidence(),root_cause={'status':'UNKNOWN'})
    assert r['status']=='SUPPORTED' and r['candidate_source']=='LOCALIZATION'
    assert r['safety']['root_cause_implied'] is False

def test_isgn_validator_rejects_missing_provenance():
    p=isgn(); p['changes'][0]['provenance']=''
    ok,errors=validate_isgn_identification_evidence(p)
    assert not ok and errors

def test_identification_plan_skips_isgn_for_discrete():
    from dor.identification import build_fault_identification_request
    r=build_fault_identification_request(ont(),loc(),root_cause=root(),case_id='x')
    assert r['status']=='PLANNED' and r['engine_admissions']['ISGN']=='SKIP'


def test_identification_plan_requires_valid_window_for_parametric():
    from dor.identification import build_fault_identification_request
    r=build_fault_identification_request(ont(category='PARAMETRIC',mode='parameter_drift'),loc(),root_cause=root(),case_id='x')
    assert r['status']=='BLOCKED' and r['engine_admissions']['ISGN']=='BLOCKED'
    r2=build_fault_identification_request(ont(category='PARAMETRIC',mode='parameter_drift'),loc(),root_cause=root(),case_id='x',window_start=10,window_end=20,window_evidence_validity='VALID')
    assert r2['status']=='PLANNED' and r2['engine_admissions']['ISGN']=='RUN'


def test_discrete_identification_abstains_if_matched_evidence_is_not_valid():
    bad=[{'evidence_id':'e1','validity':'CONTEXT_UNSUPPORTED'}]
    r=identify_fault_changes(ont(),loc(),bad,root_cause=root())
    assert r['status']=='ABSTAIN'
