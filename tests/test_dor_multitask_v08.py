from dor.multitask import arbitrate_multitask

def det(d='ABNORMAL',s='SUPPORTED'): return {'status':s,'diagnosis':d}
def loc(status='SUPPORTED',ids=('A__f',),mode='failed_off'): return {'status':status,'fault_hypotheses':[{'fault_id':x,'fault_mode':mode} for x in ids]}
def root(status='SUPPORTED',ids=('A__f',)): return {'status':status,'root_cause_set':list(ids) if status=='SUPPORTED' else [],'candidate_root_sets':[{'fault_ids':list(ids)}] if status=='ALTERNATIVES' else []}
def ident(status='SUPPORTED',ids=('A__f',),source='ROOT_CAUSE_SET',mode='failed_off'):
    return {'status':status,'candidate_source':source,'identifications':[{'fault_id':x,'status':'SUPPORTED' if status=='SUPPORTED' else status,'identified_change':{'mode':mode,'exact_mode_identified':True}} for x in ids]}

def test_detection_only_abnormal_is_supported_but_not_root():
    r=arbitrate_multitask(detection=det()); assert r['status']=='SUPPORTED' and r['diagnostic_depth']=='DETECTION' and not r['safety']['root_cause_claimed']

def test_detection_normal_with_fault_claim_abstains():
    r=arbitrate_multitask(detection=det('NORMAL'),localization=loc()); assert r['status']=='ABSTAIN' and r['contradictions']

def test_consistent_full_chain_reaches_identification():
    r=arbitrate_multitask(detection=det(),localization=loc(),root_cause=root(),identification=ident()); assert r['status']=='SUPPORTED' and r['diagnostic_depth']=='IDENTIFICATION' and r['safety']['root_cause_claimed']

def test_root_must_be_subset_of_localization():
    r=arbitrate_multitask(localization=loc(ids=('A__f',)),root_cause=root(ids=('B__f',))); assert r['status']=='ABSTAIN'

def test_identification_must_be_inside_root_or_localization_scope():
    r=arbitrate_multitask(localization=loc(),root_cause=root(),identification=ident(ids=('B__f',))); assert r['status']=='ABSTAIN'

def test_root_unknown_does_not_prevent_local_identification():
    r=arbitrate_multitask(localization=loc(),root_cause={'status':'UNKNOWN'},identification=ident(source='LOCALIZATION'))
    assert r['status']=='SUPPORTED' and r['diagnostic_depth']=='IDENTIFICATION' and not r['safety']['root_cause_claimed']

def test_identification_unknown_does_not_erase_supported_root():
    r=arbitrate_multitask(localization=loc(),root_cause=root(),identification={'status':'UNKNOWN','identifications':[]})
    assert r['status']=='SUPPORTED' and r['diagnostic_depth']=='ROOT_CAUSE'

def test_localization_alternatives_can_be_resolved_by_unique_root():
    r=arbitrate_multitask(localization=loc(status='ALTERNATIVES',ids=('A__f','B__f')),root_cause=root(ids=('A__f',)))
    assert r['status']=='SUPPORTED' and r['diagnostic_depth']=='ROOT_CAUSE'

def test_class_alternatives_are_preserved_without_mapping():
    r=arbitrate_multitask(detection=det(),classification={'status':'ALTERNATIVES','prediction_set':['F1','F2']})
    assert r['status']=='ALTERNATIVES'

def test_optional_class_abstention_does_not_veto_physical_root():
    r=arbitrate_multitask(classification={'status':'ABSTAIN'},localization=loc(),root_cause=root())
    assert r['status']=='SUPPORTED' and r['diagnostic_depth']=='ROOT_CAUSE'

def test_localization_abstention_cannot_be_bypassed():
    r=arbitrate_multitask(localization={'status':'ABSTAIN','fault_hypotheses':[]},root_cause=root())
    assert r['status']=='ABSTAIN'

def test_no_inputs_is_unknown():
    r=arbitrate_multitask(); assert r['status']=='UNKNOWN' and r['diagnostic_depth']=='NONE'

def test_exact_mode_conflict_abstains():
    r=arbitrate_multitask(localization=loc(mode='failed_off'),identification=ident(source='LOCALIZATION',mode='failed_open'))
    assert r['status']=='ABSTAIN'
