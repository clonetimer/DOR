from dor.freeze import evaluate_freeze_gate, REQUIRED_CORE_GATES

def ev():
 return {'gates':[{'gate':g,'status':'PASS','reason':'ok','evidence':['x']} for g in REQUIRED_CORE_GATES if g!='SATELLITE_VERTICAL']}
def fullv(): return {'contract':'DOR-VERTICAL-VALIDATION/0.9','status':'FULLY_VALIDATED','missing_gates':[]}
def partial(): return {'contract':'DOR-VERTICAL-VALIDATION/0.9','status':'PARTIALLY_VALIDATED','missing_gates':['CLASSIFICATION']}
def test_ready_only_when_all_core_and_vertical_pass(): assert evaluate_freeze_gate(ev(),fullv())['status']=='READY_FOR_1_0_FREEZE'
def test_partial_vertical_blocks():
 o=evaluate_freeze_gate(ev(),partial()); assert o['status']=='BLOCKED' and 'SATELLITE_VERTICAL' in o['blocking_gates']
def test_missing_core_gate_blocks():
 e=ev(); e['gates']=[x for x in e['gates'] if x['gate']!='ROOT_CAUSE_REAL_HARDWARE']; o=evaluate_freeze_gate(e,fullv()); assert 'ROOT_CAUSE_REAL_HARDWARE' in o['blocking_gates']
def test_failed_core_gate_blocks():
 e=ev(); e['gates'][0]['status']='BLOCKED'; assert evaluate_freeze_gate(e,fullv())['status']=='BLOCKED'
