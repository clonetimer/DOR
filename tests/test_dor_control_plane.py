from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dor.arbiter import arbitrate_anomaly_portfolio
from dor.control import build_plan, formalize_manifest, make_effective_m2n_request, qualify_task
from dor.reference import build_reference_applicability, normalize_context_levels, select_hierarchical_fec_output


def _request(task: str, *, label='fault_label', normal='0'):
    r={
        'contract':'M2N-DIAGNOSTIC-MODEL-REQUEST/1.0',
        'task':task,
        'data':{
            'split_column':'split','train_values':['train'],'validation_values':['val'],
            'calibration_values':['cal'],'test_values':['test'],'id_columns':['sample_id']
        },
        'features':{'raw_columns':['x'],'mechanism_columns':[],'views':['RAW']},
        'supervision':{'label_column':label},
    }
    if task=='ANOMALY_DETECTION': r['supervision']['normal_label']=normal
    return r


def _manifest(task: str, path: Path, sha: str, *, context=None, group=None, normal='0'):
    return {
        'contract':'DOR-DATASET-MANIFEST/0.4','dataset_id':'unit','domain':'test','task':task,
        'data':{'path':str(path),'sha256':sha,'row_count':None},
        'identifiers':['sample_id'],
        'split':{'column':'split','train_values':['train'],'validation_values':['val'],'calibration_values':['cal'],'test_values':['test']},
        'supervision':{'label_column':'fault_label','normal_label':normal,'class_frame':['0','1']},
        'context':{'columns':context or [],'policy':'EXACT'},
        'calibration':{'alpha':0.1,'min_group_size':2,'missing_context_policy':'ABSTAIN'},
        'leakage_guard':{'group_columns':group or ['sample_id']},
        'mechanism':{'status':'DEFERRED','reason':'unit test'},
        'provenance':{},
    }


def _sha(path: Path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_effective_request_preserves_declared_context(tmp_path: Path):
    req=_request('ANOMALY_DETECTION')
    p=tmp_path/'d.csv'; pd.DataFrame([{'sample_id':'a','split':'train','fault_label':'0','ctx':'A','x':1}]).to_csv(p,index=False)
    m=_manifest('ANOMALY_DETECTION',p,_sha(p),context=['ctx'])
    eff=make_effective_m2n_request(req,m)
    assert eff['data']['id_columns']==['sample_id','ctx']


def test_anomaly_context_shortfall_is_conditional_not_silently_pooled(tmp_path: Path):
    rows=[]
    for i in range(4): rows.append({'sample_id':f'tr{i}','split':'train','fault_label':'0','ctx':'A','x':i})
    rows += [
        {'sample_id':'v0','split':'val','fault_label':'0','ctx':'A','x':1},
        {'sample_id':'v1','split':'val','fault_label':'1','ctx':'A','x':9},
        {'sample_id':'c0','split':'cal','fault_label':'0','ctx':'A','x':1},
        {'sample_id':'c1','split':'cal','fault_label':'0','ctx':'A','x':2},
        {'sample_id':'t0','split':'test','fault_label':'0','ctx':'A','x':1},
        {'sample_id':'t1','split':'test','fault_label':'1','ctx':'B','x':9},
    ]
    p=tmp_path/'d.csv'; pd.DataFrame(rows).to_csv(p,index=False)
    m=_manifest('ANOMALY_DETECTION',p,_sha(p),context=['ctx'])
    q=qualify_task(m,_request('ANOMALY_DETECTION'),p)
    assert q.status=='CONDITIONALLY_QUALIFIED'
    c=q.checks['reference_applicability']
    assert c['unsupported_rows']==1
    assert c['status_counts']['CONTEXT_UNSUPPORTED']==1
    assert any('ABSTAIN' in x for x in q.required_next_actions) is False  # action is about collection/hierarchy, not outcome invention
    assert any('do not discover pooling' in x for x in q.required_next_actions)


def test_leakage_guard_blocks_cross_role_group(tmp_path: Path):
    rows=[
        {'sample_id':'a','run':'R1','split':'train','fault_label':'A','x':1},
        {'sample_id':'b','run':'R2','split':'train','fault_label':'B','x':2},
        {'sample_id':'c','run':'R3','split':'val','fault_label':'A','x':1},
        {'sample_id':'d','run':'R4','split':'val','fault_label':'B','x':2},
        {'sample_id':'e','run':'R5','split':'cal','fault_label':'A','x':1},
        {'sample_id':'f','run':'R6','split':'cal','fault_label':'B','x':2},
        {'sample_id':'g','run':'R1','split':'test','fault_label':'A','x':1},
        {'sample_id':'h','run':'R7','split':'test','fault_label':'B','x':2},
    ]
    p=tmp_path/'d.csv'; pd.DataFrame(rows).to_csv(p,index=False)
    m=_manifest('FAULT_CLASSIFICATION',p,_sha(p),group=['run'],normal='A')
    m['supervision']['class_frame']=['A','B']
    req=_request('FAULT_CLASSIFICATION'); req['supervision']={'label_column':'fault_label'}
    q=qualify_task(m,req,p)
    assert q.status=='UNSUPPORTED'
    assert q.checks['leakage_guard']['cross_role_group_count']==1


def test_build_plan_skips_uncertified_isgn_but_runs_other_engines(tmp_path: Path):
    rows=[]
    roles=['train']*4+['val']*2+['cal']*2+['test']*2
    labels=['A','B','A','B','A','B','A','B','A','B']
    for i,(r,y) in enumerate(zip(roles,labels)):
        rows.append({'sample_id':str(i),'split':r,'fault_label':y,'x':i})
    p=tmp_path/'d.csv'; pd.DataFrame(rows).to_csv(p,index=False)
    m=_manifest('FAULT_CLASSIFICATION',p,_sha(p),normal='A')
    m['supervision']['class_frame']=['A','B']; m['calibration']['min_group_size']=1
    req=_request('FAULT_CLASSIFICATION'); req['supervision']={'label_column':'fault_label'}
    plan=build_plan(m,req,p,tmp_path/'out')
    admissions={x.engine:x.status for x in plan.engine_admissions}
    assert admissions=={'ISGN':'SKIP','M2N':'RUN','FEC':'RUN','DOR':'RUN'}
    assert [s.step_id for s in plan.steps]==['S00','S05','S10','S20','S30','S40','S45','S50','S60']


def test_formalize_old_opssat_style_manifest_does_not_use_test_labels_to_discover_context(tmp_path: Path):
    p=tmp_path/'d.csv'; pd.DataFrame([{'sample_id':'a','channel':'C','split':'train','fault_label':'0','x':1}]).to_csv(p,index=False)
    old={'contract':'DOR-DATASET-MANIFEST-PROTOTYPE/0.3','dataset':'OPSSAT-AD v2','channel_role_plan':{'C':{}},'derived_sha256':_sha(p)}
    req=_request('ANOMALY_DETECTION'); req['data']['id_columns']=['sample_id','channel']
    m=formalize_manifest(old,req,p)
    assert m['contract']=='DOR-DATASET-MANIFEST/0.4'
    assert m['context']['columns']==['channel']
    assert m['calibration']['missing_context_policy']=='ABSTAIN'


def test_anomaly_portfolio_disagreement_is_alternatives(tmp_path: Path):
    a=tmp_path/'a.csv'; b=tmp_path/'b.csv'
    pd.DataFrame([{'sample_id':'x','decision':'NORMAL','calibration_supported':True}]).to_csv(a,index=False)
    pd.DataFrame([{'sample_id':'x','decision':'ABNORMAL','calibration_supported':True}]).to_csv(b,index=False)
    out=tmp_path/'o.csv'; summ=tmp_path/'s.json'
    arbitrate_anomaly_portfolio([a,b],out,summ,id_column='sample_id')
    d=pd.read_csv(out)
    assert d.loc[0,'diagnosis_status']=='ALTERNATIVES'
    assert set(json.loads(d.loc[0,'diagnosis_set']))=={'NORMAL','ABNORMAL'}

def test_requested_unbound_mechanism_blocks_downstream_engines(tmp_path: Path):
    rows=[]
    roles=['train']*4+['val']*2+['cal']*2+['test']*2
    labels=['A','B','A','B','A','B','A','B','A','B']
    for i,(r,y) in enumerate(zip(roles,labels)):
        rows.append({'sample_id':str(i),'split':r,'fault_label':y,'x':i})
    p=tmp_path/'d.csv'; pd.DataFrame(rows).to_csv(p,index=False)
    m=_manifest('FAULT_CLASSIFICATION',p,_sha(p),normal='A')
    m['supervision']['class_frame']=['A','B']; m['calibration']['min_group_size']=1
    m['mechanism']={'status':'REQUESTED','reason':'unit requires discovery'}
    req=_request('FAULT_CLASSIFICATION'); req['supervision']={'label_column':'fault_label'}
    plan=build_plan(m,req,p,tmp_path/'out')
    admissions={x.engine:x.status for x in plan.engine_admissions}
    assert admissions['ISGN']=='BLOCKED'
    assert admissions['M2N']=='BLOCKED'
    assert admissions['FEC']=='BLOCKED'
    assert admissions['DOR']=='BLOCKED'


def test_declared_hierarchy_uses_parent_only_when_exact_reference_missing(tmp_path: Path):
    rows=[]
    # Calibration: exact A|P supported; B|P unseen. Parent P has 4 rows.
    for i in range(4): rows.append({'sample_id':f'tr{i}','split':'train','fault_label':'A' if i%2==0 else 'B','fine':'A','parent':'P','x':i})
    rows += [
        {'sample_id':'v0','split':'val','fault_label':'A','fine':'A','parent':'P','x':1},
        {'sample_id':'v1','split':'val','fault_label':'B','fine':'B','parent':'P','x':2},
        {'sample_id':'c0','split':'cal','fault_label':'A','fine':'A','parent':'P','x':1},
        {'sample_id':'c1','split':'cal','fault_label':'B','fine':'A','parent':'P','x':2},
        {'sample_id':'c2','split':'cal','fault_label':'A','fine':'A','parent':'P','x':3},
        {'sample_id':'c3','split':'cal','fault_label':'B','fine':'A','parent':'P','x':4},
        {'sample_id':'t0','split':'test','fault_label':'A','fine':'A','parent':'P','x':1},
        {'sample_id':'t1','split':'test','fault_label':'B','fine':'B','parent':'P','x':2},
    ]
    p=tmp_path/'d.csv'; pd.DataFrame(rows).to_csv(p,index=False)
    m=_manifest('FAULT_CLASSIFICATION',p,_sha(p),context=['fine','parent'],normal='A')
    m['supervision']['class_frame']=['A','B']; m['calibration']['min_group_size']=2
    m['context']={
        'columns':['fine','parent'],'policy':'HIERARCHICAL','fallback_policy':'DECLARED_ANCESTOR_ONLY',
        'levels':[
            {'level_id':'exact','columns':['fine','parent'],'min_group_size':2,'fallback_allowed':False,'sharing_basis':'exact'},
            {'level_id':'parent','columns':['parent'],'min_group_size':4,'fallback_allowed':True,'sharing_basis':'declared operating-regime equivalence hypothesis'},
        ],
    }
    app=build_reference_applicability(m,p,id_column='sample_id')
    by_id={r['row_id']:r for r in app['test_rows']}
    assert by_id['t0']['status']=='EXACT_SUPPORTED'
    assert by_id['t0']['selected_level']=='exact'
    assert by_id['t1']['status']=='HIERARCHY_SUPPORTED'
    assert by_id['t1']['selected_level']=='parent'
    assert app['safety']['test_labels_used'] is False
    assert app['safety']['implicit_pooling'] is False


def test_sparse_seen_context_is_power_limited_without_declared_parent(tmp_path: Path):
    rows=[]
    for i in range(4): rows.append({'sample_id':f'tr{i}','split':'train','fault_label':'0','ctx':'A','x':i})
    rows += [
        {'sample_id':'v0','split':'val','fault_label':'0','ctx':'A','x':1},
        {'sample_id':'v1','split':'val','fault_label':'1','ctx':'A','x':9},
        {'sample_id':'c0','split':'cal','fault_label':'0','ctx':'B','x':1},
        {'sample_id':'c1','split':'cal','fault_label':'0','ctx':'A','x':2},
        {'sample_id':'c2','split':'cal','fault_label':'0','ctx':'A','x':3},
        {'sample_id':'t0','split':'test','fault_label':'0','ctx':'B','x':1},
        {'sample_id':'t1','split':'test','fault_label':'1','ctx':'C','x':9},
    ]
    p=tmp_path/'d.csv'; pd.DataFrame(rows).to_csv(p,index=False)
    m=_manifest('ANOMALY_DETECTION',p,_sha(p),context=['ctx']); m['calibration']['min_group_size']=2
    app=build_reference_applicability(m,p,id_column='sample_id')
    by_id={r['row_id']:r for r in app['test_rows']}
    assert by_id['t0']['status']=='POWER_LIMITED'
    assert by_id['t1']['status']=='CONTEXT_UNSUPPORTED'
    assert app['summary']['status_counts']['HIERARCHY_SUPPORTED']==0


def test_coarser_level_must_be_nested_and_justified(tmp_path: Path):
    p=tmp_path/'d.csv'; pd.DataFrame([{'sample_id':'a','split':'train','fault_label':'0','a':'1','b':'2','x':1}]).to_csv(p,index=False)
    m=_manifest('ANOMALY_DETECTION',p,_sha(p),context=['a','b'])
    m['context']={'columns':['a','b'],'policy':'HIERARCHICAL','levels':[
        {'level_id':'exact','columns':['a','b'],'min_group_size':2},
        {'level_id':'bad','columns':['b'],'min_group_size':2,'fallback_allowed':True},
    ]}
    try:
        normalize_context_levels(m)
        assert False, 'expected missing sharing basis to be rejected'
    except ValueError as e:
        assert 'sharing_basis' in str(e)


def test_select_hierarchical_fec_output_uses_declared_level_per_row(tmp_path: Path):
    exact=tmp_path/'exact.csv'; parent=tmp_path/'parent.csv'
    pd.DataFrame([
        {'sample_id':'t0','prediction_set':'["A"]','prediction_set_size':1,'calibration_supported':True,'true_label':'A'},
        {'sample_id':'t1','prediction_set':'[]','prediction_set_size':0,'calibration_supported':False,'true_label':'B'},
    ]).to_csv(exact,index=False)
    pd.DataFrame([
        {'sample_id':'t0','prediction_set':'["A", "B"]','prediction_set_size':2,'calibration_supported':True,'true_label':'A'},
        {'sample_id':'t1','prediction_set':'["B"]','prediction_set_size':1,'calibration_supported':True,'true_label':'B'},
    ]).to_csv(parent,index=False)
    app={
        'contract':'DOR-REFERENCE-APPLICABILITY/0.4','id_column':'sample_id','test_rows':[
            {'row_id':'t0','status':'EXACT_SUPPORTED','selected_level':'exact','selected_columns':['fine'],'selected_key':['A'],'n_reference':10,'sharing_basis':'exact','reason':'exact'},
            {'row_id':'t1','status':'HIERARCHY_SUPPORTED','selected_level':'parent','selected_columns':['parent'],'selected_key':['P'],'n_reference':20,'sharing_basis':'declared parent','reason':'fallback'},
        ]
    }
    out=tmp_path/'selected.csv'; select_hierarchical_fec_output({'exact':exact,'parent':parent},app,out,id_column='sample_id')
    d=pd.read_csv(out,dtype={'sample_id':str})
    assert d.loc[d.sample_id=='t0','prediction_set'].iloc[0]=='["A"]'
    assert d.loc[d.sample_id=='t1','prediction_set'].iloc[0]=='["B"]'
    assert d.loc[d.sample_id=='t1','dor_reference_status'].iloc[0]=='HIERARCHY_SUPPORTED'
