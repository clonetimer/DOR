from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

from m2n.temporal_design import qualify_temporal_design, run_temporal_design, predict_temporal_design


def _spec(objective='CATEGORICAL_PREDICTION'):
    mc={
        'objective': objective,
        'input_representation':'MULTIVARIATE_SEQUENCE',
        'mechanism_usage':'NONE',
        'prediction_unit':'INDEPENDENT_CONTEXT',
        'input_columns':['x1','x2'],
        'time_column':'t_s',
        'constraints':{'temporal_training':{'epochs':5,'hidden_size':8,'seeds':[1,2],'lr':0.02,'lookback':3}},
    }
    if objective=='CATEGORICAL_PREDICTION': mc['label_column']='label'
    if objective=='DYNAMICS_IDENTIFICATION': mc['target_columns']=['y']
    return {
      'schema_version':'2.7','template':'generic_mechanism_graph',
      'data':{'split_column':'split','train_values':['train'],'validation_values':['val'],'time_column':'t_s'},
      'mechanism':{'inputs':[{'name':'x1','dim':{}},{'name':'x2','dim':{}}], 'nodes':[{'id':'yhat','op':'positive_gain','inputs':['x1'],'params':{'learnable':True,'init':1.0,'gain_dim':{}}}], 'outputs':[{'name':'yhat','dim':{}}]},
      'knowledge':{'slots':[{'id':'local','level':'K1','role':'local','input_dims':[{}],'output_dim':{},'exact_family':'gain','learnable_parameters':['g'],'knowledge_source':'DECLARED_ENGINEERING'}]},
      'supervision':{'input_columns':{'x1':'x1'},'output_columns':{'yhat':'x1'}},
      'qualification':{'independent_context_column':'cycle_id','evidence_scope':'BLINDED_FOR_VALIDATION'},
      'modeling_contract':mc,
    }


def _classification_data():
    rows=[]
    # Same marginal values in both classes, different order. 8 train + 4 val contexts.
    for c in range(12):
        label='A' if c%2==0 else 'B'; split='train' if c<8 else 'val'
        base=np.array([0.,0.,1.,1.,0.,0.,-1.,-1.])
        x=base if label=='A' else base[::-1]
        for t,v in enumerate(x): rows.append((c,t,split,v,0.5*v,label))
    return {
        'cycle_id':np.array([r[0] for r in rows]), 't_s':np.array([r[1] for r in rows],float), 'split':np.array([r[2] for r in rows]),
        'x1':np.array([r[3] for r in rows],float), 'x2':np.array([r[4] for r in rows],float), 'label':np.array([r[5] for r in rows]),
    }


def test_temporal_qualification_generates_distinct_architecture_candidates():
    q=qualify_temporal_design(_spec(),_classification_data())
    assert q.status=='READY'
    ids={c.id for c in q.candidates}
    assert {'summary_mlp','tcn','gru','transformer'} <= ids
    assert q.train_contexts==8 and q.validation_contexts==4
    assert q.regular_sampling


def test_temporal_qualification_rejects_context_split_leakage():
    d=_classification_data(); d['split'][0]='val'
    q=qualify_temporal_design(_spec(),d)
    assert q.status=='UNSUPPORTED'
    assert any('crosses split' in r for r in q.reasons)


def _write_csv(path:Path,data):
    cols=list(data)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(cols)
        for i in range(len(data[cols[0]])): w.writerow([data[c][i] for c in cols])


def test_temporal_design_end_to_end_sequence_classification(tmp_path:Path):
    spec=_spec(); sp=tmp_path/'spec.json'; sp.write_text(json.dumps(spec),encoding='utf-8')
    dp=tmp_path/'data.csv'; _write_csv(dp,_classification_data())
    r=run_temporal_design(sp,dp,tmp_path/'out')
    assert r.status in ('SELECTED','AMBIGUOUS')
    assert set(r.candidate_validation_risks)>={'summary_mlp','tcn','gru','transformer'}
    assert (tmp_path/'out'/'temporal_design_result.json').is_file()
    assert (tmp_path/'out'/'models').is_dir()
    pred=predict_temporal_design(tmp_path/'out',dp,tmp_path/'pred.csv')
    text=pred.read_text(encoding='utf-8')
    assert 'portfolio_argmax' in text and 'cycle_id' in text


def test_temporal_forecast_candidate_generation():
    spec=_spec('DYNAMICS_IDENTIFICATION')
    d=_classification_data(); d['y']=0.7*d['x1']+0.2*d['x2']
    q=qualify_temporal_design(spec,d)
    assert q.status=='READY'
    ids={c.id for c in q.candidates}
    assert {'linear_ar','tcn','gru','transformer'} <= ids


def test_temporal_design_end_to_end_forecast_and_predict(tmp_path:Path):
    spec=_spec('DYNAMICS_IDENTIFICATION')
    d=_classification_data(); d['y']=np.roll(d['x1'],1); d['y'][np.arange(len(d['y']))%8==0]=0.0
    sp=tmp_path/'spec_forecast.json'; sp.write_text(json.dumps(spec),encoding='utf-8')
    dp=tmp_path/'forecast.csv'; _write_csv(dp,d)
    r=run_temporal_design(sp,dp,tmp_path/'forecast_out')
    assert r.status in ('SELECTED','AMBIGUOUS')
    assert set(r.candidate_validation_risks)>={'linear_ar','tcn','gru','transformer'}
    pred=predict_temporal_design(tmp_path/'forecast_out',dp,tmp_path/'forecast_pred.csv')
    assert 'portfolio_prediction__y' in pred.read_text(encoding='utf-8')


def test_temporal_selection_can_freeze_a_clear_winner(tmp_path:Path):
    spec=_spec(); spec['modeling_contract']['constraints']['temporal_candidates']=['SUMMARY_MLP','GRU']
    spec['modeling_contract']['constraints']['temporal_training']['epochs']=8
    sp=tmp_path/'sel.json'; sp.write_text(json.dumps(spec),encoding='utf-8')
    dp=tmp_path/'sel.csv'; _write_csv(dp,_classification_data())
    r=run_temporal_design(sp,dp,tmp_path/'sel_out')
    assert r.status=='SELECTED'
    assert r.selected_candidate_id=='gru'



def test_temporal_nonfinite_inputs_fail_closed_without_training(tmp_path:Path):
    spec=_spec('DYNAMICS_IDENTIFICATION')
    d=_classification_data(); d['y']=0.5*d['x1']
    x=d['x1'].astype(object); x[5]=''; d['x1']=x
    q=qualify_temporal_design(spec,d)
    assert q.status=='AMBIGUOUS'
    assert any('preprocessing/missing-data strategy' in r for r in q.reasons)
    sp=tmp_path/'missing.json'; sp.write_text(json.dumps(spec),encoding='utf-8')
    dp=tmp_path/'missing.csv'; _write_csv(dp,d)
    r=run_temporal_design(sp,dp,tmp_path/'missing_out')
    assert r.status=='AMBIGUOUS'
    assert not r.candidate_validation_risks


def _k1_integrator_case():
    rows=[]
    for c in range(12):
        split='train' if c<8 else 'val'
        state=0.25*c
        for t in range(20):
            u=((t%5)-2)*0.1 + 0.01*c
            rows.append((c,float(t),split,state,u))
            state=state+2.0*u
    return {
        'cycle_id':np.array([r[0] for r in rows]),
        't_s':np.array([r[1] for r in rows],float),
        'split':np.array([r[2] for r in rows]),
        'state':np.array([r[3] for r in rows],float),
        'u':np.array([r[4] for r in rows],float),
    }


def _k1_integrator_spec():
    spec=_spec('DYNAMICS_IDENTIFICATION')
    spec['modeling_contract'].update({
        'mechanism_usage':'PREFERRED',
        'input_columns':['state','u'],
        'target_columns':['state'],
    })
    spec['modeling_contract']['constraints']={
        'temporal_candidates':['K1_INTEGRATOR','LINEAR_AR'],
        'temporal_physics':{
            'kind':'K1_INTEGRATOR',
            'state_columns':['state'],
            'drive_columns':['u'],
            'dt':1.0,
            'gain_mode':'LEARNED',
        },
        'temporal_training':{'epochs':3,'hidden_size':8,'seeds':[1,2,3],'lr':0.02,'lookback':3},
    }
    return spec


def test_k1_integrator_enters_temporal_competition_and_recovers_gain(tmp_path:Path):
    spec=_k1_integrator_spec(); d=_k1_integrator_case()
    q=qualify_temporal_design(spec,d)
    assert q.status=='READY'
    assert {c.id for c in q.candidates}=={'k1_integrator','linear_ar'}
    sp=tmp_path/'k1.json'; sp.write_text(json.dumps(spec),encoding='utf-8')
    dp=tmp_path/'k1.csv'; _write_csv(dp,d)
    r=run_temporal_design(sp,dp,tmp_path/'k1_out')
    assert 'k1_integrator' in r.candidate_validation_risks
    assert max(r.candidate_validation_risks['k1_integrator']) <= 1e-8
    payload=__import__('torch').load(tmp_path/'k1_out'/'models'/'k1_integrator.pt',map_location='cpu',weights_only=False)
    assert abs(payload['model_metadata']['learned_gains'][0]-2.0) <= 1e-8
    pred=predict_temporal_design(tmp_path/'k1_out',dp,tmp_path/'k1_pred.csv')
    assert 'k1_integrator__prediction__state' in pred.read_text(encoding='utf-8')


def test_linear_ar_is_deterministic_across_declared_seeds(tmp_path:Path):
    spec=_k1_integrator_spec(); d=_k1_integrator_case()
    sp=tmp_path/'lin.json'; sp.write_text(json.dumps(spec),encoding='utf-8')
    dp=tmp_path/'lin.csv'; _write_csv(dp,d)
    r=run_temporal_design(sp,dp,tmp_path/'lin_out')
    vals=list(r.candidate_validation_risks['linear_ar'])
    assert len(vals)==3
    assert max(vals)-min(vals) < 1e-12


def _quaternion_case():
    def qmul(a,b):
        aw,av=a[0],a[1:]; bw,bv=b[0],b[1:]
        return np.r_[aw*bw-np.dot(av,bv), aw*bv+bw*av+np.cross(av,bv)]
    rows=[]
    for c in range(10):
        split='train' if c<6 else 'val'
        q=np.array([1.0,0.0,0.0,0.0])
        for t in range(18):
            w=np.array([0.002*(c+1), -0.0015*np.sin(0.2*t+c), 0.001*np.cos(0.3*t)])
            rows.append((c,float(t),split,*q,*w))
            th=np.linalg.norm(w)
            if th>1e-15:
                dq=np.r_[np.cos(th/2.0), w/th*np.sin(th/2.0)]
            else:
                dq=np.r_[1.0,0.5*w]
            q=qmul(dq,q); q=q/np.linalg.norm(q)
    arr=list(zip(*rows))
    return {
        'cycle_id':np.asarray(arr[0]),'t_s':np.asarray(arr[1],float),'split':np.asarray(arr[2]),
        'q0':np.asarray(arr[3],float),'q1':np.asarray(arr[4],float),'q2':np.asarray(arr[5],float),'q3':np.asarray(arr[6],float),
        'wx':np.asarray(arr[7],float),'wy':np.asarray(arr[8],float),'wz':np.asarray(arr[9],float),
    }


def _quaternion_spec(mechanism_usage='REQUIRED'):
    spec=_spec('DYNAMICS_IDENTIFICATION')
    spec['modeling_contract'].update({
        'mechanism_usage':mechanism_usage,
        'input_columns':['q0','q1','q2','q3','wx','wy','wz'],
        'target_columns':['q0','q1','q2','q3'],
    })
    candidates=['QUATERNION_KINEMATICS'] if mechanism_usage=='REQUIRED' else ['QUATERNION_KINEMATICS','LINEAR_AR']
    spec['modeling_contract']['constraints']={
        'temporal_candidates':candidates,
        'temporal_target_semantics':'UNIT_QUATERNION',
        'temporal_physics':{
            'kind':'QUATERNION_KINEMATICS','state_columns':['q0','q1','q2','q3'],
            'drive_columns':['wx','wy','wz'],'dt':1.0,'composition':'LEFT','rate_sign':1.0,
        },
        'temporal_training':{'epochs':3,'hidden_size':8,'seeds':[1,2,3],'lr':0.02,'lookback':3},
    }
    return spec


def test_required_quaternion_mechanism_can_execute_as_single_contract_candidate(tmp_path:Path):
    spec=_quaternion_spec('REQUIRED'); d=_quaternion_case()
    q=qualify_temporal_design(spec,d)
    assert q.status=='READY'
    assert [c.id for c in q.candidates]==['quaternion_kinematics']
    sp=tmp_path/'q.json'; sp.write_text(json.dumps(spec),encoding='utf-8')
    dp=tmp_path/'q.csv'; _write_csv(dp,d)
    r=run_temporal_design(sp,dp,tmp_path/'q_out')
    assert r.status=='SELECTED' and r.selected_candidate_id=='quaternion_kinematics'
    assert r.selection['selection_basis']=='REQUIRED_MECHANISM_CONTRACT'
    assert max(r.candidate_validation_risks['quaternion_kinematics']) <= 1e-8
    pred=predict_temporal_design(tmp_path/'q_out',dp,tmp_path/'q_pred.csv')
    text=pred.read_text(encoding='utf-8')
    assert 'quaternion_kinematics__angular_error_deg' in text


def test_preferred_quaternion_mechanism_competes_with_generic_temporal_baseline():
    spec=_quaternion_spec('PREFERRED'); d=_quaternion_case()
    q=qualify_temporal_design(spec,d)
    assert q.status=='READY'
    assert {c.id for c in q.candidates}=={'quaternion_kinematics','linear_ar'}


def test_unit_quaternion_semantics_requires_four_targets():
    spec=_quaternion_spec('PREFERRED'); spec['modeling_contract']['target_columns']=['q0','q1','q2']
    d=_quaternion_case(); q=qualify_temporal_design(spec,d)
    assert q.status in ('AMBIGUOUS','UNSUPPORTED')
    assert any('exactly four' in r for r in q.reasons)


def _affine_state_space_case():
    rows=[]
    for c in range(10):
        split='train' if c<6 else 'val'
        x=1.0+0.2*c
        for t in range(24):
            u1=0.3*np.sin(0.25*t+0.1*c)+0.02*c
            u2=0.2*np.cos(0.17*t-0.05*c)
            rows.append((c,float(t),split,x,u1,u2))
            x=0.92*x+0.35*u1-0.18*u2+0.12
    arr=list(zip(*rows))
    return {
        'cycle_id':np.asarray(arr[0]),'t_s':np.asarray(arr[1],float),'split':np.asarray(arr[2]),
        'x':np.asarray(arr[3],float),'u1':np.asarray(arr[4],float),'u2':np.asarray(arr[5],float),
    }


def _affine_state_space_spec(mechanism_usage='PREFERRED'):
    spec=_spec('DYNAMICS_IDENTIFICATION')
    spec['modeling_contract'].update({
        'mechanism_usage':mechanism_usage,
        'input_columns':['x','u1','u2'],
        'target_columns':['x'],
    })
    candidates=['AFFINE_STATE_SPACE'] if mechanism_usage=='REQUIRED' else ['AFFINE_STATE_SPACE','LINEAR_AR']
    spec['modeling_contract']['constraints']={
        'temporal_candidates':candidates,
        'temporal_physics':{
            'kind':'AFFINE_STATE_SPACE','state_columns':['x'],'drive_columns':['u1','u2'],
            'dt':1.0,'include_bias':True,'ridge':1e-10,
        },
        'temporal_training':{'epochs':3,'hidden_size':8,'seeds':[1,2,3],'lr':0.02,'lookback':3},
    }
    return spec


def test_affine_state_space_enters_temporal_competition_and_recovers_parameters(tmp_path:Path):
    spec=_affine_state_space_spec('PREFERRED'); d=_affine_state_space_case()
    q=qualify_temporal_design(spec,d)
    assert q.status=='READY'
    assert {c.id for c in q.candidates}=={'affine_state_space','linear_ar'}
    sp=tmp_path/'aff.json'; sp.write_text(json.dumps(spec),encoding='utf-8')
    dp=tmp_path/'aff.csv'; _write_csv(dp,d)
    r=run_temporal_design(sp,dp,tmp_path/'aff_out')
    assert max(r.candidate_validation_risks['affine_state_space']) <= 1e-8
    payload=__import__('torch').load(tmp_path/'aff_out'/'models'/'affine_state_space.pt',map_location='cpu',weights_only=False) if (tmp_path/'aff_out'/'models'/'affine_state_space.pt').exists() else None
    # If selection froze the equally exact linear baseline instead, rerun REQUIRED to inspect the admitted state-space artifact.
    if payload is None:
        req=_affine_state_space_spec('REQUIRED'); rsp=tmp_path/'aff_req.json'; rsp.write_text(json.dumps(req),encoding='utf-8')
        rr=run_temporal_design(rsp,dp,tmp_path/'aff_req_out')
        assert rr.status=='SELECTED' and rr.selected_candidate_id=='affine_state_space'
        payload=__import__('torch').load(tmp_path/'aff_req_out'/'models'/'affine_state_space.pt',map_location='cpu',weights_only=False)
    meta=payload['model_metadata']
    assert abs(meta['state_matrix_A'][0][0]-0.92) < 1e-8
    assert abs(meta['drive_matrix_B'][0][0]-0.35) < 1e-8
    assert abs(meta['drive_matrix_B'][0][1]+0.18) < 1e-8
    assert abs(meta['bias'][0]-0.12) < 1e-8


def test_required_affine_state_space_executes_and_predicts(tmp_path:Path):
    spec=_affine_state_space_spec('REQUIRED'); d=_affine_state_space_case()
    sp=tmp_path/'req_aff.json'; sp.write_text(json.dumps(spec),encoding='utf-8')
    dp=tmp_path/'req_aff.csv'; _write_csv(dp,d)
    r=run_temporal_design(sp,dp,tmp_path/'req_aff_out')
    assert r.status=='SELECTED' and r.selected_candidate_id=='affine_state_space'
    assert r.selection['selection_basis']=='REQUIRED_MECHANISM_CONTRACT'
    pred=predict_temporal_design(tmp_path/'req_aff_out',dp,tmp_path/'req_aff_pred.csv')
    assert 'affine_state_space__prediction__x' in pred.read_text(encoding='utf-8')


def _bounded_integrator_case():
    rows=[]
    # Variable capacity across independent contexts; deliberately force saturation.
    for c in range(8):
        split='train' if c<6 else 'val'
        cap=10.0+0.5*c
        x=0.0
        for t in range(14):
            u=2.0 if t>=3 else 0.5
            rows.append((c,float(t),split,x,u,cap))
            x=min(max(x+u,0.0),cap)
    a=list(zip(*rows))
    return {'cycle_id':np.asarray(a[0]),'t_s':np.asarray(a[1],float),'split':np.asarray(a[2]),
            'x':np.asarray(a[3],float),'u':np.asarray(a[4],float),'cap':np.asarray(a[5],float)}


def _bounded_integrator_spec(mechanism_usage='REQUIRED'):
    spec=_spec('DYNAMICS_IDENTIFICATION')
    spec['schema_version']='2.9'
    spec['modeling_contract'].update({'mechanism_usage':mechanism_usage,'input_columns':['x','u','cap'],'target_columns':['x']})
    spec['modeling_contract']['constraints']={
        'temporal_candidates':['BOUNDED_INTEGRATOR'] if mechanism_usage=='REQUIRED' else ['BOUNDED_INTEGRATOR','LINEAR_AR'],
        'temporal_physics':{
            'kind':'BOUNDED_INTEGRATOR','state_columns':['x'],'drive_columns':['u'],'dt':1.0,
            'gain_mode':'FIXED','fixed_gains':[1.0],'lower_bounds':[0.0],'upper_bound_columns':['cap'],
        },
        'temporal_training':{'epochs':2,'hidden_size':8,'seeds':[1,2,3],'lr':0.02,'lookback':3},
    }
    return spec


def test_bounded_integrator_executes_variable_bounds_and_predicts(tmp_path:Path):
    spec=_bounded_integrator_spec('REQUIRED'); d=_bounded_integrator_case()
    q=qualify_temporal_design(spec,d)
    assert q.status=='READY' and [c.id for c in q.candidates]==['bounded_integrator']
    sp=tmp_path/'b.json'; sp.write_text(json.dumps(spec),encoding='utf-8')
    dp=tmp_path/'b.csv'; _write_csv(dp,d)
    r=run_temporal_design(sp,dp,tmp_path/'bout')
    assert r.status=='SELECTED' and r.selected_candidate_id=='bounded_integrator'
    assert max(r.candidate_validation_risks['bounded_integrator']) <= 1e-8
    pred=predict_temporal_design(tmp_path/'bout',dp,tmp_path/'bpred.csv')
    assert 'bounded_integrator__prediction__x' in pred.read_text(encoding='utf-8')


def test_bounded_integrator_rejects_missing_bound_contract():
    spec=_bounded_integrator_spec('REQUIRED')
    del spec['modeling_contract']['constraints']['temporal_physics']['upper_bound_columns']
    q=qualify_temporal_design(spec,_bounded_integrator_case())
    assert q.status in ('AMBIGUOUS','UNSUPPORTED')
    assert any('upper_bounds or upper_bound_columns' in r for r in q.reasons)


def _context_support_case():
    rows=[]
    # train envelope length [6,8], duration [5,7]
    for cid,L,sp in [('tr0',6,'train'),('tr1',7,'train'),('tr2',8,'train'),('va0',6,'val'),('va1',8,'val'),('va_long',20,'val')]:
        for i in range(L):
            rows.append((cid,float(i),sp,float(i),float(i+0.1)))
    a=np.asarray(rows,dtype=object)
    return {'cycle_id':a[:,0], 't_s':a[:,1].astype(float), 'split':a[:,2], 'x1':a[:,3].astype(float), 'x2':a[:,4].astype(float)}


def test_temporal_context_support_partitions_validation_without_inventing_thresholds(tmp_path:Path):
    spec=_spec(); spec['modeling_contract']['objective']='DYNAMICS_IDENTIFICATION'; spec['modeling_contract']['target_columns']=['x1']
    spec['modeling_contract']['constraints']={
        'max_dt_relative_spread':0.01,
        'temporal_candidates':['LINEAR_AR','GRU'],
        'temporal_training':{'epochs':2,'hidden_size':6,'seeds':[1,2],'lr':0.02,'lookback':2},
        'temporal_context_support':{'mode':'EMPIRICAL_TRAIN_ENVELOPE','selection_scope':'IN_SUPPORT_ONLY','dimensions':['length','duration']},
    }
    d=_context_support_case(); q=qualify_temporal_design(spec,d)
    assert q.status=='READY'
    assert q.context_support['train_envelope']['length']=={'min':6.0,'max':8.0}
    assert q.context_support['validation_in_support_contexts']==2
    assert q.context_support['validation_out_of_support_contexts']==1
    # design must still run because two in-support validation contexts remain
    sp=tmp_path/'spec.json'; dp=tmp_path/'data.csv'; sp.write_text(json.dumps(spec))
    _write_csv(dp,d)
    r=run_temporal_design(sp,dp,tmp_path/'cs_out')
    assert r.status in ('SELECTED','AMBIGUOUS')
    assert r.qualification['context_support']['selection_validation_contexts']==2


def test_temporal_prediction_exposes_frozen_context_support_flag(tmp_path:Path):
    spec=_spec(); spec['modeling_contract']['objective']='DYNAMICS_IDENTIFICATION'; spec['modeling_contract']['target_columns']=['x1']
    spec['modeling_contract']['constraints']={
        'max_dt_relative_spread':0.01,
        'temporal_candidates':['LINEAR_AR','GRU'],
        'temporal_training':{'epochs':2,'hidden_size':6,'seeds':[1,2],'lr':0.02,'lookback':2},
        'temporal_context_support':{'mode':'EMPIRICAL_TRAIN_ENVELOPE','selection_scope':'IN_SUPPORT_ONLY','dimensions':['length','duration']},
    }
    d=_context_support_case(); sp=tmp_path/'spec.json'; dp=tmp_path/'data.csv'; sp.write_text(json.dumps(spec)); _write_csv(dp,d)
    r=run_temporal_design(sp,dp,tmp_path/'pred_cs_out')
    pred=predict_temporal_design(tmp_path/'pred_cs_out',dp,tmp_path/'pred_cs.csv')
    import csv as _csv
    rows=list(_csv.DictReader(open(pred)))
    flags={row['cycle_id']:row['m2n_context_support'] for row in rows}
    assert flags['va0']=='IN_SUPPORT'
    assert flags['va_long']=='OUT_OF_SUPPORT'


def _operating_context_support_case():
    rows=[]
    contexts=[
        ('tr0','train',0.0),('tr1','train',1.0),('tr2','train',2.0),('tr3','train',1.5),
        ('va0','val',0.5),('va1','val',1.8),('va_far','val',10.0),
    ]
    for cid,sp,regime in contexts:
        for i in range(7):
            x=float(i)+0.05*regime
            rows.append((cid,float(i),sp,x,x+0.1,regime))
    a=np.asarray(rows,dtype=object)
    return {
        'cycle_id':a[:,0], 't_s':a[:,1].astype(float), 'split':a[:,2],
        'x1':a[:,3].astype(float), 'x2':a[:,4].astype(float), 'regime':a[:,5].astype(float),
    }


def test_temporal_operating_condition_support_filters_regime_extrapolation(tmp_path:Path):
    spec=_spec(); spec['schema_version']='2.12'
    spec['modeling_contract']['objective']='DYNAMICS_IDENTIFICATION'
    spec['modeling_contract']['input_columns']=['x1','x2','regime']
    spec['modeling_contract']['target_columns']=['x1']
    spec['modeling_contract']['constraints']={
        'max_dt_relative_spread':0.01,
        'temporal_candidates':['LINEAR_AR','GRU'],
        'temporal_training':{'epochs':2,'hidden_size':6,'seeds':[1,2],'lr':0.02,'lookback':2},
        'temporal_context_support':{
            'mode':'EMPIRICAL_TRAIN_ENVELOPE','selection_scope':'IN_SUPPORT_ONLY','dimensions':['length','duration'],
            'operating_condition':{
                'columns':['regime'],'statistic':'MEAN','metric':'STANDARDIZED_NEAREST_NEIGHBOR',
                'threshold_rule':'TRAIN_LEAVE_ONE_OUT_MAX',
            },
        },
    }
    d=_operating_context_support_case(); q=qualify_temporal_design(spec,d)
    assert q.status=='READY'
    cs=q.context_support
    assert cs['validation_shape_out_of_support_contexts']==0
    assert cs['validation_operating_out_of_support_contexts']==1
    assert cs['validation_in_support_contexts']==2
    op=cs['operating_condition_support']
    assert op['threshold_rule']=='TRAIN_LEAVE_ONE_OUT_MAX'
    assert op['validation_out_of_support_contexts']==1
    sp=tmp_path/'spec_op.json'; dp=tmp_path/'data_op.csv'; sp.write_text(json.dumps(spec)); _write_csv(dp,d)
    r=run_temporal_design(sp,dp,tmp_path/'op_out')
    assert r.status in ('SELECTED','AMBIGUOUS')
    assert r.qualification['context_support']['selection_validation_contexts']==2


def test_temporal_prediction_exposes_operating_condition_support(tmp_path:Path):
    spec=_spec(); spec['schema_version']='2.12'
    spec['modeling_contract']['objective']='DYNAMICS_IDENTIFICATION'
    spec['modeling_contract']['input_columns']=['x1','x2','regime']
    spec['modeling_contract']['target_columns']=['x1']
    spec['modeling_contract']['constraints']={
        'max_dt_relative_spread':0.01,
        'temporal_candidates':['LINEAR_AR','GRU'],
        'temporal_training':{'epochs':2,'hidden_size':6,'seeds':[1,2],'lr':0.02,'lookback':2},
        'temporal_context_support':{
            'mode':'EMPIRICAL_TRAIN_ENVELOPE','selection_scope':'IN_SUPPORT_ONLY','dimensions':['length','duration'],
            'operating_condition':{
                'columns':['regime'],'statistic':'MEAN','metric':'STANDARDIZED_NEAREST_NEIGHBOR',
                'threshold_rule':'TRAIN_LEAVE_ONE_OUT_MAX',
            },
        },
    }
    d=_operating_context_support_case(); sp=tmp_path/'spec_op2.json'; dp=tmp_path/'data_op2.csv'; sp.write_text(json.dumps(spec)); _write_csv(dp,d)
    run_temporal_design(sp,dp,tmp_path/'op_pred_out')
    pred=predict_temporal_design(tmp_path/'op_pred_out',dp,tmp_path/'op_pred.csv')
    import csv as _csv
    rows=list(_csv.DictReader(open(pred)))
    first={}
    for row in rows: first.setdefault(row['cycle_id'],row)
    assert first['va0']['m2n_context_support']=='IN_SUPPORT'
    assert first['va0']['m2n_context_operating_support']=='IN_SUPPORT'
    assert first['va_far']['m2n_context_support']=='OUT_OF_SUPPORT'
    assert first['va_far']['m2n_context_shape_support']=='IN_SUPPORT'
    assert first['va_far']['m2n_context_operating_support']=='OUT_OF_SUPPORT'
    assert float(first['va_far']['m2n_context_operating_distance']) > 0


def _closed_rollout_case():
    rows=[]
    # Smooth but slightly curved scalar state; contexts stay independent.
    for c in range(8):
        split='train' if c<5 else 'val'
        x=1.0+0.05*c
        for t in range(14):
            rows.append((f'c{c}',float(t),split,x))
            x=0.985*x-0.002+0.0002*t
    a=np.asarray(rows,dtype=object)
    return {
        'cycle_id':a[:,0], 't_s':a[:,1].astype(float), 'split':a[:,2],
        'x1':a[:,3].astype(float),
    }


def _closed_rollout_spec(horizon=4):
    spec=_spec('DYNAMICS_IDENTIFICATION')
    spec['schema_version']='2.16'
    spec['modeling_contract']['input_columns']=['x1']
    spec['modeling_contract']['target_columns']=['x1']
    spec['modeling_contract']['constraints']={
        'temporal_candidates':['LINEAR_AR','TCN'],
        'temporal_training':{'epochs':2,'hidden_size':6,'seeds':[1,2],'lr':0.02,'lookback':3},
        'forecast_evaluation':{'mode':'RECURSIVE_ROLLOUT','horizon_steps':horizon},
    }
    return spec


def test_recursive_rollout_qualification_ready_for_closed_state():
    q=qualify_temporal_design(_closed_rollout_spec(4),_closed_rollout_case())
    assert q.status=='READY'
    assert q.forecast_evaluation['mode']=='RECURSIVE_ROLLOUT'
    assert q.forecast_evaluation['horizon_steps']==4
    assert q.forecast_evaluation['resolved_lookback']==3


def test_recursive_rollout_rejects_future_exogenous_inputs():
    spec=_closed_rollout_spec(4)
    spec['modeling_contract']['input_columns']=['x1','x2']
    d=_closed_rollout_case(); d['x2']=0.5*np.asarray(d['x1'])
    q=qualify_temporal_design(spec,d)
    assert q.status=='UNSUPPORTED'
    assert any('input_columns == target_columns' in r for r in q.reasons)


def test_recursive_rollout_rejects_horizon_beyond_sequence_support():
    q=qualify_temporal_design(_closed_rollout_spec(20),_closed_rollout_case())
    assert q.status=='UNSUPPORTED'
    assert any('exceeds the minimum-sequence supported horizon' in r for r in q.reasons)


def test_recursive_rollout_design_preserves_one_step_risks_and_predicts_horizon(tmp_path:Path):
    spec=_closed_rollout_spec(4); d=_closed_rollout_case()
    sp=tmp_path/'spec_rollout.json'; dp=tmp_path/'rollout.csv'
    sp.write_text(json.dumps(spec),encoding='utf-8'); _write_csv(dp,d)
    r=run_temporal_design(sp,dp,tmp_path/'rollout_out')
    assert r.status in ('SELECTED','AMBIGUOUS')
    assert r.forecast_evaluation['mode']=='RECURSIVE_ROLLOUT'
    assert r.candidate_one_step_validation_risks is not None
    assert set(r.candidate_one_step_validation_risks)==set(r.candidate_validation_risks)
    pred=predict_temporal_design(tmp_path/'rollout_out',dp,tmp_path/'rollout_pred.csv')
    import csv as _csv
    rows=list(_csv.DictReader(open(pred)))
    assert rows
    assert rows[0]['m2n_forecast_mode']=='RECURSIVE_ROLLOUT'
    assert rows[0]['m2n_forecast_horizon_steps']=='4'
    key='portfolio_rollout_h4_prediction__x1'
    assert key in rows[0]
    vals=[row[key].lower() for row in rows]
    assert any(v not in ('nan','') for v in vals)
    assert any(v=='nan' for v in vals)
