from pathlib import Path
import csv, json
import numpy as np
from m2n import run_goal_driven_design, predict_model_artifact


def _mechanics_spec():
    return {
      'schema_version':'1.0','template':'second_order_mechanics','goal':{'name':'test'},
      'data':{'split_column':'split','train_values':['train'],'validation_values':['val']},
      'mechanism':{
        'coordinate':{'position':'q','velocity':'v','position_dim':{'length':1}},
        'inputs':[{'name':'u','dim':{'length':1,'time':-2}}],
        'drive':{'id':'actuator','input':'u','gain_dim':{},'init':1.0,'law':'gain'},
        'resistance':{'id':'friction','default_law':'viscous','viscous_gain_init':0.1,'coulomb_gain_init':0.1},
        'dt':0.01},
      'knowledge':{'resistance':{'level':'K2','finite_candidates':['viscous','viscous_coulomb'],'knowledge_source':'FINITE_ENGINEERING_HYPOTHESIS'}},
      'supervision':{'position_column':'q','velocity_column':'v','input_columns':{'u':'u'},'target_acceleration_column':'a'},
      'qualification':{'min_train_rows':50,'min_validation_rows':20},
      'training':{'epochs':180,'lr':0.08,'restarts':[1,2,3]},
      'selection':{'practical_ratio':0.85,'restart_stability':0.75}}


def test_goal_design_selects_and_predicts(tmp_path:Path):
    spec=_mechanics_spec(); spec['schema_version']='2.3'; spec['data']['time_column']='t_s'
    spec['qualification'].update({'independent_context_column':'cycle_id','min_train_independent_contexts':3,'min_validation_independent_contexts':2})
    sp=tmp_path/'device_card.json'; sp.write_text(json.dumps(spec))
    rng=np.random.default_rng(3); n=160; v=rng.uniform(-1,1,n); v[np.abs(v)<.05]+=.1; u=rng.uniform(-1,1,n); a=1.2*u-.2*v-.45*np.sign(v); split=np.array(['train']*110+['val']*50,dtype=object)
    tr_context=np.array(['tr0']*37+['tr1']*37+['tr2']*36,dtype=object); va_context=np.array(['va0']*25+['va1']*25,dtype=object); context=np.r_[tr_context,va_context]
    t=np.r_[np.r_[np.arange(37),np.arange(37),np.arange(36)],np.r_[np.arange(25),np.arange(25)]]
    dp=tmp_path/'data.csv'
    with dp.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['q','v','u','a','split','cycle_id','t_s']); w.writerows(zip(np.zeros(n),v,u,a,split,context,t))
    out=tmp_path/'out'; r=run_goal_driven_design(sp,dp,out)
    assert r.status=='SELECTED'; assert 'viscous_coulomb' in r.selected_candidate_id
    arts=list((out/'models').glob('*.pt')); assert len(arts)==1
    pp=tmp_path/'pred.csv'; predict_model_artifact(arts[0],dp,pp); assert pp.exists()
    with pp.open(newline='') as f:
        header=next(csv.reader(f))
    assert header[:2]==['m2n_independent_context','time_s']


def test_goal_design_ambiguous_on_one_direction_friction(tmp_path:Path):
    spec=_mechanics_spec(); sp=tmp_path/'device_card.json'; sp.write_text(json.dumps(spec))
    rng=np.random.default_rng(4); n=100; v=rng.uniform(.1,1,n); u=rng.uniform(-1,1,n); a=1.2*u-.2*v-.45; split=np.array(['train']*70+['val']*30,dtype=object)
    dp=tmp_path/'data.csv'
    with dp.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['q','v','u','a','split']); w.writerows(zip(np.zeros(n),v,u,a,split))
    r=run_goal_driven_design(sp,dp,tmp_path/'out')
    assert r.status=='AMBIGUOUS'
    assert not (tmp_path/'out'/'models').exists()


def test_peak_goal_design(tmp_path:Path):
    spec={
      'schema_version':'1.0','template':'separable_peak_response','goal':{'name':'peak'},
      'data':{'split_column':'split','train_values':['train'],'validation_values':['val']},
      'mechanism':{'axes':[{'name':'x1','dim':{}},{'name':'x2','dim':{}},{'name':'x3','dim':{}}],
        'families':[{'name':'one','active_axes':['x1']},{'name':'two','active_axes':['x1','x2']},{'name':'three','active_axes':['x1','x2','x3']}],
        'output_name':'y','output_dim':{},'amplitude_init':1.0},
      'knowledge':{'response_surface':{'level':'K2','knowledge_source':'BLINDED_FOR_VALIDATION'}},
      'supervision':{'input_columns':{'x1':'x1','x2':'x2','x3':'x3'},'target_column':'y'},
      'training':{'epochs':220,'lr':.06,'restarts':[1,2,3]},
      'selection':{'practical_ratio':.8,'restart_stability':.75}}
    sp=tmp_path/'s.json'; sp.write_text(json.dumps(spec)); rng=np.random.default_rng(6); n=180; xs=[rng.normal(size=n) for _ in range(3)]; y=1.5*np.exp(-.5*((xs[0]-.2)/.8)**2-.5*((xs[1]+.1)/1.0)**2-.5*((xs[2]-.15)/.6)**2); split=np.array(['train']*120+['val']*60,dtype=object); dp=tmp_path/'d.csv'
    with dp.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['x1','x2','x3','y','split']); w.writerows(zip(xs[0],xs[1],xs[2],y,split))
    r=run_goal_driven_design(sp,dp,tmp_path/'o'); assert r.status=='SELECTED'; assert 'three' in r.selected_candidate_id

def test_goal_design_can_derive_acceleration(tmp_path:Path):
    spec=_mechanics_spec(); spec['supervision'].pop('target_acceleration_column'); spec['supervision']['derive_acceleration']={'time_column':'time','method':'gradient'}
    spec['knowledge']['resistance']={'level':'K1','exact_family':'viscous','learnable_parameters':['c_v'],'knowledge_source':'DECLARED_ENGINEERING'}
    sp=tmp_path/'s.json'; sp.write_text(json.dumps(spec))
    n=120; t=np.linspace(0,1.19,n); v=np.sin(t); q=1-np.cos(t); u=np.cos(t)+.2*np.sin(t); split=np.array(['train']*80+['val']*40,dtype=object); dp=tmp_path/'d.csv'
    with dp.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['time','q','v','u','split']); w.writerows(zip(t,q,v,u,split))
    r=run_goal_driven_design(sp,dp,tmp_path/'o'); assert r.status=='FIXED'; assert r.diagnostics.get('derived_supervision')
