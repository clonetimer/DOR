import csv, json
from pathlib import Path
import numpy as np

from m2n.goal_design import run_goal_driven_design, generate_model_evidence
from m2n.phm_handoff import export_fec_contact_pack
from m2n.backend_router import assess_model_set_routing

A={'length':1,'time':-2}; V={'length':1,'time':-1}

def graph_spec(k3=False):
    k={'id':'resistance','level':'K2','role':'local resistance','input_dims':[V],'output_dim':A,'finite_candidates':['viscous','viscous_coulomb'],'knowledge_source':'FINITE_ENGINEERING_HYPOTHESIS'}
    bindings={
      'viscous':[{'path':'nodes.3.params.learnable','value':False},{'path':'nodes.3.params.value','value':0.0}],
      'viscous_coulomb':[{'path':'nodes.3.params.learnable','value':True},{'path':'nodes.3.params.init','value':0.1}],
    }
    if k3:
        k.update({'allow_k3_escape':True,'local_neural_family':'local_nn'})
        bindings['local_nn']=[{'path':'nodes.1.op','value':'dissipative_local_neural'},{'path':'nodes.1.params','value':{'output_dim':A,'hidden':6}},{'path':'nodes.3.params.learnable','value':False},{'path':'nodes.3.params.value','value':0.0}]
    return {
      'schema_version':'2.0','template':'generic_mechanism_graph','goal':{'name':'generic graph'},'data':{'split_column':'split','train_values':['train'],'validation_values':['val']},
      'mechanism':{'inputs':[{'name':'u','dim':A},{'name':'v','dim':V}], 'nodes':[
        {'id':'drive','op':'positive_gain','inputs':['u'],'params':{'learnable':True,'init':1.0,'gain_dim':{}}},
        {'id':'visc','op':'positive_gain','inputs':['v'],'params':{'learnable':True,'init':0.1,'gain_dim':{'time':-1}}},
        {'id':'sgn','op':'sign','inputs':['v'],'params':{}},
        {'id':'coul','op':'positive_gain','inputs':['sgn'],'params':{'learnable':True,'init':0.1,'gain_dim':A}},
        {'id':'resist','op':'add','inputs':['visc','coul'],'params':{}}, {'id':'a','op':'subtract','inputs':['drive','resist'],'params':{}}], 'outputs':[{'name':'a','dim':A}]},
      'knowledge':{'slots':[k]},'bindings':{'resistance':bindings},'supervision':{'input_columns':{'u':'u','v':'v'},'output_columns':{'a':'a'}},
      'qualification':{'min_train_rows':80,'min_validation_rows':40},'training':{'epochs':350,'lr':0.04,'restarts':[11,12,13],'final_seed':90},'selection':{'practical_ratio':0.8,'restart_stability':0.75},
      **({'k3_policy':{'enabled':True,'finite_rejection_nrmse':0.12,'minimum_relative_improvement':0.15,'max_k3_nrmse':0.12}} if k3 else {})
    }

def write_data(path:Path, nonlinear=False):
    rng=np.random.default_rng(5); n=360; u=rng.uniform(-1,1,n); v=rng.uniform(-1,1,n); v[np.abs(v)<.05]+=.08
    if nonlinear: a=1.2*u-.15*v-.9*v**3
    else: a=1.2*u-.2*v-.45*np.sign(v)
    split=np.array(['train']*240+['val']*120,dtype=object); idx=np.arange(n); rng.shuffle(idx)
    with path.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['time_s','u','v','a','split']); w.writerows(zip(np.arange(n)*.01,u[idx],v[idx],a[idx],split))

def test_p1_generic_graph_selects_finite(tmp_path):
    sp=tmp_path/'s.json'; sp.write_text(json.dumps(graph_spec(False))); dp=tmp_path/'d.csv'; write_data(dp,False)
    r=run_goal_driven_design(sp,dp,tmp_path/'o'); assert r.status=='SELECTED'; assert 'viscous_coulomb' in r.selected_candidate_id

def test_p4_qualified_k3_requires_finite_rejection(tmp_path):
    spec=graph_spec(True); sp=tmp_path/'s.json'; sp.write_text(json.dumps(spec)); dp=tmp_path/'d.csv'; write_data(dp,True)
    r=run_goal_driven_design(sp,dp,tmp_path/'o'); assert r.status=='K3_SELECTED'; d=json.loads((tmp_path/'o'/'design_result.json').read_text()); assert d['k3_gate']['status']=='K3_SUPPORTED'

def test_p5_evidence_and_fec_contact_pack(tmp_path):
    sp=tmp_path/'s.json'; sp.write_text(json.dumps(graph_spec(False))); dp=tmp_path/'d.csv'; write_data(dp,False); out=tmp_path/'o'
    r=run_goal_driven_design(sp,dp,out); assert r.status=='SELECTED'
    ev=generate_model_evidence(out,dp,tmp_path/'evidence.csv'); assert ev.is_file()
    c,j=export_fec_contact_pack(ev,tmp_path/'contacts.csv',tmp_path/'contract.json',window_s=.1,min_samples=4); assert c.is_file() and j.is_file()
    text=c.read_text(); assert 'm2n_best_finite_residual__a' in text and 'm2n_model_disagreement' in text

def test_p3_sbmi_routing_only_when_model_family_large(tmp_path):
    small=assess_model_set_routing({'a':('x','y'),'b':('u','v','w')},tmp_path,exact_enumeration_limit=32); assert small.status=='NOT_NEEDED'
    large=assess_model_set_routing({f's{i}':('a','b','c') for i in range(4)},tmp_path,exact_enumeration_limit=32); assert large.status=='MODEL_SET_BACKEND_RECOMMENDED'; assert Path(large.artifact).is_file()

def test_p2_rank_deficient_commissioning_abstains(tmp_path):
    spec=graph_spec(False); sp=tmp_path/'s.json'; sp.write_text(json.dumps(spec))
    n=160; u=np.linspace(-1,1,n); v=2*u; a=1.2*u-.2*v-.4*np.sign(v); split=np.array(['train']*110+['val']*50,dtype=object); dp=tmp_path/'d.csv'
    with dp.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['u','v','a','split']); w.writerows(zip(u,v,a,split))
    r=run_goal_driven_design(sp,dp,tmp_path/'o'); assert r.status=='AMBIGUOUS'; assert r.diagnostics['standardized_input_rank']<r.diagnostics['standardized_input_dimension']


def test_p3_symmatika_is_prepared_only_after_qualification(tmp_path):
    from m2n.backend_router import run_backend_requests
    spec=graph_spec(False); spec['backends']={'requests':[{'backend':'symmatika','slot_id':'resistance','feature_columns':['u','v'],'target_column':'a'}]}
    dp=tmp_path/'d.csv'; write_data(dp,False)
    from m2n.goal_design import _read_csv, _split_indices
    data=_read_csv(dp); tri,_=_split_indices(data,spec['data'])
    a=run_backend_requests(spec,data,tri,tmp_path/'b',qualification_status='READY'); assert a[0].status=='EXTERNAL_JOB_PREPARED'; assert Path(a[0].artifact).is_file()
