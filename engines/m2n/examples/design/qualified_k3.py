"""P4: finite K2 must fail a declared absolute fit gate before local K3 is allowed."""
from __future__ import annotations
import csv, json, tempfile
from pathlib import Path
import numpy as np
from m2n import run_goal_driven_design

A={'length':1,'time':-2}; V={'length':1,'time':-1}
spec={
  'schema_version':'2.0','template':'generic_mechanism_graph','goal':{'name':'qualified K3 demo'},
  'data':{'split_column':'split','train_values':['train'],'validation_values':['val']},
  'mechanism':{'inputs':[{'name':'u','dim':A},{'name':'v','dim':V}], 'nodes':[
    {'id':'drive','op':'positive_gain','inputs':['u'],'params':{'learnable':True,'init':1.0,'gain_dim':{}}},
    {'id':'visc','op':'positive_gain','inputs':['v'],'params':{'learnable':True,'init':0.1,'gain_dim':{'time':-1}}},
    {'id':'sgn','op':'sign','inputs':['v'],'params':{}},
    {'id':'coul','op':'positive_gain','inputs':['sgn'],'params':{'learnable':True,'init':0.1,'gain_dim':A}},
    {'id':'resist','op':'add','inputs':['visc','coul'],'params':{}},
    {'id':'a','op':'subtract','inputs':['drive','resist'],'params':{}},
  ],'outputs':[{'name':'a','dim':A}]},
  'knowledge':{'slots':[{
    'id':'resistance','level':'K2','role':'local resistance','input_dims':[V],'output_dim':A,
    'finite_candidates':['viscous','viscous_coulomb'],'knowledge_source':'FINITE_ENGINEERING_HYPOTHESIS',
    'allow_k3_escape':True,'local_neural_family':'local_nn'
  }]},
  'bindings':{'resistance':{
    'viscous':[{'path':'nodes.3.params.learnable','value':False},{'path':'nodes.3.params.value','value':0.0}],
    'viscous_coulomb':[{'path':'nodes.3.params.learnable','value':True},{'path':'nodes.3.params.init','value':0.1}],
    'local_nn':[
      {'path':'nodes.1.op','value':'dissipative_local_neural'},
      {'path':'nodes.1.params','value':{'output_dim':A,'hidden':6}},
      {'path':'nodes.3.params.learnable','value':False},{'path':'nodes.3.params.value','value':0.0},
    ],
  }},
  'supervision':{'input_columns':{'u':'u','v':'v'},'output_columns':{'a':'a'}},
  'qualification':{'min_train_rows':80,'min_validation_rows':40},
  'training':{'epochs':350,'lr':0.04,'restarts':[11,12,13],'final_seed':90},
  'selection':{'practical_ratio':0.8,'restart_stability':0.75},
  'k3_policy':{'enabled':True,'finite_rejection_nrmse':0.12,'minimum_relative_improvement':0.15,'max_k3_nrmse':0.12,'minimal_augmentation':{'enabled':True,'epochs':40,'fit_slack':0.10,'constraint_weight':150.0}},
}

with tempfile.TemporaryDirectory(prefix='m2n_p4_') as td:
    td=Path(td); rng=np.random.default_rng(5); n=360
    u=rng.uniform(-1,1,n); v=rng.uniform(-1,1,n); v[np.abs(v)<.05]+=.08
    a=1.2*u-.15*v-.9*v**3
    split=np.array(['train']*240+['val']*120,dtype=object); idx=np.arange(n); rng.shuffle(idx)
    (td/'device_card.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
    with (td/'normal.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['u','v','a','split']); w.writerows(zip(u[idx],v[idx],a[idx],split))
    r=run_goal_driven_design(td/'device_card.json',td/'normal.csv',td/'output')
    result=json.loads((td/'output'/'design_result.json').read_text())
    print('status=',r.status)
    print('finite_best_nrmse=',result['k3_gate']['finite_best_nrmse'])
    print('k3_nrmse=',result['k3_gate']['k3_nrmse'])
    print('relative_improvement=',result['k3_gate']['relative_improvement'])
    print('minimal_augmentation=',result.get('k3_training_policy'))
    print('credit_assignment_audit=',result.get('k3_credit_assignment_audit'))
    assert r.status=='K3_SELECTED'
    print('P4_QUALIFIED_K3_PASS')
