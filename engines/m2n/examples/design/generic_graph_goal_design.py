"""P1/P2: generic mechanism graph -> qualification -> finite K2 design."""
from __future__ import annotations
import csv, json, tempfile
from pathlib import Path
import numpy as np
from m2n import run_goal_driven_design

A={'length':1,'time':-2}; V={'length':1,'time':-1}
spec={
  'schema_version':'2.0','template':'generic_mechanism_graph',
  'goal':{'name':'select a local resistance family'},
  'data':{'split_column':'split','train_values':['train'],'validation_values':['val']},
  'mechanism':{
    'inputs':[{'name':'u','dim':A},{'name':'v','dim':V}],
    'nodes':[
      {'id':'drive','op':'positive_gain','inputs':['u'],'params':{'learnable':True,'init':1.0,'gain_dim':{}}},
      {'id':'visc','op':'positive_gain','inputs':['v'],'params':{'learnable':True,'init':0.1,'gain_dim':{'time':-1}}},
      {'id':'sgn','op':'sign','inputs':['v'],'params':{}},
      {'id':'coul','op':'positive_gain','inputs':['sgn'],'params':{'learnable':True,'init':0.1,'gain_dim':A}},
      {'id':'resist','op':'add','inputs':['visc','coul'],'params':{}},
      {'id':'a','op':'subtract','inputs':['drive','resist'],'params':{}},
    ],
    'outputs':[{'name':'a','dim':A}],
  },
  'knowledge':{'slots':[{
    'id':'resistance','level':'K2','role':'local resistance','input_dims':[V],'output_dim':A,
    'finite_candidates':['viscous','viscous_coulomb'],'knowledge_source':'FINITE_ENGINEERING_HYPOTHESIS'
  }]},
  'bindings':{'resistance':{
    'viscous':[{'path':'nodes.3.params.learnable','value':False},{'path':'nodes.3.params.value','value':0.0}],
    'viscous_coulomb':[{'path':'nodes.3.params.learnable','value':True},{'path':'nodes.3.params.init','value':0.1}],
  }},
  'supervision':{'input_columns':{'u':'u','v':'v'},'output_columns':{'a':'a'}},
  'qualification':{'min_train_rows':80,'min_validation_rows':40},
  'training':{'epochs':300,'lr':0.05,'restarts':[11,12,13],'final_seed':90},
  'selection':{'practical_ratio':0.8,'restart_stability':0.75},
}

with tempfile.TemporaryDirectory(prefix='m2n_p1_') as td:
    td=Path(td); rng=np.random.default_rng(7); n=360
    u=rng.uniform(-1,1,n); v=rng.uniform(-1,1,n); v[np.abs(v)<.05]+=.08
    a=1.2*u-.2*v-.45*np.sign(v)
    split=np.array(['train']*240+['val']*120,dtype=object)
    idx=np.arange(n); rng.shuffle(idx)
    (td/'device_card.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
    with (td/'normal.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['u','v','a','split']); w.writerows(zip(u[idx],v[idx],a[idx],split))
    r=run_goal_driven_design(td/'device_card.json',td/'normal.csv',td/'output')
    print('status  =',r.status)
    print('selected=',r.selected_candidate_id)
    assert r.status=='SELECTED' and 'viscous_coulomb' in r.selected_candidate_id
    print('P1_P2_GENERIC_GRAPH_PASS')
