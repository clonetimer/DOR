from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

from m2n.temporal_design import qualify_temporal_design, run_temporal_design, predict_temporal_design
from m2n.model_family import qualify_model_families


def _spec():
    return {
      'schema_version':'2.10','template':'generic_mechanism_graph',
      'data':{'split_column':'split','train_values':['train'],'validation_values':['val'],'time_column':'t_s'},
      'qualification':{'independent_context_column':'case_id','evidence_scope':'BLINDED_FOR_VALIDATION'},
      'modeling_contract':{
        'objective':'CATEGORICAL_PREDICTION','input_representation':'WAVEFORM_SEQUENCE','mechanism_usage':'NONE',
        'prediction_unit':'INDEPENDENT_CONTEXT','input_columns':['p1','p2'],'label_column':'label','time_column':'t_s',
        'required_outputs':['class_score'],
        'constraints':{
          'waveform_representation':{'sample_window':[0,10],'feature_set':'TIME_DOMAIN_V1','ridge_alpha':1.0},
          'temporal_candidates':['WAVEFORM_FEATURE_LINEAR','SUMMARY_MLP'],
          'temporal_training':{'epochs':3,'hidden_size':8,'seeds':[1,2,3],'lr':0.02}
        }
      },
      'selection':{'practical_ratio':0.8,'restart_stability':0.75},
    }


def _data():
    rows=[]
    # Both classes have the same marginal value counts and therefore the same
    # whole-sequence mean/std/min/max.  They differ in local line-integral shape
    # inside the declared waveform region of interest.
    a=np.array([1.,-1.,1.,-1.,1.,-1.,1.,-1.,1.,-1.]+[0.]*10)
    b=np.array([1.,1.,1.,1.,1.,-1.,-1.,-1.,-1.,-1.]+[0.]*10)
    for c in range(12):
        label='A' if c%2==0 else 'B'; split='train' if c<8 else 'val'; x=a if label=='A' else b
        for t,v in enumerate(x): rows.append((f'c{c}',float(t),split,v,0.5*v,label))
    return {
      'case_id':np.array([r[0] for r in rows]),'t_s':np.array([r[1] for r in rows]),'split':np.array([r[2] for r in rows]),
      'p1':np.array([r[3] for r in rows]),'p2':np.array([r[4] for r in rows]),'label':np.array([r[5] for r in rows]),
    }


def _write(path:Path,d):
    cols=list(d)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(cols)
        for i in range(len(d[cols[0]])): w.writerow([d[c][i] for c in cols])


def test_waveform_sequence_qualifies_feature_and_raw_representations():
    q=qualify_temporal_design(_spec(),_data())
    assert q.status=='READY'
    ids={x.id for x in q.candidates}
    assert ids=={'waveform_feature_linear','summary_mlp'}
    plan=qualify_model_families(_spec(),data=_data())
    fam={x.family.value:x.status for x in plan.families}
    assert fam['CLASSICAL_ML']=='CONDITIONAL'
    assert fam['DEEP_LEARNING']=='CONDITIONAL'


def test_waveform_roi_is_fail_closed_when_not_declared():
    s=_spec(); del s['modeling_contract']['constraints']['waveform_representation']['sample_window']
    q=qualify_temporal_design(s,_data())
    assert q.status in ('AMBIGUOUS','UNSUPPORTED')
    assert any('sample_window' in r for r in q.reasons)


def test_waveform_feature_candidate_can_win_and_predict(tmp_path:Path):
    s=_spec(); sp=tmp_path/'spec.json'; sp.write_text(json.dumps(s),encoding='utf-8')
    d=_data(); dp=tmp_path/'data.csv'; _write(dp,d)
    r=run_temporal_design(sp,dp,tmp_path/'out')
    assert r.status=='SELECTED'
    assert r.selected_candidate_id=='waveform_feature_linear'
    assert max(r.candidate_validation_risks['waveform_feature_linear']) < 1e-6
    pred=predict_temporal_design(tmp_path/'out',dp,tmp_path/'pred.csv')
    text=pred.read_text(encoding='utf-8')
    assert 'waveform_feature_linear__score__A' in text
    assert 'portfolio_argmax' in text
