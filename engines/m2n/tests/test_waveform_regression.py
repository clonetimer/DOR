from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

from m2n.temporal_design import qualify_temporal_design, run_temporal_design, predict_temporal_design
from m2n.model_family import qualify_model_families


def _spec():
    return {
      'schema_version':'2.12','template':'generic_mechanism_graph',
      'data':{'split_column':'split','train_values':['train'],'validation_values':['val'],'time_column':'t_s'},
      'qualification':{'independent_context_column':'case_id','evidence_scope':'BLINDED_FOR_VALIDATION'},
      'modeling_contract':{
        'objective':'STATIC_REGRESSION','input_representation':'WAVEFORM_SEQUENCE','mechanism_usage':'NONE',
        'prediction_unit':'INDEPENDENT_CONTEXT','input_columns':['p1','p2'],'target_columns':['gain'],'time_column':'t_s',
        'required_outputs':['prediction'],
        'constraints':{
          'waveform_representation':{'sample_window':[0,12],'feature_set':'TIME_DOMAIN_V1','ridge_alpha':0.01},
          'temporal_candidates':['WAVEFORM_FEATURE_LINEAR','SUMMARY_MLP'],
          'temporal_training':{'epochs':2,'hidden_size':4,'seeds':[1,2,3],'lr':0.01}
        }
      },
      'selection':{'practical_ratio':0.8,'restart_stability':0.75},
    }


def _data():
    rows=[]
    # Target is encoded in deterministic ROI amplitude; contexts remain independent.
    for c,g in enumerate(np.linspace(0.5,4.0,14)):
        split='train' if c<10 else 'val'
        wave=np.r_[np.tile([g,-g],6),np.zeros(8)]
        for t,v in enumerate(wave): rows.append((f'c{c}',float(t),split,float(v),float(0.3*v),float(g)))
    return {
      'case_id':np.array([r[0] for r in rows]),'t_s':np.array([r[1] for r in rows]),'split':np.array([r[2] for r in rows]),
      'p1':np.array([r[3] for r in rows]),'p2':np.array([r[4] for r in rows]),'gain':np.array([r[5] for r in rows]),
    }


def _write(path:Path,d):
    cols=list(d)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(cols)
        for i in range(len(d[cols[0]])): w.writerow([d[c][i] for c in cols])


def test_waveform_static_regression_qualifies_feature_and_raw_candidates():
    q=qualify_temporal_design(_spec(),_data())
    assert q.status=='READY'
    assert {x.id for x in q.candidates}=={'waveform_feature_linear','summary_mlp'}
    plan=qualify_model_families(_spec(),data=_data())
    fam={x.family.value:x.status for x in plan.families}
    assert fam['CLASSICAL_ML']=='CONDITIONAL'
    assert fam['DEEP_LEARNING']=='CONDITIONAL'


def test_waveform_static_regression_design_and_prediction(tmp_path:Path):
    sp=tmp_path/'spec.json'; sp.write_text(json.dumps(_spec()),encoding='utf-8')
    d=_data(); dp=tmp_path/'data.csv'; _write(dp,d)
    r=run_temporal_design(sp,dp,tmp_path/'out')
    assert r.status in ('SELECTED','AMBIGUOUS')
    assert 'waveform_feature_linear' in r.candidate_validation_risks
    assert max(r.candidate_validation_risks['waveform_feature_linear']) < 0.05
    pred=predict_temporal_design(tmp_path/'out',dp,tmp_path/'pred.csv')
    text=pred.read_text(encoding='utf-8')
    assert 'portfolio_prediction__gain' in text
    assert 'actual__gain' in text


def test_waveform_static_regression_requires_context_constant_targets(tmp_path:Path):
    s=_spec(); sp=tmp_path/'spec.json'; sp.write_text(json.dumps(s),encoding='utf-8')
    d=_data().copy(); d['gain']=d['gain'].copy(); d['gain'][1]+=0.1
    dp=tmp_path/'data.csv'; _write(dp,d)
    try:
        run_temporal_design(sp,dp,tmp_path/'out')
    except ValueError as e:
        assert 'constant within context' in str(e)
    else:
        raise AssertionError('expected fail-closed target consistency error')
