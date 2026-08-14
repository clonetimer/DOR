from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

from m2n.temporal_design import qualify_temporal_design, run_temporal_design, predict_temporal_design


def _spec():
    return {
      'schema_version':'2.14','template':'generic_mechanism_graph',
      'data':{'split_column':'split','train_values':['train'],'validation_values':['val'],'time_column':'t_s'},
      'qualification':{'independent_context_column':'case_id','evidence_scope':'BLINDED_FOR_VALIDATION'},
      'modeling_contract':{
        'objective':'CATEGORICAL_PREDICTION','input_representation':'WAVEFORM_SEQUENCE','mechanism_usage':'NONE',
        'prediction_unit':'INDEPENDENT_CONTEXT','input_columns':['vibration','rotation_hz'],'label_column':'label','time_column':'t_s',
        'required_outputs':['class_score'],
        'constraints':{
          'waveform_representation':{
            'sample_window':[0,256],'feature_set':'TIME_DOMAIN_V1','ridge_alpha':1.0,
            'order_domain':{
              'signal_columns':['vibration'],'rotation_frequency_column':'rotation_hz','sample_rate_hz':1000.0,
              'carrier_band_hz':[100.0,400.0],'characteristic_orders':[3.0,5.0],
              'harmonics':1,'sidebands':[0.0],'ridge_alpha':1.0,
            },
          },
          'temporal_candidates':['WAVEFORM_FEATURE_LINEAR','WAVEFORM_ORDER_LINEAR'],
          'temporal_training':{'epochs':2,'hidden_size':8,'seeds':[1,2,3],'lr':0.01},
        },
      },
      'selection':{'practical_ratio':0.8,'restart_stability':0.75},
    }


def _data():
    fs=1000.0; n=256; rows=[]
    for c in range(16):
        label='A' if c%2==0 else 'B'; order=3.0 if label=='A' else 5.0
        rot=10.0 if (c//2)%2==0 else 20.0; split='train' if c<12 else 'val'
        t=np.arange(n)/fs; rng=np.random.default_rng(100+c)
        sig=(1.0+0.65*np.sin(2*np.pi*order*rot*t+0.1*c))*np.sin(2*np.pi*200.0*t)+0.03*rng.normal(size=n)
        for i,x in enumerate(sig): rows.append((f'c{c}',t[i],split,float(x),rot,label))
    return {
      'case_id':np.array([r[0] for r in rows]),'t_s':np.array([r[1] for r in rows]),'split':np.array([r[2] for r in rows]),
      'vibration':np.array([r[3] for r in rows]),'rotation_hz':np.array([r[4] for r in rows]),'label':np.array([r[5] for r in rows]),
    }


def _write(path:Path,d):
    cols=list(d)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(cols)
        for i in range(len(d[cols[0]])): w.writerow([d[c][i] for c in cols])


def test_order_representation_qualifies_only_from_explicit_engineering_contract():
    q=qualify_temporal_design(_spec(),_data())
    assert q.status=='READY'
    assert {x.id for x in q.candidates}=={'waveform_feature_linear','waveform_order_linear'}


def test_order_representation_fails_closed_when_rotation_frequency_is_not_context_constant():
    d=_data(); idx=np.where(d['case_id']=='c0')[0]; d['rotation_hz'][idx[-1]]=11.0
    q=qualify_temporal_design(_spec(),d)
    assert q.status in ('AMBIGUOUS','UNSUPPORTED')
    assert any('rotation frequency must be finite, positive and constant' in r for r in q.reasons)


def test_order_representation_can_win_across_rotation_speeds_and_predict(tmp_path:Path):
    spec=_spec(); sp=tmp_path/'spec.json'; sp.write_text(json.dumps(spec),encoding='utf-8')
    d=_data(); dp=tmp_path/'data.csv'; _write(dp,d)
    r=run_temporal_design(sp,dp,tmp_path/'out')
    assert r.status=='SELECTED'
    assert r.selected_candidate_id=='waveform_order_linear'
    assert max(r.candidate_validation_risks['waveform_order_linear']) < 1e-6
    assert np.median(r.candidate_validation_risks['waveform_order_linear']) < np.median(r.candidate_validation_risks['waveform_feature_linear'])
    pred=predict_temporal_design(tmp_path/'out',dp,tmp_path/'pred.csv')
    text=pred.read_text(encoding='utf-8')
    assert 'waveform_order_linear__score__A' in text
    assert 'portfolio_argmax' in text
