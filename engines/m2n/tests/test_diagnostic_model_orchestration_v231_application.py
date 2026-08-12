from pathlib import Path
import json
import numpy as np
import pandas as pd

from m2n.diagnostic_modeling import run_diagnostic_model_design


def test_mlp_supports_string_fault_labels(tmp_path: Path):
    rng=np.random.default_rng(11)
    n=240
    x=rng.normal(size=(n,2))
    y=np.where(x[:,0]+0.3*x[:,1]>0,'FAULT_A','NORMAL')
    split=np.array(['train']*120+['val']*60+['cal']*30+['test']*30)
    df=pd.DataFrame({'sample_id':np.arange(n),'x1':x[:,0],'x2':x[:,1],'fault_label':y,'split':split})
    dp=tmp_path/'d.csv'; df.to_csv(dp,index=False)
    req={
      'contract':'M2N-DIAGNOSTIC-MODEL-REQUEST/1.0','task':'FAULT_CLASSIFICATION',
      'data':{'split_column':'split','train_values':['train'],'validation_values':['val'],'calibration_values':['cal'],'test_values':['test'],'id_columns':['sample_id'],'include_test_labels':True},
      'features':{'raw_columns':['x1','x2'],'views':['RAW']},
      'supervision':{'label_column':'fault_label'},
      'candidate_policy':{'families':['mlp']},'training':{'restarts':[1,2],'final_seed':3},
      'selection':{'primary_metric':'macro_f1','practical_margin':0.02,'restart_stability':0.5}
    }
    rp=tmp_path/'r.json'; rp.write_text(json.dumps(req))
    out=tmp_path/'out'; r=run_diagnostic_model_design(rp,dp,out)
    assert r.status=='FIXED'
    metrics=json.loads((out/'validation_metrics.json').read_text())
    assert metrics['failures']['raw__mlp']==[]
    assert (out/'scores/calibration_scores.csv').exists()


def test_ambiguous_freezes_only_practical_equivalence_set_and_exports_each_scores(tmp_path: Path):
    rng=np.random.default_rng(7)
    n=320
    y=np.array(['N','F']*(n//2))
    raw=rng.normal(size=n)
    mech=(y=='F').astype(float)+rng.normal(0,0.01,size=n)
    split=np.array(['train']*160+['val']*80+['cal']*40+['test']*40)
    df=pd.DataFrame({'sample_id':np.arange(n),'raw':raw,'mech':mech,'fault_label':y,'split':split})
    dp=tmp_path/'d.csv'; df.to_csv(dp,index=False)
    req={
      'contract':'M2N-DIAGNOSTIC-MODEL-REQUEST/1.0','task':'FAULT_CLASSIFICATION',
      'data':{'split_column':'split','train_values':['train'],'validation_values':['val'],'calibration_values':['cal'],'test_values':['test'],'id_columns':['sample_id'],'include_test_labels':True},
      'features':{'raw_columns':['raw'],'mechanism_columns':['mech'],'views':['RAW','MECHANISM','RAW_PLUS_MECHANISM']},
      'supervision':{'label_column':'fault_label'},
      'candidate_policy':{'families':['hist_gradient_boosting']},'training':{'restarts':[1,2,3],'final_seed':4},
      'selection':{'primary_metric':'macro_f1','practical_margin':0.02,'restart_stability':0.75}
    }
    rp=tmp_path/'r.json'; rp.write_text(json.dumps(req))
    out=tmp_path/'out'; r=run_diagnostic_model_design(rp,dp,out)
    assert r.status=='AMBIGUOUS'
    b=json.loads((out/'diagnostic_model_bundle.json').read_text())
    assert set(b['retained_candidate_ids'])=={'mechanism__hist_gradient_boosting','raw_plus_mechanism__hist_gradient_boosting'}
    assert 'raw__hist_gradient_boosting' not in b['retained_candidate_ids']
    for cid in b['retained_candidate_ids']:
        assert (out/b['score_artifacts'][cid]['calibration']).exists()
        assert (out/b['score_artifacts'][cid]['test']).exists()
        assert (out/b['score_artifacts'][cid]['contract']).exists()
