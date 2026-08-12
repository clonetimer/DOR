from pathlib import Path
import pandas as pd, json
from dor.arbiter import arbitrate_anomaly_csv, arbitrate_class_portfolio

def test_anomaly_unsupported_becomes_abstain(tmp_path:Path):
    inp=tmp_path/'f.csv'; pd.DataFrame([{'id':1,'decision':'NORMAL','calibration_supported':True},{'id':2,'decision':'CONTEXT_UNSUPPORTED','calibration_supported':False}]).to_csv(inp,index=False)
    out=tmp_path/'o.csv'; summ=tmp_path/'s.json'; arbitrate_anomaly_csv(inp,out,summ); d=pd.read_csv(out)
    assert list(d.diagnosis_status)==['SUPPORTED','ABSTAIN']

def test_class_portfolio_preserves_alternatives(tmp_path:Path):
    a=tmp_path/'a.csv'; b=tmp_path/'b.csv'
    pd.DataFrame([{'sample_id':'x','prediction_set':'["A"]','calibration_supported':True,'true_label':'A'}]).to_csv(a,index=False)
    pd.DataFrame([{'sample_id':'x','prediction_set':'["B"]','calibration_supported':True,'true_label':'A'}]).to_csv(b,index=False)
    out=tmp_path/'o.csv'; summ=tmp_path/'s.json'; arbitrate_class_portfolio([a,b],out,summ); d=pd.read_csv(out)
    assert d.diagnosis_status.iloc[0]=='ALTERNATIVES'; assert set(json.loads(d.diagnosis_set.iloc[0]))=={'A','B'}
