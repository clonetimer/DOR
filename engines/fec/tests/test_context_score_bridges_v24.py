from pathlib import Path
import csv, json

from fec.extensions.anomaly_score_bridge import fit_anomaly_score_reference, infer_anomaly_decisions
from fec.extensions.context_class_score_bridge import fit_context_class_score_reference, infer_context_class_prediction_sets


def test_anomaly_context_exact_groups_and_unseen_abstain(tmp_path: Path):
    cal=tmp_path/'cal.csv'
    with cal.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['id','channel','anomaly_score','true_label']); w.writeheader()
        for ch,off in [('A',0.0),('B',1.0)]:
            for i in range(25): w.writerow({'id':f'{ch}{i}','channel':ch,'anomaly_score':off+i/100,'true_label':0})
    ref=tmp_path/'ref.json'; fit_anomaly_score_reference(cal,ref,context_columns='channel',alpha=.1,min_group_size=20)
    payload=json.loads(ref.read_text()); assert payload['contract']=='FEC-ANOMALY-CONTEXT-REFERENCE/1.0'
    assert payload['context_columns']==['channel']
    score=tmp_path/'score.csv'
    with score.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['id','channel','anomaly_score']); w.writeheader()
        w.writerow({'id':'1','channel':'A','anomaly_score':99}); w.writerow({'id':'2','channel':'C','anomaly_score':99})
    out=tmp_path/'out.csv'; infer_anomaly_decisions(score,ref,out); rows=list(csv.DictReader(out.open()))
    assert rows[0]['calibration_supported']=='True' and rows[0]['decision']=='ABNORMAL'
    assert rows[1]['calibration_supported']=='False' and rows[1]['decision']=='CONTEXT_UNSUPPORTED'


def _class_rows(path:Path, groups, include_label=True):
    fields=['id','ctx','score__N','score__F']+(['true_label'] if include_label else [])
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); i=0
        for ctx,n in groups:
            for j in range(n):
                y='N' if j%2==0 else 'F'; row={'id':i,'ctx':ctx,'score__N':.9 if y=='N' else .1,'score__F':.1 if y=='N' else .9}
                if include_label: row['true_label']=y
                w.writerow(row); i+=1


def test_context_class_reference_and_sparse_context_abstain(tmp_path:Path):
    cal=tmp_path/'cal.csv'; _class_rows(cal,[('A',40),('B',10)],True)
    ref=tmp_path/'ref.json'; fit_context_class_score_reference(cal,ref,context_columns='ctx',min_group_size=20)
    payload=json.loads(ref.read_text()); assert payload['contract']=='FEC-CONTEXT-CLASS-CONFORMAL-REFERENCE/1.0'
    groups={tuple(g['context']):g for g in payload['groups']}; assert groups[('A',)]['supported'] is True; assert groups[('B',)]['supported'] is False
    score=tmp_path/'score.csv'; _class_rows(score,[('A',4),('B',4),('C',4)],False)
    out=tmp_path/'out.csv'; infer_context_class_prediction_sets(score,ref,out); rows=list(csv.DictReader(out.open()))
    by={}
    for r in rows: by.setdefault(r['ctx'],[]).append(r)
    assert all(r['calibration_supported']=='True' for r in by['A'])
    assert all(r['calibration_supported']=='False' and r['prediction_set_status']=='ABSTAIN' for r in by['B']+by['C'])
