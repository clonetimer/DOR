#!/usr/bin/env python3
from pathlib import Path
import pandas as pd, json, hashlib

pump_path=Path('/mnt/data/dor03_real_pump/diagnosis.csv')
acc_path=Path('/mnt/data/dor04_real_hydraulic/diagnosis.csv')
outdir=Path(__file__).resolve().parent
p=pd.read_csv(pump_path,dtype={'sample_id':str,'true_label':str})
a=pd.read_csv(acc_path,dtype={'sample_id':str,'true_label':str})
x=p[['sample_id','diagnosis_status','diagnosis_set','true_label']].merge(
    a[['sample_id','diagnosis_status','diagnosis_set','true_label']],on='sample_id',suffixes=('_pump','_acc')
)
# Qualification is evidence-status based only; no truth label is consulted here.
q=x[(x.diagnosis_status_pump=='SUPPORTED') & (x.diagnosis_status_acc=='SUPPORTED')].copy()
rows=[]
for _,r in q.iterrows():
    for comp,scol in [('pump','diagnosis_set_pump'),('accumulator','diagnosis_set_acc')]:
        vals=json.loads(r[scol])
        if len(vals)!=1:
            continue
        rows.append({
            'case_id':r.sample_id,
            'evidence_id':f'{r.sample_id}:{comp}:class',
            'scope_type':'COMPONENT','scope_id':comp,'state':'CLASS','value':str(vals[0]),
            'strength':1.0,'validity':'VALID','provenance':f'calibrated DOR classification output ({comp})'
        })
e=pd.DataFrame(rows)
e.to_csv(outdir/'evidence.csv',index=False)
# Truth is kept in a separate post-hoc audit file and is never consumed by localize-batch.
truth=q[['sample_id','true_label_pump','true_label_acc']].rename(columns={'sample_id':'case_id'})
truth.to_csv(outdir/'truth_posthoc.csv',index=False)
manifest={
    'qualified_case_rule':'pump diagnosis_status == SUPPORTED AND accumulator diagnosis_status == SUPPORTED',
    'truth_used_for_qualification':False,'n_intersection':int(len(x)),'n_qualified':int(len(q)),
    'pump_source_sha256':hashlib.sha256(pump_path.read_bytes()).hexdigest(),
    'accumulator_source_sha256':hashlib.sha256(acc_path.read_bytes()).hexdigest(),
    'evidence_sha256':hashlib.sha256((outdir/'evidence.csv').read_bytes()).hexdigest(),
}
(outdir/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
