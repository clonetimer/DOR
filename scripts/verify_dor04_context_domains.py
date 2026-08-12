#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))

def main():
    checks=[]
    # OPSSAT: strict exact channel; no hierarchy fallback
    op=ROOT/'validation/opssat_ad'
    q=load(op/'task_qualification.json'); p=load(op/'diagnostic_plan.json'); b=load(op/'diagnosis_bundle.json'); a=load(op/'reference_applicability.json')
    checks += [
      {'check':'opssat:qualification','ok':q['status']=='CONDITIONALLY_QUALIFIED','value':q['status']},
      {'check':'opssat:plan_contract','ok':p['contract']=='DOR-DIAGNOSTIC-PLAN/0.4','value':p['contract']},
      {'check':'opssat:bundle_contract','ok':b['contract']=='DOR-DIAGNOSIS-BUNDLE/0.4','value':b['contract']},
      {'check':'opssat:no_hierarchy_pooling','ok':a['summary']['status_counts']['HIERARCHY_SUPPORTED']==0},
      {'check':'opssat:power_limited','ok':a['summary']['status_counts']['POWER_LIMITED']==6,'value':a['summary']['status_counts']['POWER_LIMITED']},
    ]
    # Accumulator: explicit hierarchy exact -> pump_regime
    ac=ROOT/'validation/hydraulic_accumulator'
    q=load(ac/'task_qualification.json'); p=load(ac/'diagnostic_plan.json'); b=load(ac/'diagnosis_bundle.json'); a=load(ac/'reference_applicability.json')
    selected=pd.read_csv(ac/'fec/raw_plus_mechanism__hist_gradient_boosting/test_selected_reference.csv',dtype={'true_label':str})
    covered=[]
    for y,s in zip(selected['true_label'].astype(str),selected['prediction_set'].astype(str)):
        try: labs=set(map(str,json.loads(s)))
        except Exception: labs=set()
        covered.append(y in labs)
    selected['covered']=covered
    child=selected[(selected['cooler'].astype(str)=='100') & (selected['pump'].astype(str)=='0')]
    checks += [
      {'check':'acc:qualification','ok':q['status']=='CONDITIONALLY_QUALIFIED','value':q['status']},
      {'check':'acc:hierarchy_rows','ok':a['summary']['status_counts']['HIERARCHY_SUPPORTED']==160,'value':a['summary']['status_counts']['HIERARCHY_SUPPORTED']},
      {'check':'acc:exact_rows','ok':a['summary']['status_counts']['EXACT_SUPPORTED']==80,'value':a['summary']['status_counts']['EXACT_SUPPORTED']},
      {'check':'acc:decision_support','ok':b['diagnosis_summary']['supported_fraction']==234/240,'value':b['diagnosis_summary']['supported_fraction']},
      {'check':'acc:child_scope_warning_evidence','ok':len(child)==40 and abs(float(child.covered.mean())-0.85)<1e-12,'value':float(child.covered.mean()) if len(child) else None},
      {'check':'acc:no_test_driven_pooling','ok':a['safety']['test_labels_used'] is False and a['safety']['test_scores_used'] is False and a['safety']['implicit_pooling'] is False},
    ]
    ok=all(x['ok'] for x in checks)
    print(json.dumps({'status':'PASS' if ok else 'FAIL','checks':checks},indent=2,ensure_ascii=False))
    return 0 if ok else 1

if __name__=='__main__': raise SystemExit(main())
