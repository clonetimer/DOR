#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json, os, subprocess, sys, tempfile

ROOT=Path(__file__).resolve().parents[1]

def run(cmd):
    env=os.environ.copy(); env['PYTHONPATH']=str(ROOT/'src')
    return subprocess.run(cmd,cwd=ROOT,env=env,text=True,capture_output=True,check=True)

def main():
    checks=[]
    nasa=ROOT/'examples/nasa_adapt_published_fixture'
    with tempfile.TemporaryDirectory() as td:
        td=Path(td)
        proc=run([sys.executable,'-m','dor.cli','ontology-check','--ontology',str(nasa/'fault_ontology.json'),'--mechanism-map',str(nasa/'mechanism_component_map.json')])
        result=json.loads(proc.stdout)
        checks.append({'check':'nasa_ontology_check','ok':result.get('state')=='PASS'})
        expected={
          'evidence_relay_stuck_open.csv':'Relay1_stuckOpen',
          'evidence_e261_stuck.csv':'E261_sensor_stuck',
          'evidence_st516_offset.csv':'ST516_sensor_offset',
        }
        for fname,fid in expected.items():
            out=td/(fname+'.json')
            run([sys.executable,'-m','dor.cli','localize','--ontology',str(nasa/'fault_ontology.json'),'--mechanism-map',str(nasa/'mechanism_component_map.json'),'--evidence',str(nasa/fname),'--output',str(out)])
            x=json.loads(out.read_text())
            checks.append({'check':f'nasa:{fid}','ok':x.get('status')=='SUPPORTED' and fid in x.get('supported_fault_set',[]) and x.get('safety',{}).get('root_cause_claimed') is False and x.get('safety',{}).get('propagation_inferred') is False})

    hb=json.loads((ROOT/'validation/hydraulic_localization_v05/localization_batch.json').read_text())
    hs=json.loads((ROOT/'validation/hydraulic_localization_v05/posthoc_audit_summary.json').read_text())
    em=json.loads((ROOT/'examples/hydraulic_localization_real/evidence_manifest.json').read_text())
    checks.append({'check':'hydraulic:n_cases','ok':hb.get('n_cases')==29})
    checks.append({'check':'hydraulic:all_supported','ok':hb.get('status_counts')=={'SUPPORTED':29}})
    checks.append({'check':'hydraulic:no_truth_in_reasoning','ok':hb.get('safety',{}).get('test_truth_used_for_reasoning') is False and em.get('truth_used_for_qualification') is False})
    checks.append({'check':'hydraulic:posthoc_match','ok':hs.get('exact_fault_set_match_rate')==1.0})
    ok=all(x['ok'] for x in checks)
    print(json.dumps({'status':'PASS' if ok else 'FAIL','checks':checks},indent=2,ensure_ascii=False))
    return 0 if ok else 1
if __name__=='__main__': raise SystemExit(main())
