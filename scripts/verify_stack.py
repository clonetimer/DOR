#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import hashlib, json

ROOT=Path(__file__).resolve().parents[1]

def sha256(p:Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    checks=[]
    wheels={
      'isgn':'isgn_toolkit-2.4.0+timer.1-py3-none-any.whl',
      'm2n':'m2n_toolkit-2.16.1+timer.1-py3-none-any.whl',
      'fec':'fec_toolkit-2.4.2+timer.1-py3-none-any.whl',
      'dor':'dor_diagnostic_stack-1.0.5+timer.1-py3-none-any.whl',
    }
    for k,n in wheels.items():
        p=ROOT/'wheels'/n; checks.append({'check':f'wheel:{k}','ok':p.is_file() and p.stat().st_size>0,'sha256':sha256(p) if p.is_file() else None})
    schemas=[
      'dataset_manifest_0.4.schema.json','reference_applicability_0.4.schema.json','task_qualification_0.4.schema.json',
      'diagnostic_plan_0.4.schema.json','diagnosis_bundle_0.4.schema.json',
      'fault_ontology_0.5.schema.json','mechanism_component_map_0.5.schema.json','localization_result_0.5.schema.json','localization_batch_0.5.schema.json','root_cause_result_0.6.schema.json',
      'fault_identification_request_0.7.schema.json','fault_identification_result_0.7.schema.json','isgn_identification_evidence_0.7.schema.json','isgn_mechanism_baseline_0.7.schema.json',
      'multitask_diagnosis_0.8.schema.json','vertical_validation_0.9.schema.json','development_freeze_gate_1.0.schema.json',
      'auto_design_request_1.0.schema.json','device_mechanism_card_1.0.schema.json',
      'multimodal_dataset_manifest_1.0.schema.json','public_diagnostic_dataset_catalog_1.0.schema.json',
      'isgn_active_discovery_advisory_1.0.schema.json','production_evidence_gate_1.0.schema.json',
      'prospective_benchmark_1.0.schema.json','deployment_ops_evidence_1.0.schema.json','public_source_intake_1.0.schema.json'
    ]
    for n in schemas:
        p=ROOT/'schemas'/n; checks.append({'check':f'schema:{n}','ok':p.is_file()})

    # v1.0.5 practical-evidence closure artifacts must preserve the distinction
    # between assisted POC support and production readiness.
    pe=ROOT/'release'/'PRODUCTION_EVIDENCE_STATUS.json'
    if pe.is_file():
        ped=json.loads(pe.read_text())
        profiles=ped.get('profiles',{})
        checks.append({'check':'production_evidence_semantics','ok':profiles.get('ASSISTED_POC',{}).get('status')=='SUPPORTED' and profiles.get('ASSISTED_PRODUCTION',{}).get('status')=='BLOCKED' and profiles.get('SAFETY_CRITICAL_AUTONOMY',{}).get('status')=='BLOCKED'})
    else:
        checks.append({'check':'production_evidence_semantics','ok':False,'reason':'release/PRODUCTION_EVIDENCE_STATUS.json missing'})

    oprem=ROOT/'validation'/'production_evidence'/'opssat_channelwise'/'summary.json'
    if oprem.is_file():
        opd=json.loads(oprem.read_text())
        checks.append({'check':'opssat_channelwise_remediation','ok':opd.get('selection_safety',{}).get('test_labels_used_for_candidate_selection') is False and opd.get('selection_safety',{}).get('therefore_this_is_development_remediation_not_a_fresh_blind_benchmark') is True})
    else:
        checks.append({'check':'opssat_channelwise_remediation','ok':False})

    hyd=ROOT/'validation'/'production_evidence'/'hydraulic_practical_stress.json'
    if hyd.is_file():
        hydj=json.loads(hyd.read_text())
        checks.append({'check':'hydraulic_practical_stress_semantics','ok':hydj.get('candidate_selection',{}).get('test_labels_used_for_selection') is False and hydj.get('evidence_scope',{}).get('synthetic_stress_is_not_real_distribution_shift_evidence') is True})
    else:
        checks.append({'check':'hydraulic_practical_stress_semantics','ok':False})

    # v1.0.5 Phase-2 evidence-integrity controls: good metrics do not repair
    # non-prospective evaluation, and selected benchmark segments do not become
    # operational exposure merely because a per-hour proxy can be calculated.
    prospective=ROOT/'validation'/'production_evidence'/'phase2'/'current_opssat_prospective_result.json'
    if prospective.is_file():
        pr=json.loads(prospective.read_text())
        checks.append({'check':'prospective_fail_closed_semantics','ok':pr.get('status')=='INELIGIBLE' and pr.get('production_gate_credit') is False and pr.get('metric_summary',{}).get('all_passed') is True})
    else:
        checks.append({'check':'prospective_fail_closed_semantics','ok':False})

    ops=ROOT/'validation'/'production_evidence'/'phase2'/'opssat_selected_segment_ops_result.json'
    if ops.is_file():
        opj=json.loads(ops.read_text())
        checks.append({'check':'selected_segment_ops_proxy_semantics','ok':opj.get('status')=='PROXY_ONLY' and opj.get('production_gate_credit') is False and opj.get('safety',{}).get('selected_segment_proxy_is_not_operational_false_alarms_per_hour') is True})
    else:
        checks.append({'check':'selected_segment_ops_proxy_semantics','ok':False})

    intake_specs=[
      'UCI_HYDRAULIC_447.json','NASA_ADAPT_DATASET_TEXT_ZIP.json',
      'NASA_ADAPT_EXPERIMENT_LOG_XLS.json','NASA_ADAPT_ADAPTHTMLDIAGRAMS_ZIP.json'
    ]
    checks.append({'check':'phase2_source_intake_specs','ok':all((ROOT/'release'/'source_intake_specs'/name).is_file() for name in intake_specs)})

    # v0.4 inherited real-domain regression remains in the release.
    op=json.loads((ROOT/'validation/opssat_ad/diagnosis_bundle.json').read_text())
    opa=json.loads((ROOT/'validation/opssat_ad/reference_applicability.json').read_text())
    checks.append({'check':'opssat_v04_bundle','ok':op.get('contract')=='DOR-DIAGNOSIS-BUNDLE/0.4' and op['diagnosis_summary']['supported_fraction']==523/529 and op['diagnosis_summary']['abstain_fraction']==6/529})
    checks.append({'check':'opssat_reference_power_gate','ok':opa['summary']['status_counts']['POWER_LIMITED']==6 and opa['summary']['status_counts']['HIERARCHY_SUPPORTED']==0})

    ac=json.loads((ROOT/'validation/hydraulic_accumulator/diagnosis_bundle.json').read_text())
    aca=json.loads((ROOT/'validation/hydraulic_accumulator/reference_applicability.json').read_text())
    checks.append({'check':'accumulator_v04_bundle','ok':ac.get('contract')=='DOR-DIAGNOSIS-BUNDLE/0.4' and ac['diagnosis_summary']['supported_fraction']==234/240 and ac['diagnosis_summary']['abstain_fraction']==6/240})
    checks.append({'check':'accumulator_declared_hierarchy','ok':aca['summary']['status_counts']['EXACT_SUPPORTED']==80 and aca['summary']['status_counts']['HIERARCHY_SUPPORTED']==160 and aca['safety']['implicit_pooling'] is False})

    # v0.5 real semantic handoff.
    hb=json.loads((ROOT/'validation/hydraulic_localization_v05/localization_batch.json').read_text())
    hs=json.loads((ROOT/'validation/hydraulic_localization_v05/posthoc_audit_summary.json').read_text())
    checks.append({'check':'hydraulic_v05_batch_contract','ok':hb.get('contract')=='DOR-LOCALIZATION-BATCH/0.5' and hb.get('n_cases')==29 and hb.get('status_counts',{}).get('SUPPORTED')==29})
    checks.append({'check':'hydraulic_v05_truth_separation','ok':hb.get('safety',{}).get('test_truth_used_for_reasoning') is False and hs.get('exact_fault_set_match_rate')==1.0})

    # NASA source-derived contract fixture exists and is explicitly scoped.
    no=json.loads((ROOT/'examples/nasa_adapt_published_fixture/fault_ontology.json').read_text())
    checks.append({'check':'nasa_fixture_ontology','ok':no.get('contract')=='DOR-FAULT-ONTOLOGY/0.5' and len(no.get('faults',[]))>=5 and 'source-derived' in str(no.get('scope_note','')).lower()})

    # v0.6 source-constrained root-cause software fixture.
    rc=json.loads((ROOT/'examples/nasa_adapt_v06_source_constrained_fixture/root_cause.json').read_text())
    checks.append({'check':'nasa_v06_rootcause_fixture','ok':rc.get('contract')=='DOR-ROOT-CAUSE-RESULT/0.6' and rc.get('status')=='SUPPORTED' and rc.get('root_cause_set')==['EY260_failed_open'] and rc.get('safety',{}).get('test_truth_used_for_reasoning') is False})

    excluded_roots={'.git','.venv','build','dist'}
    bad=[]
    for p in ROOT.rglob('*'):
        relative=p.relative_to(ROOT)
        if set(relative.parts) & excluded_roots:
            continue
        if p.name in {'.pytest_cache','__pycache__'} or p.suffix in {'.pyc','.pyo'}:
            bad.append(str(relative))
    checks.append({'check':'release_hygiene','ok':not bad,'bad':bad[:20]})
    ok=all(c['ok'] for c in checks)
    print(json.dumps({'status':'PASS' if ok else 'FAIL','checks':checks},indent=2,ensure_ascii=False))
    return 0 if ok else 1
if __name__=='__main__': raise SystemExit(main())
