from __future__ import annotations

import argparse
import json
from pathlib import Path

from .arbiter import arbitrate_anomaly_csv, arbitrate_anomaly_portfolio, arbitrate_class_portfolio
from .control import build_plan, formalize_manifest, load_json, make_effective_m2n_request, write_json
from .reference import applicability_dataframe, build_reference_applicability
from .executor import execute_chain
from .ontology import load_json as load_semantic_json, validate_fault_ontology, validate_mechanism_component_map
from .localization import localize_from_files, localize_batch_from_files
from .rootcause import infer_root_causes_from_files
from .identification import identify_from_files, build_identification_request_from_files
from .multitask import arbitrate_multitask_from_files
from .vertical import evaluate_vertical_from_file
from .freeze import evaluate_freeze_gate_from_files
from .isgn_identification_bridge import build_isgn_identification_evidence_from_files
from .stack import stack_info
from .auto_design import predict_pipeline, run_auto_design
from .dataset_registry import build_benchmark_plan, catalog_summary, default_catalog_path, load_catalog
from .public_adapters import prepare_spacecraft_dataset, prepare_uci_hydraulic
from .research_registry import default_method_catalog_path, load_method_catalog, method_catalog_summary
from .modality_calibration import assess_modality_mask_calibration


def main() -> int:
    p=argparse.ArgumentParser(prog='dor',description='DOR v0.7 development preview: automatic diagnostic pipeline design plus v0.6 localization/root-cause reasoning')
    sp=p.add_subparsers(dest='command',required=True)
    sp.add_parser('stack-info',help='show installed four-tool component versions')

    ad=sp.add_parser('auto-design',help='design a qualified detection/localization/classification/identification pipeline from device mechanisms and data')
    ad.add_argument('--device-card',required=True); ad.add_argument('--manifest',required=True); ad.add_argument('--data',required=True); ad.add_argument('--request',required=True); ad.add_argument('--out',required=True)
    ad.add_argument('--ontology',default=None); ad.add_argument('--resume',action='store_true')
    ai=sp.add_parser('auto-predict',help='run a frozen DOR auto-designed pipeline bundle')
    ai.add_argument('--bundle',required=True); ai.add_argument('--data',required=True); ai.add_argument('--output',required=True)

    dc=sp.add_parser('dataset-catalog',help='validate and summarize the evidence-qualified public diagnostic dataset catalog')
    dc.add_argument('--catalog',default=str(default_catalog_path()))
    rcatalog=sp.add_parser('method-catalog',help='validate and summarize the recent primary-source method evidence catalog')
    rcatalog.add_argument('--catalog',default=str(default_method_catalog_path()))
    bp=sp.add_parser('benchmark-plan',help='check whether a public dataset can evaluate requested diagnostic stages without overclaiming')
    bp.add_argument('--catalog',default=str(default_catalog_path()))
    bp.add_argument('--dataset-id',required=True); bp.add_argument('--stages',nargs='+',required=True); bp.add_argument('--output',default=None)
    uh=sp.add_parser('uci-hydraulic-prepare',help='convert the official UCI Hydraulic ZIP into a provenance-bound multirate DOR dataset')
    uh.add_argument('--archive',required=True); uh.add_argument('--out',required=True); uh.add_argument('--sensors',nargs='*',default=None); uh.add_argument('--seed',type=int,default=447); uh.add_argument('--block-size',type=int,default=5)
    sc=sp.add_parser('spacecraft-prepare',help='stream a sat-sim spacecraft file tree into a leakage-bounded DOR table')
    sc.add_argument('--dataset-root',required=True); sc.add_argument('--out',required=True); sc.add_argument('--feature-profile',choices=['causal_raw','deployable_full'],default='causal_raw')
    mf=sp.add_parser('mask-calibration-plan',help='audit whether calibration data can support declared modality-presence masks')
    mf.add_argument('--modalities',nargs='+',required=True); mf.add_argument('--calibration-rows',type=int,required=True); mf.add_argument('--min-group-size',type=int,default=20); mf.add_argument('--output',default=None)

    mc=sp.add_parser('manifest-check',help='formalize/validate a DatasetManifest and bind it to a concrete data artifact')
    mc.add_argument('--manifest',required=True); mc.add_argument('--request',required=True); mc.add_argument('--data',required=True); mc.add_argument('--output',required=True)

    rp=sp.add_parser('reference-plan',help='compute v0.4 reference applicability from declared context hierarchy without running M2N/FEC')
    rp.add_argument('--manifest',required=True); rp.add_argument('--request',required=True); rp.add_argument('--data',required=True); rp.add_argument('--out',required=True)

    pl=sp.add_parser('plan',help='build DOR TaskQualification + engine admissions + DiagnosticPlan without executing engines')
    pl.add_argument('--manifest',required=True); pl.add_argument('--request',required=True); pl.add_argument('--data',required=True); pl.add_argument('--out',required=True)

    run=sp.add_parser('run',help='one-command replay: qualify -> plan -> M2N -> FEC -> DOR -> DiagnosisBundle')
    run.add_argument('--manifest',required=True); run.add_argument('--request',required=True); run.add_argument('--data',required=True); run.add_argument('--out',required=True); run.add_argument('--dry-run',action='store_true'); run.add_argument('--resume',action='store_true',help='reuse an existing M2N bundle only when request/data SHA-256 bindings match')

    aa=sp.add_parser('anomaly-arbitrate',help='convert one FEC anomaly output into DOR supported/abstain diagnosis rows')
    aa.add_argument('--fec-output',required=True); aa.add_argument('--output',required=True); aa.add_argument('--summary',required=True)
    ap=sp.add_parser('anomaly-portfolio-arbitrate',help='conservative DOR arbitration over multiple FEC-calibrated anomaly detectors')
    ap.add_argument('--fec-output',action='append',required=True); ap.add_argument('--output',required=True); ap.add_argument('--summary',required=True); ap.add_argument('--id-column',default=None)
    oc=sp.add_parser('ontology-check',help='validate DOR v0.5 Fault Ontology and optional Mechanism-to-Component Map')
    oc.add_argument('--ontology',required=True); oc.add_argument('--mechanism-map',default=None); oc.add_argument('--output',default=None)
    loc=sp.add_parser('localize',help='localize candidate component/fault hypotheses from validated evidence without claiming root cause')
    loc.add_argument('--ontology',required=True); loc.add_argument('--evidence',required=True); loc.add_argument('--mechanism-map',default=None); loc.add_argument('--output',required=True)
    loc.add_argument('--min-support',type=float,default=0.45); loc.add_argument('--retain-margin',type=float,default=0.08); loc.add_argument('--max-contradiction-fraction',type=float,default=0.35)
    lb=sp.add_parser('localize-batch',help='batch v0.5 localization for evidence grouped by case_id; test truth is not consumed')
    lb.add_argument('--ontology',required=True); lb.add_argument('--evidence',required=True); lb.add_argument('--mechanism-map',default=None); lb.add_argument('--output',required=True); lb.add_argument('--case-column',default='case_id')
    lb.add_argument('--min-support',type=float,default=0.45); lb.add_argument('--retain-margin',type=float,default=0.08); lb.add_argument('--max-contradiction-fraction',type=float,default=0.35)
    rc=sp.add_parser('root-cause',help='v0.6 prototype: infer root-cause candidate sets only with timestamped fault-specific support and provenance-bound propagation paths')
    rc.add_argument('--ontology',required=True); rc.add_argument('--localization',required=True); rc.add_argument('--evidence',required=True); rc.add_argument('--mechanism-map',default=None); rc.add_argument('--output',required=True)
    rc.add_argument('--max-root-set-size',type=int,default=3); rc.add_argument('--coverage-required',type=float,default=1.0); rc.add_argument('--alternative-margin',type=float,default=0.05); rc.add_argument('--temporal-tolerance',type=float,default=0.0); rc.add_argument('--max-path-depth',type=int,default=8)
    ip=sp.add_parser('identify-plan',help='create a SHA-bound identification request and decide whether ISGN differential identification is required')
    ip.add_argument('--ontology',required=True); ip.add_argument('--localization',required=True); ip.add_argument('--root-cause',default=None); ip.add_argument('--case-id',default=None); ip.add_argument('--window-start',type=float,default=None); ip.add_argument('--window-end',type=float,default=None); ip.add_argument('--window-validity',default='VALID'); ip.add_argument('--output',required=True)
    ib=sp.add_parser('isgn-identification',help='run the bounded ISGN differential adapter on an admitted baseline and abnormal dataset')
    ib.add_argument('--mechanism-spec',required=True); ib.add_argument('--normal',required=True); ib.add_argument('--abnormal',required=True); ib.add_argument('--ontology',required=True); ib.add_argument('--mechanism-map',required=True); ib.add_argument('--window-validity',default='VALID'); ib.add_argument('--output',required=True)
    ident=sp.add_parser('identify',help='identify discrete/sensor changes or consume qualified ISGN differential evidence')
    ident.add_argument('--ontology',required=True); ident.add_argument('--localization',required=True); ident.add_argument('--evidence',required=True); ident.add_argument('--root-cause',default=None); ident.add_argument('--isgn-identification',default=None); ident.add_argument('--output',required=True)
    mt=sp.add_parser('multi-task-arbitrate',help='cross-check detection/classification/localization/root-cause/identification evidence')
    mt.add_argument('--detection',default=None); mt.add_argument('--classification',default=None); mt.add_argument('--localization',default=None); mt.add_argument('--root-cause',default=None); mt.add_argument('--identification',default=None); mt.add_argument('--case-id',default=None); mt.add_argument('--output',required=True)
    vv=sp.add_parser('vertical-validate',help='evaluate domain/task evidence grades without upgrading cross-domain evidence')
    vv.add_argument('--registry',required=True); vv.add_argument('--output',required=True)
    fg=sp.add_parser('freeze-gate',help='evaluate explicit development-exit gates')
    fg.add_argument('--evidence',required=True); fg.add_argument('--vertical',required=True); fg.add_argument('--output',required=True)
    ca=sp.add_parser('class-arbitrate',help='conservative arbitration over one or more FEC class prediction-set CSVs')
    ca.add_argument('--fec-output',action='append',required=True,help='repeat for each retained model')
    ca.add_argument('--output',required=True); ca.add_argument('--summary',required=True); ca.add_argument('--id-column',default='sample_id'); ca.add_argument('--true-label-column',default='true_label')
    args=p.parse_args()

    if args.command=='stack-info':
        print(json.dumps(stack_info(),indent=2)); return 0
    if args.command=='auto-design':
        result=run_auto_design(args.device_card,args.manifest,args.data,args.request,args.out,ontology_path=args.ontology,resume=args.resume)
        print(json.dumps({'state':result['status'],'device_id':result['device_id'],'candidate_count':result['candidate_count'],'out':str(Path(args.out).resolve())},indent=2)); return 0 if result['status']!='UNSUPPORTED' else 2
    if args.command=='auto-predict':
        output=predict_pipeline(args.bundle,args.data,args.output)
        print(json.dumps({'state':'OK','output':str(output.resolve())},indent=2)); return 0
    if args.command=='dataset-catalog':
        print(json.dumps(catalog_summary(load_catalog(args.catalog)),indent=2)); return 0
    if args.command=='method-catalog':
        print(json.dumps(method_catalog_summary(load_method_catalog(args.catalog)),indent=2)); return 0
    if args.command=='benchmark-plan':
        result=build_benchmark_plan(load_catalog(args.catalog),args.dataset_id,args.stages)
        if args.output: write_json(args.output,result)
        print(json.dumps(result,indent=2)); return 0 if result['state']!='BLOCKED' else 2
    if args.command=='uci-hydraulic-prepare':
        result=prepare_uci_hydraulic(args.archive,args.out,sensors=args.sensors,seed=args.seed,block_size=args.block_size)
        print(json.dumps({'state':'OK','binding':result,'out':str(Path(args.out).resolve())},indent=2)); return 0
    if args.command=='spacecraft-prepare':
        result=prepare_spacecraft_dataset(args.dataset_root,args.out,feature_profile=args.feature_profile)
        print(json.dumps({'state':'OK','binding':result,'out':str(Path(args.out).resolve())},indent=2)); return 0
    if args.command=='mask-calibration-plan':
        result=assess_modality_mask_calibration(args.modalities,calibration_rows=args.calibration_rows,min_group_size=args.min_group_size).to_dict()
        if args.output: write_json(args.output,result)
        print(json.dumps(result,indent=2)); return 0 if result['status']!='INSUFFICIENT_FOR_EXHAUSTIVE_MASKS' else 2
    if args.command=='manifest-check':
        m=formalize_manifest(load_json(args.manifest),load_json(args.request),args.data); write_json(args.output,m); print(json.dumps({'state':'OK','contract':m['contract'],'output':str(Path(args.output).resolve())},indent=2)); return 0
    if args.command=='reference-plan':
        out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
        m=formalize_manifest(load_json(args.manifest),load_json(args.request),args.data)
        app=build_reference_applicability(m,args.data,id_column=(m.get('identifiers') or [None])[0])
        write_json(out/'dataset_manifest.json',m); write_json(out/'reference_applicability.json',app); applicability_dataframe(app).to_csv(out/'reference_applicability.csv',index=False)
        print(json.dumps({'state':'OK','policy':app['policy'],'summary':app['summary'],'out':str(out.resolve())},indent=2)); return 0
    if args.command=='plan':
        out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
        rawm=load_json(args.manifest); rawr=load_json(args.request); m=formalize_manifest(rawm,rawr,args.data); r=make_effective_m2n_request(rawr,m)
        write_json(out/'dataset_manifest.json',m); write_json(out/'m2n_request_effective.json',r)
        plan=build_plan(m,r,args.data,out); write_json(out/'task_qualification.json',plan.qualification.to_dict()); write_json(out/'engine_admissions.json',[x.to_dict() for x in plan.engine_admissions]); write_json(out/'diagnostic_plan.json',plan.to_dict())
        app=build_reference_applicability(m,args.data,id_column=(m.get('identifiers') or [None])[0]); write_json(out/'reference_applicability.json',app); applicability_dataframe(app).to_csv(out/'reference_applicability.csv',index=False)
        print(json.dumps({'state':'OK','qualification':plan.qualification.status,'reference_summary':app['summary'],'plan':str((out/'diagnostic_plan.json').resolve())},indent=2)); return 0 if plan.qualification.status!='UNSUPPORTED' else 2
    if args.command=='run':
        report=execute_chain(args.manifest,args.request,args.data,args.out,dry_run=args.dry_run,resume=args.resume); print(json.dumps({'state':report['status'],'plan_id':report['plan_id'],'out':str(Path(args.out).resolve())},indent=2)); return 0 if report['status'] in {'COMPLETED','PLANNED'} else 2
    if args.command=='ontology-check':
        o=load_semantic_json(args.ontology); ov=validate_fault_ontology(o); result={'ontology':ov.to_dict()}
        if args.mechanism_map:
            mv=validate_mechanism_component_map(load_semantic_json(args.mechanism_map),o); result['mechanism_component_map']=mv.to_dict()
        result['state']='PASS' if all(v.get('state')=='PASS' for k,v in result.items() if isinstance(v,dict) and 'state' in v) else 'FAIL'
        if args.output: write_json(args.output,result)
        print(json.dumps(result,indent=2)); return 0 if result['state']=='PASS' else 2
    if args.command=='localize':
        result=localize_from_files(args.ontology,args.evidence,mechanism_map_path=args.mechanism_map,min_support=args.min_support,retain_margin=args.retain_margin,max_contradiction_fraction=args.max_contradiction_fraction)
        write_json(args.output,result); print(json.dumps({'state':'OK','localization_status':result['status'],'output':str(Path(args.output).resolve())},indent=2)); return 0
    if args.command=='localize-batch':
        result=localize_batch_from_files(args.ontology,args.evidence,mechanism_map_path=args.mechanism_map,case_column=args.case_column,min_support=args.min_support,retain_margin=args.retain_margin,max_contradiction_fraction=args.max_contradiction_fraction)
        write_json(args.output,result); print(json.dumps({'state':'OK','n_cases':result['n_cases'],'status_counts':result['status_counts'],'output':str(Path(args.output).resolve())},indent=2)); return 0
    if args.command=='root-cause':
        result=infer_root_causes_from_files(args.ontology,args.localization,args.evidence,mechanism_map_path=args.mechanism_map,max_root_set_size=args.max_root_set_size,coverage_required=args.coverage_required,alternative_margin=args.alternative_margin,temporal_tolerance=args.temporal_tolerance,max_path_depth=args.max_path_depth)
        write_json(args.output,result); print(json.dumps({'state':'OK','root_cause_status':result['status'],'root_cause_set':result.get('root_cause_set',[]),'output':str(Path(args.output).resolve())},indent=2)); return 0
    if args.command=='identify-plan':
        result=build_identification_request_from_files(args.ontology,args.localization,root_cause_path=args.root_cause,case_id=args.case_id,window_start=args.window_start,window_end=args.window_end,window_evidence_validity=args.window_validity)
        write_json(args.output,result); print(json.dumps({'state':'OK','request_status':result['status'],'isgn_admission':result.get('engine_admissions',{}).get('ISGN'),'output':str(Path(args.output).resolve())},indent=2)); return 0 if result['status']!='BLOCKED' else 2
    if args.command=='isgn-identification':
        result=build_isgn_identification_evidence_from_files(args.mechanism_spec,args.normal,args.abnormal,args.ontology,args.mechanism_map,window_evidence_validity=args.window_validity)
        write_json(args.output,result); print(json.dumps({'state':'OK','isgn_identification_status':result['status'],'supported_changes':sum(bool(x.get('supported')) for x in result.get('changes',[])),'output':str(Path(args.output).resolve())},indent=2)); return 0
    if args.command=='identify':
        result=identify_from_files(args.ontology,args.localization,args.evidence,root_cause_path=args.root_cause,isgn_identification_path=args.isgn_identification)
        write_json(args.output,result); print(json.dumps({'state':'OK','identification_status':result['status'],'n_identifications':len(result.get('identifications',[])),'output':str(Path(args.output).resolve())},indent=2)); return 0
    if args.command=='anomaly-arbitrate':
        arbitrate_anomaly_csv(args.fec_output,args.output,args.summary); print(json.dumps({'state':'OK','output':str(Path(args.output).resolve()),'summary':str(Path(args.summary).resolve())},indent=2)); return 0
    if args.command=='anomaly-portfolio-arbitrate':
        arbitrate_anomaly_portfolio(args.fec_output,args.output,args.summary,id_column=args.id_column); print(json.dumps({'state':'OK','output':str(Path(args.output).resolve()),'summary':str(Path(args.summary).resolve())},indent=2)); return 0
    if args.command=='multi-task-arbitrate':
        result=arbitrate_multitask_from_files(detection_path=args.detection,classification_path=args.classification,localization_path=args.localization,root_cause_path=args.root_cause,identification_path=args.identification,case_id=args.case_id)
        write_json(args.output,result); print(json.dumps({'state':'OK','multitask_status':result['status'],'diagnostic_depth':result['diagnostic_depth'],'output':str(Path(args.output).resolve())},indent=2)); return 0 if result['status']!='ABSTAIN' else 2
    if args.command=='vertical-validate':
        result=evaluate_vertical_from_file(args.registry); write_json(args.output,result); print(json.dumps({'state':'OK','vertical_status':result['status'],'missing_gates':result['missing_gates'],'output':str(Path(args.output).resolve())},indent=2)); return 0 if result['status']!='BLOCKED' else 2
    if args.command=='freeze-gate':
        result=evaluate_freeze_gate_from_files(args.evidence,args.vertical); write_json(args.output,result); print(json.dumps({'state':'OK','freeze_status':result['status'],'blocking_gates':result['blocking_gates'],'output':str(Path(args.output).resolve())},indent=2)); return 0 if result['status']=='READY_FOR_1_0_FREEZE' else 3
    if args.command=='class-arbitrate':
        arbitrate_class_portfolio(args.fec_output,args.output,args.summary,id_column=args.id_column,true_label_column=args.true_label_column); print(json.dumps({'state':'OK','output':str(Path(args.output).resolve()),'summary':str(Path(args.summary).resolve())},indent=2)); return 0
    return 2


if __name__=='__main__':
    raise SystemExit(main())
