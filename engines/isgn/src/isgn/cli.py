from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from .io import save_npz, load_npz, save_algebraic_npz, load_algebraic_npz
from .api import recover_graph, complete_graph, certify_completion, calibrate_completion_coverage, save_checkpoint, load_checkpoint, fit_normal_phm, extract_mechanism_features
from .symbolic import export_explicit_relation, validate_expression
from .fusion import build_operational_model
from .consensus import build_consensus_model
from .fallbacks import fit_linear_state_space_fallback, save_fallback
from .backends import backend_status, backend_manifest, SymMatikaExternalBackend
from .observation_gate import assess_observation
from .observation_scope import ObservationCandidate, compare_observation_candidates, project_state_scope
from .mechanism_scope import project_mechanism_scope
from .friction import profile_smooth_stribeck
from .phm import mechanism_health_evidence, calibrate_phm_health, score_phm_health
from .mechanism_recovery import recover_missing_terms
from .physics_reducer import reduce_algebraic_search_space
from .trajectory_recovery import recover_missing_dynamics_trajectory

def _dump(obj,path=None):
    txt=json.dumps(obj,indent=2,ensure_ascii=False,default=lambda x: float(x) if hasattr(x,'item') else str(x))
    if path: Path(path).write_text(txt,encoding='utf-8')
    print(txt)

def _backend_options(a):
    if not getattr(a,'backend_options',None): return {}
    text=a.backend_options
    p=Path(text)
    if p.exists(): return json.loads(p.read_text(encoding='utf-8'))
    return json.loads(text)

def cmd_doctor(a):
    import scipy, sklearn
    import isgn
    out={"isgn":isgn.__version__,"python":sys.version.split()[0],"torch":torch.__version__,
         "numpy":np.__version__,"cuda":bool(torch.cuda.is_available()),
         "backends":{"native":True,**backend_status()},"status":"PASS"}
    _dump(out)

def cmd_backends(a): _dump(backend_manifest())

def cmd_observe(a):
    ds=load_npz(a.data)
    rep=assess_observation(ds)
    _dump(rep.to_dict(),a.out)




def _named_npz(items, *, production_eligible):
    out=[]
    for item in items or []:
        if '=' not in item:
            raise ValueError(f"expected NAME=PATH, got {item!r}")
        name,path=item.split('=',1)
        name=name.strip(); path=path.strip()
        if not name or not path:
            raise ValueError(f"expected NAME=PATH, got {item!r}")
        out.append(ObservationCandidate(name,load_npz(path),production_eligible=production_eligible))
    return out

def cmd_observe_compare(a):
    candidates=_named_npz(a.candidate,production_eligible=True)+_named_npz(a.audit_only,production_eligible=False)
    rep=compare_observation_candidates(candidates)
    _dump(rep.to_dict(),a.out)

def cmd_scope(a):
    ds=load_npz(a.data)
    states=[x.strip() for x in a.states.split(',') if x.strip()]
    scoped=project_state_scope(ds,states,variant_suffix=a.name or 'scope')
    save_npz(scoped,a.out)
    gate=assess_observation(scoped)
    payload={"status":"PASS","out":a.out,"states":states,"X_shape":list(scoped.X.shape),"U_shape":list(scoped.U.shape),"dt":scoped.dt,"observation":gate.to_dict()}
    _dump(payload,a.report)


def cmd_mechanism_scope(a):
    ds=load_npz(a.data)
    states=[x.strip() for x in a.states.split(',') if x.strip()]
    drivers=[x.strip() for x in (a.drivers or '').split(',') if x.strip()]
    scoped=project_mechanism_scope(ds,target_states=states,driver_states=drivers,
                                   driver_role=a.driver_role,driver_sampling=a.driver_sampling,
                                   variant_suffix=a.name or 'mechanism-scope')
    save_npz(scoped,a.out)
    gate=assess_observation(scoped)
    payload={"status":"PASS","out":a.out,"states":states,"drivers":drivers,
             "X_shape":list(scoped.X.shape),"U_shape":list(scoped.U.shape),"dt":scoped.dt,
             "mechanism_scope":scoped.schema.get("mechanism_scope",{}),"observation":gate.to_dict()}
    _dump(payload,a.report)

def cmd_ehss_bridge(a):
    from .bridges.ehss import load_ehss_scenarios, save_ehss_bridge
    result=load_ehss_scenarios(a.scenario, view=a.view, allow_faulted=a.allow_faulted)
    save_ehss_bridge(result, a.out, a.contract_out)
    _dump({
        "status":"PASS",
        "view":a.view,
        "out":a.out,
        "contract_out":a.contract_out,
        "X_shape":list(result.dataset.X.shape),
        "U_shape":list(result.dataset.U.shape),
        "dt":result.dataset.dt,
        "trajectory_id":[int(x) for x in result.dataset.trajectory_id],
        "production_mechanism_eligible":result.contract["production_mechanism_eligible"],
    })

def cmd_simulate(a):
    from .simulation import simulate
    ds,_=simulate(a.system,a.n,T=a.steps+1,dt=a.dt,seed=a.seed,regime=a.regime,variant=a.variant)
    save_npz(ds,a.out); print(a.out)

def _json_arg(value):
    p=Path(value)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else json.loads(value)

def cmd_recover_missing(a):
    tr=load_npz(a.train); va=load_npz(a.val)
    known=_json_arg(a.known_terms) if a.known_terms else {}
    candidates=_json_arg(a.candidates)
    if not isinstance(known,dict): raise ValueError("--known-terms must be a JSON object mapping term to fixed coefficient")
    if not isinstance(candidates,list): raise ValueError("--candidates must be a JSON list of term strings")
    r=recover_missing_terms(tr,va,target=a.target,known_terms=known,candidate_terms=candidates)
    _dump(r.to_dict(),a.out)


def cmd_reduce_physics(a):
    tr=load_algebraic_npz(a.train); va=load_algebraic_npz(a.val)
    inputs=[x.strip() for x in a.inputs.split(',') if x.strip()]
    r=reduce_algebraic_search_space(tr,va,inputs=inputs,target=a.target,max_total_order=a.max_order,allow_inverse=a.allow_inverse,degree=a.degree,max_terms=a.max_terms)
    _dump(r.to_dict(),a.out)

def cmd_recover_missing_trajectory(a):
    tr=load_npz(a.train); va=load_npz(a.val)
    known=_json_arg(a.known_terms)
    candidates=_json_arg(a.candidates)
    targets=[x.strip() for x in a.unknown_targets.split(',') if x.strip()]
    if not isinstance(known,dict) or not all(isinstance(v,dict) for v in known.values()):
        raise ValueError('--known-terms must map target state to {term: coefficient}')
    if not isinstance(candidates,dict): raise ValueError('--candidates must map unknown target to a list of term strings')
    r=recover_missing_dynamics_trajectory(tr,va,known_terms=known,unknown_targets=targets,candidate_terms=candidates,hidden=a.hidden,depth=a.depth,epochs=a.epochs,batch_windows=a.batch_windows,window_steps=a.window_steps,learning_rate=a.learning_rate,seed=a.seed)
    _dump(r.to_dict(),a.out)

def cmd_recover(a):
    tr=load_npz(a.train); va=load_npz(a.val)
    r=recover_graph(tr,va,seed=a.seed,device=a.device,epochs=a.epochs,refit_epochs=a.refit,
                    constitutive_search=a.constitutive,backend=a.backend,backend_options=_backend_options(a))
    save_checkpoint(r,a.model); _dump(r.summary.to_dict(),a.summary)

def _load_prior(path, section=None):
    if not path: return {}
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    if section and isinstance(data,dict) and section in data:
        value=data[section]
        if not isinstance(value,dict): raise ValueError(f"prior section {section!r} must be an object")
        return value
    # Plain target -> [sources] mapping remains the compact CLI format.
    if isinstance(data,dict) and all(isinstance(v,list) for v in data.values()): return data
    if section:
        raise ValueError(f"prior file is neither a plain target mapping nor a wrapper containing {section!r}")
    return data

def cmd_complete(a):
    tr=load_npz(a.train); va=load_npz(a.val)
    r=complete_graph(tr,va,known_present=_load_prior(a.known_present,"known_present"),
                     known_absent=_load_prior(a.known_absent,"known_absent") or None,
                     seed=a.seed,device=a.device,epochs=a.epochs,refit_epochs=a.refit,
                     backend=a.backend,backend_options=_backend_options(a))
    save_checkpoint(r,a.model); _dump(r.summary.to_dict(),a.summary)

def cmd_inspect(a):
    r=load_checkpoint(a.model,device=a.device); _dump(r.summary.to_dict())

def cmd_calibrate_coverage(a):
    r=load_checkpoint(a.model,device=a.device)
    groups=[load_npz(p) for p in a.replicate_group]
    cal=calibrate_completion_coverage(r,groups,quantile=a.quantile,min_floor=a.min_floor,device=a.device)
    _dump(cal.to_dict(),a.out)

def cmd_certify(a):
    r=load_checkpoint(a.model,device=a.device); ds=load_npz(a.data)
    calibration=json.loads(Path(a.coverage_calibration).read_text(encoding='utf-8')) if a.coverage_calibration else None
    d=certify_completion(r,ds,declared_present=_load_prior(a.declared_present,"known_present"),
                         coverage_floor=a.coverage_floor,coverage_calibration=calibration,
                         coverage_threshold=a.coverage_threshold,device=a.device)
    _dump(d,a.out)

def cmd_phm(a):
    r=load_checkpoint(a.model,device=a.device); normal=load_npz(a.normal); data=load_npz(a.data)
    if a.certificate:
        cert=json.loads(Path(a.certificate).read_text(encoding='utf-8'))
        if cert.get('status') != 'COMPLETE' and not a.allow_noncomplete:
            raise ValueError(f"PHM blocked by completion certificate status={cert.get('status')!r}; use --allow-noncomplete only for explicit research/audit")
    nom=fit_normal_phm(normal,r.summary.graph,operators=r.summary.operators,operator_parameters=r.summary.operator_parameters); feat=extract_mechanism_features(data,nom)
    np.savetxt(a.out,feat,delimiter=',',header=','.join(nom.feature_names),comments='')
    meta={"feature_names":nom.feature_names,"rows":int(feat.shape[0]),"columns":int(feat.shape[1]),"source_backend":r.summary.backend,"operators":r.summary.operators,"operator_parameters":r.summary.operator_parameters}
    Path(str(a.out)+'.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(a.out)


def cmd_phm_health(a):
    r=load_checkpoint(a.model,device=a.device); normal=load_npz(a.normal); data=load_npz(a.data)
    if a.certificate:
        cert=json.loads(Path(a.certificate).read_text(encoding='utf-8'))
        if cert.get('status') != 'COMPLETE' and not a.allow_noncomplete:
            raise ValueError(f"PHM blocked by completion certificate status={cert.get('status')!r}; use --allow-noncomplete only for explicit research/audit")
    nom=fit_normal_phm(normal,r.summary.graph,operators=r.summary.operators,operator_parameters=r.summary.operator_parameters)
    cal=calibrate_phm_health(normal,nom)
    raw,names=mechanism_health_evidence(data,nom); scores,score_names=score_phm_health(data,nom,cal)
    np.savetxt(a.out,raw,delimiter=',',header=','.join(names),comments='')
    if a.scores_out:
        np.savetxt(a.scores_out,scores,delimiter=',',header=','.join(score_names),comments='')
    meta={"feature_names":names,"score_feature_names":score_names,"rows":int(raw.shape[0]),"calibration":cal.to_dict(),"source_backend":r.summary.backend,"operators":r.summary.operators,"operator_parameters":r.summary.operator_parameters}
    Path(str(a.out)+'.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    if a.calibration_out: Path(a.calibration_out).write_text(json.dumps(cal.to_dict(),indent=2),encoding='utf-8')
    print(a.out)


def _float_grid(text):
    vals=[float(x.strip()) for x in str(text).split(',') if x.strip()]
    if not vals or any((not np.isfinite(x) or x<=0) for x in vals):
        raise ValueError("grid must contain positive finite comma-separated values")
    return vals

def _json_mapping(text):
    if not text: return {}
    p=Path(text)
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else json.loads(text)

def cmd_stribeck_refit(a):
    r=load_checkpoint(a.model,device=a.device); tr=load_npz(a.train); va=load_npz(a.val)
    rr,profile=profile_smooth_stribeck(
        r,tr,va,target=a.target,source=a.source,
        smoothing_grid=_float_grid(a.smoothing_grid),
        breakaway_grid=_float_grid(a.breakaway_grid),
        expected_signs=_json_mapping(a.expected_signs),ridge=a.ridge)
    save_checkpoint(rr,a.out_model,metadata={"stage":"constitutive-profile","family":"SmoothStribeck","profile":profile.to_dict()})
    payload={"status":"PASS","model":a.out_model,"profile":profile.to_dict(),"summary":rr.summary.to_dict()}
    _dump(payload,a.out)

def cmd_audit_motor(a):
    from .real_motor import load_dc_motor_2021, audit_trajectories, propose_dc_motor_splits, save_audit
    tr=load_dc_motor_2021(a.workbook,min_samples=a.min_samples)
    payload={"audit":audit_trajectories(tr),"splits":propose_dc_motor_splits(tr)}
    save_audit(a.out,payload)
    audit_summary={k:v for k,v in payload["audit"].items() if k!="trajectory_audits"}
    compact={"status":"PASS","trajectory_count":len(tr),"output":a.out,
             "split_counts":{k:len(v) for k,v in payload["splits"].items()},"audit":audit_summary}
    _dump(compact)



def cmd_build(a):
    tr=load_npz(a.train); va=load_npz(a.val); cert=load_npz(a.cert_data) if a.cert_data else va
    sg=None
    if a.symbolic_inputs and a.symbolic_target and a.symbolic_out:
        sg={'inputs':a.symbolic_inputs.split(','),'target':a.symbolic_target,'out_csv':a.symbolic_out}
    out=build_operational_model(tr,va,backend=a.backend,seed=a.seed,device=a.device,epochs=a.epochs,refit_epochs=a.refit,
                                constitutive_search=a.constitutive,known_present=_load_prior(a.known_present,"known_present"),
                                known_absent=_load_prior(a.known_absent,"known_absent") or None,declared_present=_load_prior(a.declared_present,"known_present"),
                                certification_data=cert,coverage_floor=a.coverage_floor,coverage_threshold=a.coverage_threshold,
                                backend_options=_backend_options(a),symbolic_gap=sg)
    if out.recovery is not None:
        save_checkpoint(out.recovery,a.model)
    _dump(out.report(),a.out)


def cmd_consensus(a):
    tr=load_npz(a.train); va=load_npz(a.val); cert=load_npz(a.cert_data) if a.cert_data else va
    names=[x.strip() for x in a.backends.split(',') if x.strip()] if a.backends else None
    out=build_consensus_model(tr,va,backends=names,seed=a.seed,device=a.device,epochs=a.epochs,refit_epochs=a.refit,
                              constitutive_search=a.constitutive,known_present=_load_prior(a.known_present,"known_present"),
                              known_absent=_load_prior(a.known_absent,"known_absent") or None,declared_present=_load_prior(a.declared_present,"known_present"),
                              certification_data=cert,coverage_floor=a.coverage_floor,coverage_threshold=a.coverage_threshold,
                              min_graph_jaccard=a.min_graph_jaccard,min_operator_agreement=a.min_operator_agreement)
    if out.primary is not None:
        save_checkpoint(out.primary,a.model)
    _dump(out.report(),a.out)


def cmd_fallback_linear(a):
    tr=load_npz(a.train); va=load_npz(a.val)
    r=fit_linear_state_space_fallback(tr,va,ridge=a.ridge,max_steps=a.max_steps)
    save_fallback(r,a.model); _dump(r.summary.to_dict(),a.out)


def cmd_symbolic_export(a):
    ds=load_npz(a.data)
    task=export_explicit_relation(ds,inputs=a.inputs.split(','),target=a.target,out_csv=a.out,max_samples=a.max_samples,seed=a.seed)
    _dump(task.to_dict(),str(a.out)+'.task.json')

def cmd_symbolic_validate(a):
    ds=load_npz(a.data)
    _dump(validate_expression(ds,inputs=a.inputs.split(','),target=a.target,expression=a.expression),a.out)

def cmd_symmatika_run(a):
    task=json.loads(Path(a.task).read_text(encoding='utf-8'))
    b=SymMatikaExternalBackend(a.bridge)
    r=b.run_explicit(task['csv_path'],task['variables'],population=a.population,depth=a.depth,timeout=a.timeout)
    _dump(r.to_dict(),a.out)


def cmd_route_evidence(a):
    from .atlas import load_atlas
    from .evidence_router import HealthEvidenceEvent, route_health_evidence_batch
    atlas=load_atlas(a.atlas)
    raw=json.loads(Path(a.events).read_text(encoding='utf-8'))
    if isinstance(raw,dict) and 'events' in raw: raw=raw['events']
    if not isinstance(raw,list): raise ValueError('events JSON must be a list or {"events": [...]}')
    events=[HealthEvidenceEvent(**x) for x in raw]
    routes=route_health_evidence_batch(atlas,events)
    payload={'format':'isgn-evidence-routing-v1','system_id':atlas.system_id,'routes':[r.to_dict() for r in routes]}
    _dump(payload,a.out)

def cmd_synthesize_atlas(a):
    from .atlas import save_atlas
    from .atlas_synthesis import synthesis_input_from_dict, synthesize_mechanism_atlas
    payload=json.loads(Path(a.evidence).read_text(encoding="utf-8"))
    system_id,variables,relations,metadata=synthesis_input_from_dict(payload)
    result=synthesize_mechanism_atlas(system_id=system_id,variables=variables,relations=relations,metadata=metadata)
    save_atlas(result.atlas,a.out)
    if a.report: Path(a.report).write_text(json.dumps(result.to_dict(),indent=2,ensure_ascii=False),encoding="utf-8")
    _dump({"status":result.status,"atlas":a.out,"selected_proposals":result.selected_proposals,"uncovered_targets":result.uncovered_targets,
           "n_scopes":len(result.atlas.scopes),"n_uncovered_regions":len(result.atlas.uncovered_regions)})

def cmd_compile_architecture(a):
    from .atlas import load_atlas
    from .architecture import compile_mechanism_atlas
    atlas=load_atlas(a.atlas)
    bp=compile_mechanism_atlas(atlas,include_uncertified=a.include_uncertified,
                               activate_uncovered_residuals=a.activate_uncovered_residuals)
    _dump(bp.to_dict(),a.out)


def cmd_plan_closed_loop(a):
    from .closed_loop import closed_loop_input_from_dict, plan_closed_loop_atlas
    payload=json.loads(Path(a.evidence).read_text(encoding="utf-8"))
    kwargs=closed_loop_input_from_dict(payload)
    plan=plan_closed_loop_atlas(**kwargs,max_actions=a.max_actions)
    _dump(plan.to_dict(),a.out)

def cmd_evaluate_closed_loop(a):
    from .closed_loop import evaluate_closed_loop_plan
    plan=json.loads(Path(a.plan).read_text(encoding="utf-8"))
    after=json.loads(Path(a.after).read_text(encoding="utf-8"))
    result=evaluate_closed_loop_plan(plan,after=after)
    _dump(result.to_dict(),a.out)


def cmd_closed_loop_cycle(a):
    from .closed_loop import run_closed_loop_cycle
    before=json.loads(Path(a.before).read_text(encoding="utf-8"))
    after=json.loads(Path(a.after).read_text(encoding="utf-8"))
    result=run_closed_loop_cycle(before=before,after=after,max_actions=a.max_actions)
    _dump(result.to_dict(),a.out)

def _common_fit(q):
    q.add_argument('--train',required=True);q.add_argument('--val',required=True);q.add_argument('--model',required=True);q.add_argument('--summary')
    q.add_argument('--seed',type=int,default=101);q.add_argument('--device',default='cpu');q.add_argument('--epochs',type=int,default=1000);q.add_argument('--refit',type=int,default=300)
    q.add_argument('--backend',choices=['native','pysindy','reference-sindy','sklearn-lasso','auto'],default='native')
    q.add_argument('--backend-options',help='JSON string or JSON file passed to discovery backend')

def parser():
    p=argparse.ArgumentParser(prog='isgn',description='ISGN: partial-physics mechanism recovery, completion, certification, and optional PHM tooling')
    sp=p.add_subparsers(dest='cmd',required=True)
    q=sp.add_parser('doctor'); q.set_defaults(fn=cmd_doctor)
    q=sp.add_parser('route-evidence',help='route health evidence through a certified Mechanism Atlas without inventing diagnoses'); q.add_argument('--atlas',required=True); q.add_argument('--events',required=True,help='JSON list of HealthEvidenceEvent records'); q.add_argument('--out'); q.set_defaults(fn=cmd_route_evidence)
    q=sp.add_parser('synthesize-atlas',help='propose multi-scope Atlas from stable relation evidence plus observation qualification'); q.add_argument('--evidence',required=True); q.add_argument('--out',required=True); q.add_argument('--report'); q.set_defaults(fn=cmd_synthesize_atlas)
    q=sp.add_parser('compile-architecture',help='compile certified Atlas scopes into a safe neural architecture blueprint'); q.add_argument('--atlas',required=True); q.add_argument('--out'); q.add_argument('--include-uncertified',action='store_true'); q.add_argument('--activate-uncovered-residuals',action='store_true'); q.set_defaults(fn=cmd_compile_architecture)
    q=sp.add_parser('plan-closed-loop',help='turn current Atlas/Observation/Consensus/Completion gaps into safe next-information actions'); q.add_argument('--evidence',required=True); q.add_argument('--out'); q.add_argument('--max-actions',type=int); q.set_defaults(fn=cmd_plan_closed_loop)
    q=sp.add_parser('evaluate-closed-loop',help='evaluate whether a closed-loop action plan was actually resolved by new production evidence'); q.add_argument('--plan',required=True); q.add_argument('--after',required=True); q.add_argument('--out'); q.set_defaults(fn=cmd_evaluate_closed_loop)
    q=sp.add_parser('closed-loop-cycle',help='plan, evaluate new evidence, and immediately re-plan the remaining Atlas information gaps'); q.add_argument('--before',required=True); q.add_argument('--after',required=True); q.add_argument('--out'); q.add_argument('--max-actions',type=int); q.set_defaults(fn=cmd_closed_loop_cycle)
    q=sp.add_parser('backends'); q.set_defaults(fn=cmd_backends)
    q=sp.add_parser('observe',help='pre-discovery Observation Gate'); q.add_argument('--data',required=True); q.add_argument('--out'); q.set_defaults(fn=cmd_observe)
    q=sp.add_parser('observe-compare',help='compare multiple acquisition conditions and derive state-by-state discovery routes'); q.add_argument('--candidate',action='append',required=True,help='production NPZ as NAME=PATH; repeatable'); q.add_argument('--audit-only',action='append',default=[],help='audit-only/truth NPZ as NAME=PATH; repeatable'); q.add_argument('--out'); q.set_defaults(fn=cmd_observe_compare)
    q=sp.add_parser('scope',help='project a PublicDataset onto an explicit mechanism-discovery state scope'); q.add_argument('--data',required=True); q.add_argument('--states',required=True,help='comma-separated state names'); q.add_argument('--name',default='scope'); q.add_argument('--out',required=True); q.add_argument('--report'); q.set_defaults(fn=cmd_scope)
    q=sp.add_parser('mechanism-scope',help='build a local mechanism block by promoting observed states to measured drivers'); q.add_argument('--data',required=True); q.add_argument('--states',required=True,help='comma-separated integrated target states'); q.add_argument('--drivers',default='',help='comma-separated observed states promoted to local drivers'); q.add_argument('--driver-role',choices=['input','context'],default='context'); q.add_argument('--driver-sampling',choices=['left','midpoint'],default='left'); q.add_argument('--name',default='mechanism-scope'); q.add_argument('--out',required=True); q.add_argument('--report'); q.set_defaults(fn=cmd_mechanism_scope)
    q=sp.add_parser('ehss-bridge',help='convert EHSS diagnosis scenario(s) to the ISGN PublicDataset contract'); q.add_argument('--scenario',action='append',required=True,help='EHSS scenario directory; repeat for multiple equal-length trajectories'); q.add_argument('--view',default='measured-fast'); q.add_argument('--out',required=True,help='output PublicDataset NPZ'); q.add_argument('--contract-out',required=True,help='output traceable observation-contract JSON'); q.add_argument('--allow-faulted',action='store_true',help='explicitly allow faulted scenarios (blocked by default for normal discovery)'); q.set_defaults(fn=cmd_ehss_bridge)
    q=sp.add_parser('simulate'); q.add_argument('--system',choices=['thermal','tank','motor','msd'],required=True); q.add_argument('--n',type=int,default=32); q.add_argument('--steps',type=int,default=160); q.add_argument('--dt',type=float,default=.05); q.add_argument('--seed',type=int,default=101); q.add_argument('--regime',choices=['iid','near','far'],default='iid'); q.add_argument('--variant',choices=['clean','constitutive','misspecified'],default='clean'); q.add_argument('--out',required=True); q.set_defaults(fn=cmd_simulate)
    q=sp.add_parser('recover-missing',help='recover only dynamics unexplained by fixed partial physics'); q.add_argument('--train',required=True); q.add_argument('--val',required=True); q.add_argument('--target',required=True); q.add_argument('--known-terms',help='JSON object or path, e.g. {"sin(theta)":-3,"u":1}'); q.add_argument('--candidates',required=True,help='JSON list or path of candidate residual terms'); q.add_argument('--out'); q.set_defaults(fn=cmd_recover_missing)
    q=sp.add_parser('reduce-physics',help='reduce an algebraic mechanism search space using units and held-out pairwise structure evidence'); q.add_argument('--train',required=True,help='AlgebraicDataset NPZ'); q.add_argument('--val',required=True,help='AlgebraicDataset NPZ'); q.add_argument('--inputs',required=True,help='comma-separated variables'); q.add_argument('--target',required=True); q.add_argument('--max-order',type=int,default=3); q.add_argument('--allow-inverse',action='store_true'); q.add_argument('--degree',type=int,default=2); q.add_argument('--max-terms',type=int,default=3); q.add_argument('--out'); q.set_defaults(fn=cmd_reduce_physics)
    q=sp.add_parser('recover-missing-trajectory',help='learn unexplained dynamics from whole trajectories, then distill them into declared mechanism terms'); q.add_argument('--train',required=True); q.add_argument('--val',required=True); q.add_argument('--known-terms',required=True,help='JSON/path: target -> {term: fixed coefficient}'); q.add_argument('--unknown-targets',required=True,help='comma-separated state names receiving a learned residual'); q.add_argument('--candidates',required=True,help='JSON/path: target -> [candidate terms]'); q.add_argument('--hidden',type=int,default=32); q.add_argument('--depth',type=int,default=2); q.add_argument('--epochs',type=int,default=800); q.add_argument('--batch-windows',type=int,default=32); q.add_argument('--window-steps',type=int,default=12); q.add_argument('--learning-rate',type=float,default=2e-3); q.add_argument('--seed',type=int,default=0); q.add_argument('--out'); q.set_defaults(fn=cmd_recover_missing_trajectory)
    q=sp.add_parser('recover'); _common_fit(q); q.add_argument('--constitutive',action='store_true'); q.set_defaults(fn=cmd_recover)
    q=sp.add_parser('complete'); _common_fit(q); q.add_argument('--known-present',required=True);q.add_argument('--known-absent'); q.set_defaults(fn=cmd_complete)
    q=sp.add_parser('inspect');q.add_argument('--model',required=True);q.add_argument('--device',default='cpu');q.set_defaults(fn=cmd_inspect)
    q=sp.add_parser('calibrate-coverage',help='calibrate per-target numerical/measurement residual floors from repeated acquisitions');q.add_argument('--model',required=True);q.add_argument('--replicate-group',action='append',required=True,help='NPZ containing >=2 repeats of the same physical condition; repeat for multiple conditions');q.add_argument('--quantile',type=float,default=.95);q.add_argument('--min-floor',type=float,default=1e-6);q.add_argument('--device',default='cpu');q.add_argument('--out',required=True);q.set_defaults(fn=cmd_calibrate_coverage)
    q=sp.add_parser('certify');q.add_argument('--model',required=True);q.add_argument('--data',required=True);q.add_argument('--declared-present');q.add_argument('--coverage-floor',type=float);q.add_argument('--coverage-calibration');q.add_argument('--coverage-threshold',type=float,default=8.0);q.add_argument('--device',default='cpu');q.add_argument('--out');q.set_defaults(fn=cmd_certify)
    q=sp.add_parser('phm-features');q.add_argument('--model',required=True);q.add_argument('--normal',required=True);q.add_argument('--data',required=True);q.add_argument('--out',required=True);q.add_argument('--device',default='cpu');q.add_argument('--certificate',help='completion-certificate JSON; non-COMPLETE blocks PHM by default');q.add_argument('--allow-noncomplete',action='store_true',help='research/audit override for non-COMPLETE certificates');q.set_defaults(fn=cmd_phm)
    q=sp.add_parser('phm-health',help='calibrate normal-window PHM false-positive floors and emit anchored-drift/residual health evidence');q.add_argument('--model',required=True);q.add_argument('--normal',required=True);q.add_argument('--data',required=True);q.add_argument('--out',required=True);q.add_argument('--scores-out');q.add_argument('--calibration-out');q.add_argument('--device',default='cpu');q.add_argument('--certificate');q.add_argument('--allow-noncomplete',action='store_true');q.set_defaults(fn=cmd_phm_health)
    q=sp.add_parser('stribeck-refit',help='profile a reusable smooth Stribeck/Coulomb/viscous constitutive motif on an existing graph edge');q.add_argument('--model',required=True);q.add_argument('--train',required=True);q.add_argument('--val',required=True);q.add_argument('--target',required=True);q.add_argument('--source',required=True);q.add_argument('--smoothing-grid',required=True,help='positive comma-separated velocity scales');q.add_argument('--breakaway-grid',required=True,help='positive comma-separated velocity scales');q.add_argument('--expected-signs',help='JSON mapping source name to -1/0/+1, or path to JSON');q.add_argument('--ridge',type=float,default=1e-10);q.add_argument('--device',default='cpu');q.add_argument('--out-model',required=True);q.add_argument('--out');q.set_defaults(fn=cmd_stribeck_refit)
    q=sp.add_parser('audit-real-motor');q.add_argument('workbook');q.add_argument('--out',default='m3_audit.json');q.add_argument('--min-samples',type=int,default=20);q.set_defaults(fn=cmd_audit_motor)


    q=sp.add_parser('build'); _common_fit(q); q.add_argument('--cert-data'); q.add_argument('--constitutive',action='store_true');q.add_argument('--known-present');q.add_argument('--known-absent');q.add_argument('--declared-present');q.add_argument('--coverage-floor',type=float);q.add_argument('--coverage-threshold',type=float,default=8.0);q.add_argument('--symbolic-inputs');q.add_argument('--symbolic-target');q.add_argument('--symbolic-out');q.add_argument('--out');q.set_defaults(fn=cmd_build)
    q=sp.add_parser('consensus'); _common_fit(q); q.add_argument('--backends',help='comma-separated backend names; default=native+pysindy if available else native+reference-sindy'); q.add_argument('--cert-data'); q.add_argument('--constitutive',action='store_true'); q.add_argument('--known-present'); q.add_argument('--known-absent'); q.add_argument('--declared-present'); q.add_argument('--coverage-floor',type=float); q.add_argument('--coverage-threshold',type=float,default=8.0); q.add_argument('--min-graph-jaccard',type=float,default=.8); q.add_argument('--min-operator-agreement',type=float,default=.8); q.add_argument('--out'); q.set_defaults(fn=cmd_consensus)
    q=sp.add_parser('fallback-linear'); q.add_argument('--train',required=True); q.add_argument('--val',required=True); q.add_argument('--model',required=True); q.add_argument('--ridge',type=float,default=1e-8); q.add_argument('--max-steps',type=int,default=200); q.add_argument('--out'); q.set_defaults(fn=cmd_fallback_linear)
    q=sp.add_parser('symbolic-export');q.add_argument('--data',required=True);q.add_argument('--inputs',required=True,help='comma-separated input variable names');q.add_argument('--target',required=True);q.add_argument('--out',required=True);q.add_argument('--max-samples',type=int,default=10000);q.add_argument('--seed',type=int,default=0);q.set_defaults(fn=cmd_symbolic_export)
    q=sp.add_parser('symbolic-validate');q.add_argument('--data',required=True);q.add_argument('--inputs',required=True);q.add_argument('--target',required=True);q.add_argument('--expression',required=True);q.add_argument('--out');q.set_defaults(fn=cmd_symbolic_validate)
    q=sp.add_parser('symmatika-run');q.add_argument('--task',required=True);q.add_argument('--bridge');q.add_argument('--population',type=int,default=10000);q.add_argument('--depth',type=int,default=3);q.add_argument('--timeout',type=int,default=3600);q.add_argument('--out');q.set_defaults(fn=cmd_symmatika_run)
    return p

def main():
    a=parser().parse_args(); a.fn(a)
if __name__=='__main__': main()
