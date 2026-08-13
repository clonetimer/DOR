from __future__ import annotations
import argparse
import platform
import shutil
import sys
from importlib import resources
from pathlib import Path


def _version(pkg: str) -> str:
    try:
        mod=__import__(pkg)
        return getattr(mod,'__version__','unknown')
    except Exception as e:
        return f'NOT AVAILABLE ({e})'


def doctor() -> int:
    print('M2N Practitioner Toolkit doctor')
    print('--------------------------------')
    print('Python :', sys.version.split()[0])
    print('OS     :', platform.platform())
    print('numpy  :', _version('numpy'))
    print('scipy  :', _version('scipy'))
    print('torch  :', _version('torch'))
    print('sklearn:', _version('sklearn'))
    try:
        import torch
        from .canonical_ir import Dim
        from .mechanics_ir import CoordinateSpec, InputSpec, DriveSpec, ResistanceSpec, SecondOrderMechanicsIR, compile_second_order_ir
        L=Dim.of(length=1); A=L/(Dim.of(time=1)**2)
        ir=SecondOrderMechanicsIR(
            coordinate=CoordinateSpec('q','v',L),
            inputs=(InputSpec('u',A),),
            drive=DriveSpec('drive','u',Dim(),1.0),
            resistance=ResistanceSpec('drag','viscous',viscous_gain_init=0.1),
            dt=0.01,
        )
        model,_=compile_second_order_ir(ir)
        state=torch.zeros((2,2),dtype=torch.float32)
        out=model.rate(state,u=torch.ones(2))
        assert tuple(out.shape)==(2,2)
        print('compile: PASS')
        print('status : READY')
        return 0
    except Exception as e:
        print('compile: FAIL')
        print('reason :', repr(e))
        return 2


def new_project(path: str, force: bool=False) -> int:
    dst=Path(path).resolve()
    if dst.exists() and any(dst.iterdir()) and not force:
        raise SystemExit(f'target is not empty: {dst} (use --force to overwrite template files)')
    dst.mkdir(parents=True,exist_ok=True)
    src=resources.files('m2n').joinpath('templates/device_project')
    for item in src.iterdir():
        target=dst/item.name
        if item.is_dir():
            target.mkdir(parents=True,exist_ok=True)
            for child in item.iterdir():
                shutil.copyfile(child,target/child.name)
        else:
            shutil.copyfile(item,target)
    print(f'Created M2N starter project: {dst}')
    print('Next: edit mechanism_contract.py and run: python run_design.py')
    return 0


def info() -> int:
    from . import __version__, __method_baseline__
    print(f'M2N Practitioner Toolkit {__version__}')
    print(f'Method baseline: {__method_baseline__}')
    print('Primary mechanism workflow: device_card.json + normal_data.csv -> knowledge/bias/representation qualification -> architecture plan -> executable candidates -> selection/model set -> model artifact')
    print('Fault-diagnosis model workflow: diagnostic_request.json + labelled/normal data -> task/data/feature-view qualification -> finite model-family competition -> score bundle -> FEC/DOR')
    print('Optional backends serve the primary workflow only when the qualified design problem requires them; they never bypass M2N qualification/admission.')
    print('Not claimed as novel: equation discovery, unknown-physics neuralization, symbolic-to-PyTorch compilation by themselves.')
    return 0


def backends() -> int:
    from .backends import list_backends
    print('M2N optional backend registry')
    print('-----------------------------')
    for b in list_backends():
        caps=','.join(x.value for x in b.capability)
        probe=b.runtime_probe()
        installed='yes' if probe.installed else 'no'
        importable='yes' if probe.importable else 'no'
        ver=probe.version or '-'
        lic=b.code_license or b.code_license_status
        print(f'{b.id:16s} {b.integration_status:16s} installed={installed:3s} importable={importable:3s} version={ver:10s} license={lic}')
        print(f'  capability={caps}')
        if probe.error and probe.installed: print(f'  runtime_error={probe.error}')
        if b.notes: print(f'  note={b.notes}')
    return 0


def nkt() -> int:
    from .novelty_kill import default_claim_ledger, run_behavioral_nkt
    print('M2N NKT-1 claim ledger')
    print('----------------------')
    for c in default_claim_ledger():
        print(f'{c.id}: {c.status.value}')
        print(f'  {c.claim}')
        print(f'  disposition: {c.disposition}')
    results=run_behavioral_nkt()
    ok=sum(r.passed for r in results)
    print(f'Behavioral adjudication checks: {ok}/{len(results)} PASS')
    for r in results:
        print(f'  {"PASS" if r.passed else "FAIL"}: {r.case} -> {r.actual_status}/{r.actual_action}')
    return 0 if ok==len(results) else 2


def nkt2() -> int:
    from .novelty_kill import nkt2_claim_ledger, run_behavioral_nkt
    print('M2N NKT-2 targeted survivor reassessment')
    print('----------------------------------------')
    for c in nkt2_claim_ledger():
        print(f'{c.id}: {c.status.value}')
        print(f'  {c.claim}')
        print(f'  disposition: {c.disposition}')
    results=run_behavioral_nkt()
    ok=sum(r.passed for r in results)
    print(f'Frozen behavioral adjudication checks: {ok}/{len(results)} PASS')
    return 0 if ok==len(results) else 2



def symmatika_probe(source_root: str | None) -> int:
    from .symmatika_adapter import probe_symmatika_source
    p=probe_symmatika_source(source_root)
    print('M2N SymMatika external-source probe')
    print('----------------------------------')
    print('source_root    :', p.source_root or '-')
    print('source_ready   :', p.source_ready)
    print('cmake_available:', p.cmake_available)
    print('cxx_available  :', p.cxx_available)
    print('ready          :', p.ready)
    print('reason         :', p.reason)
    return 0 if p.ready else 2


def sbmi_feasibility(config_path: str) -> int:
    import json
    from .model_set_inference import ModelComponentSlot, ParameterPrior, SBMIModelSetRequest, assess_sbmi_feasibility
    d=json.loads(Path(config_path).read_text(encoding='utf-8'))
    req=SBMIModelSetRequest(
        slots=tuple(ModelComponentSlot(x['slot_id'],tuple(x['candidates'])) for x in d['slots']),
        parameter_priors=tuple(ParameterPrior(**x) for x in d.get('parameter_priors',[])),
        simulator_available=bool(d.get('simulator_available',True)),
        randomized_component_simulation=bool(d.get('randomized_component_simulation',True)),
        observation_summary_defined=bool(d.get('observation_summary_defined',True)),
        simulation_budget=d.get('simulation_budget'),
        exact_enumeration_limit=int(d.get('exact_enumeration_limit',32)),
    )
    a=assess_sbmi_feasibility(req)
    print(json.dumps(a.as_dict(),indent=2,ensure_ascii=False))
    return 0 if a.status != 'BLOCKED' else 2



def policy_preview(spec_path: str, out_path: str | None=None) -> int:
    import json
    from .bias_routing import build_bias_route_plan
    from .representation_policy import build_representation_plan
    spec=json.loads(Path(spec_path).read_text(encoding='utf-8'))
    payload={
        'bias_route_plan':build_bias_route_plan(spec).to_dict(),
        'representation_plan':[x.to_dict() for x in build_representation_plan(spec)],
        'note':'This is a qualification-oriented policy preview. Specialized architecture patterns are recommendations/contracts unless their runtime is explicitly implemented.',
    }
    text=json.dumps(payload,indent=2,ensure_ascii=False); print(text)
    if out_path: Path(out_path).write_text(text+'\n',encoding='utf-8')
    bad=any(x['eligibility']['status']=='INELIGIBLE' for x in payload['representation_plan'])
    return 2 if bad else 0


def model_plan(spec_path: str, data_path: str | None=None, out_path: str | None=None) -> int:
    import json
    from .model_family import load_and_plan_model_families
    plan=load_and_plan_model_families(spec_path,data_path)
    text=json.dumps(plan.to_dict(),indent=2,ensure_ascii=False)
    print(text)
    if out_path: Path(out_path).write_text(text+'\n',encoding='utf-8')
    return 0 if plan.status in ('READY','CONDITIONAL') else 2


def temporal_design(spec_path: str, data_path: str, out_dir: str) -> int:
    from .temporal_design import run_temporal_design
    r=run_temporal_design(spec_path,data_path,out_dir)
    print('M2N temporal design')
    print('-------------------')
    print('qualification:',r.qualification.get('status'))
    print('status       :',r.status)
    print('selected     :',r.selected_candidate_id or '-')
    print('task         :',r.task)
    print('output_dir   :',r.output_dir)
    return 0 if r.status in ('SELECTED','AMBIGUOUS') else 2


def predict_temporal(design_dir: str, data_path: str, out_path: str) -> int:
    from .temporal_design import predict_temporal_design
    p=predict_temporal_design(design_dir,data_path,out_path)
    print(f'Wrote temporal design predictions: {p}')
    return 0


def design(spec_path: str, data_path: str, out_dir: str) -> int:
    from .goal_design import run_goal_driven_design
    from .practitioner import write_next_actions, seal_design_bundle
    r=run_goal_driven_design(spec_path,data_path,out_dir)
    actions=write_next_actions(out_dir)
    manifest=seal_design_bundle(out_dir)
    print('M2N goal-driven design')
    print('----------------------')
    print('planning_status:',r.planning_status)
    print('status         :',r.status)
    print('selected       :',r.selected_candidate_id or '-')
    print('output_dir     :',r.output_dir)
    print('next_actions   :',actions)
    print('bundle_manifest:',manifest)
    return 0 if r.status in ('SELECTED','FIXED','AMBIGUOUS','K3_SELECTED','MODEL_SET_BACKEND_RECOMMENDED') else 2


def predict(artifact_path: str, data_path: str, out_path: str) -> int:
    from .goal_design import predict_model_artifact
    p=predict_model_artifact(artifact_path,data_path,out_path)
    print(f'Wrote predictions: {p}')
    return 0



def evidence(design_dir: str, data_path: str, out_path: str) -> int:
    from .goal_design import generate_model_evidence
    p=generate_model_evidence(design_dir,data_path,out_path)
    print(f'Wrote M2N model evidence: {p}')
    return 0


def fec_export(evidence_path: str, out_csv: str, contract_json: str, window_s: float=0.05) -> int:
    from .phm_handoff import export_fec_contact_pack
    c,j=export_fec_contact_pack(evidence_path,out_csv,contract_json,window_s=window_s)
    print(f'Wrote FEC Contact Pack: {c}')
    print(f'Wrote FEC handoff contract: {j}')
    return 0


def fec_calibrate(normal_leaf: str, normal_root: str, fec_root: str, out_ref: str, trusted_claim: str, scope: str) -> int:
    from .fec_runtime import calibrate_fec_from_contacts
    p=calibrate_fec_from_contacts(normal_leaf,normal_root,fec_root,out_ref,trusted_claim=trusted_claim,scope=scope)
    print(f'Wrote FEC calibration reference: {p}')
    return 0


def fec_infer(contacts: str, reference: str, fec_root: str, out_csv: str) -> int:
    from .fec_runtime import infer_fec_contacts
    p=infer_fec_contacts(contacts,reference,fec_root,out_csv)
    print(f'Wrote FEC policy evidence states: {p}')
    return 0

def init_design(path: str, template: str='mechanics', force: bool=False) -> int:
    import csv, json
    import numpy as np
    dst=Path(path).resolve()
    if dst.exists() and any(dst.iterdir()) and not force:
        raise SystemExit(f'target is not empty: {dst} (use --force to overwrite template files)')
    dst.mkdir(parents=True,exist_ok=True); (dst/'data').mkdir(exist_ok=True); (dst/'reports').mkdir(exist_ok=True)
    if template=='mechanics':
        spec={
          'schema_version':'1.0','template':'second_order_mechanics',
          'goal':{'name':'select a friction architecture from normal data'},
          'data':{'split_column':'split','train_values':['train'],'validation_values':['val']},
          'mechanism':{
            'coordinate':{'position':'q','velocity':'v','position_dim':{'length':1}},
            'inputs':[{'name':'u','dim':{'length':1,'time':-2}}],
            'drive':{'id':'actuator','input':'u','gain_dim':{},'init':1.0,'law':'gain','knowledge_source':'DECLARED_ENGINEERING'},
            'resistance':{'id':'friction','default_law':'viscous','viscous_gain_init':0.1,'coulomb_gain_init':0.1},
            'dt':0.01
          },
          'knowledge':{'resistance':{'level':'K2','finite_candidates':['viscous','viscous_coulomb'],'knowledge_source':'FINITE_ENGINEERING_HYPOTHESIS'}},
          'supervision':{'position_column':'q','velocity_column':'v','input_columns':{'u':'u'},'target_acceleration_column':'a'},
          'qualification':{'min_train_rows':80,'min_validation_rows':40},
          'training':{'epochs':300,'lr':0.06,'restarts':[101,102,103],'final_seed':9001},
          'selection':{'practical_ratio':0.8,'restart_stability':0.75}
        }
        rng=np.random.default_rng(7); n=360
        v=rng.uniform(-1,1,n); v[np.abs(v)<0.06]+=0.1
        u=rng.uniform(-0.8,0.8,n); a=1.35*u-0.22*v-0.48*np.sign(v)
        split=np.array(['train']*240+['val']*120,dtype=object)
        idx=np.arange(n); rng.shuffle(idx); v,u,a,split=v[idx],u[idx],a[idx],split
        rows=zip(np.zeros(n),v,u,a,split); header=['q','v','u','a','split']
    elif template=='peak':
        spec={
          'schema_version':'1.0','template':'separable_peak_response',
          'goal':{'name':'select which observed factors belong in a peaked local response'},
          'data':{'split_column':'split','train_values':['train'],'validation_values':['val']},
          'mechanism':{
            'axes':[
              {'name':'x1','dim':{},'transform':'identity','scale':1.0,'center_init':0.0,'width_init':1.0},
              {'name':'x2','dim':{},'transform':'identity','scale':1.0,'center_init':0.0,'width_init':1.0},
              {'name':'x3','dim':{},'transform':'identity','scale':1.0,'center_init':0.0,'width_init':1.0}],
            'families':[
              {'name':'x1_only','active_axes':['x1']},
              {'name':'x1_x2','active_axes':['x1','x2']},
              {'name':'x1_x2_x3','active_axes':['x1','x2','x3']}],
            'output_name':'y','output_dim':{},'amplitude_init':1.0
          },
          'knowledge':{'response_surface':{'level':'K2','knowledge_source':'BLINDED_FOR_VALIDATION'}},
          'supervision':{'input_columns':{'x1':'x1','x2':'x2','x3':'x3'},'target_column':'y'},
          'training':{'epochs':450,'lr':0.05,'restarts':[101,102,103]},
          'selection':{'practical_ratio':0.8,'restart_stability':0.75}
        }
        rng=np.random.default_rng(4); n=360
        x1=rng.normal(0,1,n); x2=rng.normal(0,1,n); x3=rng.normal(0,1,n)
        y=1.7*np.exp(-0.5*((x1-0.2)/0.8)**2-0.5*((x2+0.3)/1.1)**2-0.5*((x3-0.1)/0.55)**2)
        split=np.array(['train']*240+['val']*120,dtype=object)
        idx=np.arange(n); rng.shuffle(idx); x1,x2,x3,y,split=x1[idx],x2[idx],x3[idx],y[idx],split
        rows=zip(x1,x2,x3,y,split); header=['x1','x2','x3','y','split']
    elif template=='graph':
        A={'length':1,'time':-2}; V={'length':1,'time':-1}
        spec={
          'schema_version':'2.0','template':'generic_mechanism_graph',
          'goal':{'name':'design a generic mechanism graph from normal data'},
          'data':{'split_column':'split','train_values':['train'],'validation_values':['val']},
          'mechanism':{
            'inputs':[{'name':'u','dim':A},{'name':'v','dim':V}],
            'nodes':[
              {'id':'drive','op':'positive_gain','inputs':['u'],'params':{'learnable':True,'init':1.0,'gain_dim':{}}},
              {'id':'visc','op':'positive_gain','inputs':['v'],'params':{'learnable':True,'init':0.1,'gain_dim':{'time':-1}}},
              {'id':'sgn','op':'sign','inputs':['v'],'params':{}},
              {'id':'coul','op':'positive_gain','inputs':['sgn'],'params':{'learnable':True,'init':0.1,'gain_dim':A}},
              {'id':'resist','op':'add','inputs':['visc','coul'],'params':{}},
              {'id':'a','op':'subtract','inputs':['drive','resist'],'params':{}}],
            'outputs':[{'name':'a','dim':A}]},
          'knowledge':{'slots':[{'id':'resistance','level':'K2','role':'local resistance','input_dims':[V],'output_dim':A,'finite_candidates':['viscous','viscous_coulomb'],'knowledge_source':'FINITE_ENGINEERING_HYPOTHESIS'}]},
          'bindings':{'resistance':{
            'viscous':[{'path':'nodes.3.params.learnable','value':False},{'path':'nodes.3.params.value','value':0.0}],
            'viscous_coulomb':[{'path':'nodes.3.params.learnable','value':True},{'path':'nodes.3.params.init','value':0.1}]}},
          'supervision':{'input_columns':{'u':'u','v':'v'},'output_columns':{'a':'a'}},
          'qualification':{'min_train_rows':80,'min_validation_rows':40},
          'training':{'epochs':300,'lr':0.05,'restarts':[101,102,103],'final_seed':9001},
          'selection':{'practical_ratio':0.8,'restart_stability':0.75}}
        rng=np.random.default_rng(17); n=360
        v=rng.uniform(-1,1,n); v[np.abs(v)<0.06]+=0.1; u=rng.uniform(-0.8,0.8,n); a=1.25*u-0.18*v-0.42*np.sign(v)
        split=np.array(['train']*240+['val']*120,dtype=object); idx=np.arange(n); rng.shuffle(idx); u,v,a,split=u[idx],v[idx],a[idx],split
        rows=zip(u,v,a,split); header=['u','v','a','split']
    else:
        raise SystemExit('template must be mechanics, peak, or graph')
    (dst/'device_card.json').write_text(json.dumps(spec,indent=2,ensure_ascii=False),encoding='utf-8')
    with (dst/'data'/'normal_data.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)
    readme = """# M2N declarative design project

1. Edit `device_card.json`.
2. Replace `data/normal_data.csv` with your normal device data while preserving the mapped columns.
3. Preflight before training:

```bash
m2n validate --spec device_card.json --data data/normal_data.csv --out reports/preflight.json
m2n inspect-data --spec device_card.json --data data/normal_data.csv --out reports/data_inspection.json
```

4. Design and verify the sealed bundle:

```bash
m2n design --spec device_card.json --data data/normal_data.csv --out output
m2n check-bundle --design-dir output
```

5. Use the same prediction command for a selected model or a retained finite architecture set:

```bash
m2n predict-design --design-dir output --data data/normal_data.csv --out predictions.csv
```

Read `output/NEXT_ACTIONS.md` before moving to PHM evidence.
"""
    (dst/'README.md').write_text(readme,encoding='utf-8')
    print(f'Created declarative M2N design project: {dst}')
    print('Next: m2n design --spec device_card.json --data data/normal_data.csv --out output')
    return 0

def validate_practitioner(spec_path: str, data_path: str | None=None, out_path: str | None=None) -> int:
    import json
    from .practitioner import validate_device_card, inspect_tabular_data
    card=validate_device_card(spec_path)
    payload={'device_card':card.to_dict()}
    status=card.status
    if data_path:
        data=inspect_tabular_data(data_path,spec_path)
        payload['data']=data.to_dict()
        if data.status=='FAIL' or status=='FAIL': status='FAIL'
        elif data.status=='WARN' or status=='WARN': status='WARN'
        else: status='PASS'
    payload['overall_status']=status
    text=json.dumps(payload,indent=2,ensure_ascii=False)
    print(text)
    if out_path:
        Path(out_path).write_text(text+'\n',encoding='utf-8')
    return 2 if status=='FAIL' else 0


def inspect_data_cli(data_path: str, spec_path: str | None=None, out_path: str | None=None) -> int:
    import json
    from .practitioner import inspect_tabular_data
    r=inspect_tabular_data(data_path,spec_path)
    text=json.dumps(r.to_dict(),indent=2,ensure_ascii=False)
    print(text)
    if out_path:
        Path(out_path).write_text(text+'\n',encoding='utf-8')
    return 2 if r.status=='FAIL' else 0


def predict_design(design_dir: str, data_path: str, out_path: str) -> int:
    from .practitioner import predict_design_bundle
    p=predict_design_bundle(design_dir,data_path,out_path)
    print(f'Wrote design/portfolio predictions: {p}')
    return 0


def seal_bundle(design_dir: str) -> int:
    from .practitioner import write_next_actions, seal_design_bundle
    write_next_actions(design_dir)
    p=seal_design_bundle(design_dir)
    print(f'Sealed M2N design bundle: {p}')
    return 0


def check_bundle(design_dir: str) -> int:
    import json
    from .practitioner import verify_design_bundle
    r=verify_design_bundle(design_dir)
    print(json.dumps(r.to_dict(),indent=2,ensure_ascii=False))
    return 2 if r.status=='FAIL' else 0



def diagnostic_families(task: str | None=None) -> int:
    from .diagnostic_modeling import list_diagnostic_families
    rows=list_diagnostic_families(task)
    print('M2N diagnostic model family registry')
    print('------------------------------------')
    for x in rows:
        print(f'{x.id:24s} task={x.task:20s} kind={x.model_kind:16s} probability={x.probability_output}')
        print(f'  rationale={x.rationale}')
        if x.limitations: print('  limitations='+'; '.join(x.limitations))
    return 0


def diagnostic_design(request_path: str, data_path: str, out_dir: str) -> int:
    from .diagnostic_modeling import run_diagnostic_model_design
    r=run_diagnostic_model_design(request_path,data_path,out_dir)
    print('M2N diagnostic model design')
    print('---------------------------')
    print('task           :',r.task)
    print('qualification  :',r.qualification.status)
    print('status         :',r.status)
    print('selected       :',r.selected_candidate_id or '-')
    print('output_dir     :',r.output_dir)
    if r.qualification.warnings:
        print('warnings       :','; '.join(r.qualification.warnings))
    if r.selection.reasons:
        print('decision_reason:','; '.join(r.selection.reasons))
    return 0 if r.status in ('SELECTED','FIXED','AMBIGUOUS','PORTFOLIO') else 2


def diagnostic_predict(design_dir: str, data_path: str, out_path: str) -> int:
    from .diagnostic_modeling import predict_diagnostic_design
    p=predict_diagnostic_design(design_dir,data_path,out_path)
    print(f'Wrote diagnostic model scores: {p}')
    return 0

def main(argv=None) -> int:
    p=argparse.ArgumentParser(prog='m2n',description='M2N practitioner toolkit')
    sub=p.add_subparsers(dest='cmd',required=True)
    sub.add_parser('doctor',help='check runtime and compile a minimal M2N graph')
    sub.add_parser('info',help='show toolkit scope and frozen method baseline')
    dfam=sub.add_parser('diagnostic-families',help='show finite tabular diagnostic model families and their task admission scope')
    dfam.add_argument('--task',choices=['FAULT_CLASSIFICATION','ANOMALY_DETECTION'],default=None)
    sub.add_parser('backends',help='show optional scientific-ML backend status/license gate')
    sub.add_parser('nkt',help='print NKT-1 claim ledger and run behavioral adjudication checks')
    sub.add_parser('nkt2',help='print NKT-2 targeted survivor reassessment')
    sp=sub.add_parser('symmatika-probe',help='check whether an external SymMatika source tree can be used by the M2N source-job adapter')
    sp.add_argument('--source-root',default=None)
    sf=sub.add_parser('sbmi-feasibility',help='assess whether a finite M2N model family is large enough to justify SBMI-style inference')
    sf.add_argument('--config',required=True)
    np=sub.add_parser('new-project',help='create a device onboarding starter project')
    np.add_argument('path')
    np.add_argument('--force',action='store_true')
    ip=sub.add_parser('init-design',help='create a declarative device-card + normal-data design project')
    ip.add_argument('path'); ip.add_argument('--template',choices=['mechanics','peak','graph'],default='mechanics'); ip.add_argument('--force',action='store_true')
    vp=sub.add_parser('validate',help='preflight a device card and optionally its normal-data CSV before design')
    vp.add_argument('--spec',required=True); vp.add_argument('--data',default=None); vp.add_argument('--out',default=None)
    idp=sub.add_parser('inspect-data',help='inspect CSV quality, timing, splits and device-card column coverage')
    idp.add_argument('--data',required=True); idp.add_argument('--spec',default=None); idp.add_argument('--out',default=None)
    pv=sub.add_parser('policy-preview',help='preview physics-bias routing and representation eligibility before training')
    pv.add_argument('--spec',required=True); pv.add_argument('--out',default=None)
    mp=sub.add_parser('model-plan',help='qualify model families for an explicit modeling contract without training')
    mp.add_argument('--spec',required=True); mp.add_argument('--data',default=None); mp.add_argument('--out',default=None)
    td=sub.add_parser('temporal-design',help='train and select qualified temporal candidates')
    td.add_argument('--spec',required=True); td.add_argument('--data',required=True); td.add_argument('--out',required=True)
    tp=sub.add_parser('predict-temporal',help='run a frozen temporal model or retained portfolio')
    tp.add_argument('--design-dir',required=True); tp.add_argument('--data',required=True); tp.add_argument('--out',required=True)
    dp=sub.add_parser('design',help='run goal-driven M2N design from device_card.json + normal CSV')
    dp.add_argument('--spec',required=True); dp.add_argument('--data',required=True); dp.add_argument('--out',required=True)
    dd=sub.add_parser('diagnostic-design',help='qualify, train and compare a finite set of fault-diagnostic models from an explicit diagnostic request')
    dd.add_argument('--request',required=True); dd.add_argument('--data',required=True); dd.add_argument('--out',required=True)
    di=sub.add_parser('diagnostic-predict',help='run selected/retained diagnostic model artifacts and export model scores')
    di.add_argument('--design-dir',required=True); di.add_argument('--data',required=True); di.add_argument('--out',required=True)
    pp=sub.add_parser('predict',help='run inference from a goal-driven M2N model artifact')
    pp.add_argument('--artifact',required=True); pp.add_argument('--data',required=True); pp.add_argument('--out',required=True)
    pd=sub.add_parser('predict-design',help='run a selected model or retained finite architecture set from a design bundle')
    pd.add_argument('--design-dir',required=True); pd.add_argument('--data',required=True); pd.add_argument('--out',required=True)
    sb=sub.add_parser('seal-bundle',help='write NEXT_ACTIONS.md and a SHA-256 manifest for a design output directory')
    sb.add_argument('--design-dir',required=True)
    cb=sub.add_parser('check-bundle',help='verify the integrity of a sealed M2N design bundle')
    cb.add_argument('--design-dir',required=True)
    ep=sub.add_parser('evidence',help='generate mechanism residual/model-disagreement/support evidence from a frozen design')
    ep.add_argument('--design-dir',required=True); ep.add_argument('--data',required=True); ep.add_argument('--out',required=True)
    fp=sub.add_parser('fec-export',help='window M2N model evidence into a FEC-compatible Contact Pack')
    fp.add_argument('--evidence',required=True); fp.add_argument('--out',required=True); fp.add_argument('--contract',required=True); fp.add_argument('--window-s',type=float,default=0.05)
    fc=sub.add_parser('fec-calibrate',help='calibrate external FEC Core from independent normal M2N Contact Packs')
    fc.add_argument('--normal-leaf',required=True); fc.add_argument('--normal-root',required=True); fc.add_argument('--fec-root',required=True); fc.add_argument('--out',required=True); fc.add_argument('--trusted-claim',required=True); fc.add_argument('--scope',default='m2n:device')
    fi=sub.add_parser('fec-infer',help='run external FEC Core on an M2N Contact Pack using a frozen reference')
    fi.add_argument('--contacts',required=True); fi.add_argument('--reference',required=True); fi.add_argument('--fec-root',required=True); fi.add_argument('--out',required=True)
    a=p.parse_args(argv)
    if a.cmd=='doctor': return doctor()
    if a.cmd=='info': return info()
    if a.cmd=='diagnostic-families': return diagnostic_families(a.task)
    if a.cmd=='backends': return backends()
    if a.cmd=='nkt': return nkt()
    if a.cmd=='nkt2': return nkt2()
    if a.cmd=='symmatika-probe': return symmatika_probe(a.source_root)
    if a.cmd=='sbmi-feasibility': return sbmi_feasibility(a.config)
    if a.cmd=='new-project': return new_project(a.path,a.force)
    if a.cmd=='init-design': return init_design(a.path,a.template,a.force)
    if a.cmd=='validate': return validate_practitioner(a.spec,a.data,a.out)
    if a.cmd=='inspect-data': return inspect_data_cli(a.data,a.spec,a.out)
    if a.cmd=='policy-preview': return policy_preview(a.spec,a.out)
    if a.cmd=='model-plan': return model_plan(a.spec,a.data,a.out)
    if a.cmd=='temporal-design': return temporal_design(a.spec,a.data,a.out)
    if a.cmd=='predict-temporal': return predict_temporal(a.design_dir,a.data,a.out)
    if a.cmd=='design': return design(a.spec,a.data,a.out)
    if a.cmd=='diagnostic-design': return diagnostic_design(a.request,a.data,a.out)
    if a.cmd=='diagnostic-predict': return diagnostic_predict(a.design_dir,a.data,a.out)
    if a.cmd=='predict': return predict(a.artifact,a.data,a.out)
    if a.cmd=='predict-design': return predict_design(a.design_dir,a.data,a.out)
    if a.cmd=='seal-bundle': return seal_bundle(a.design_dir)
    if a.cmd=='check-bundle': return check_bundle(a.design_dir)
    if a.cmd=='evidence': return evidence(a.design_dir,a.data,a.out)
    if a.cmd=='fec-export': return fec_export(a.evidence,a.out,a.contract,a.window_s)
    if a.cmd=='fec-calibrate': return fec_calibrate(a.normal_leaf,a.normal_root,a.fec_root,a.out,a.trusted_claim,a.scope)
    if a.cmd=='fec-infer': return fec_infer(a.contacts,a.reference,a.fec_root,a.out)
    return 1

if __name__=='__main__':
    raise SystemExit(main())
