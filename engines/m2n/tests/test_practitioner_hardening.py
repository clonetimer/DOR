from __future__ import annotations
import csv
import json
from pathlib import Path

from m2n.cli import init_design
from m2n.goal_design import run_goal_driven_design
from m2n.practitioner import (
    inspect_tabular_data,
    predict_design_bundle,
    seal_design_bundle,
    validate_device_card,
    verify_design_bundle,
    write_next_actions,
)


def test_validate_and_inspect_generated_project(tmp_path: Path):
    project=tmp_path/'project'
    assert init_design(str(project),'graph',False)==0
    vr=validate_device_card(project/'device_card.json')
    assert vr.status in ('PASS','WARN'), vr
    dr=inspect_tabular_data(project/'data'/'normal_data.csv',project/'device_card.json')
    assert dr.status in ('PASS','WARN'), dr
    assert dr.diagnostics['rows'] == 360
    assert dr.diagnostics['split']['train_rows'] == 240


def test_inspect_catches_bad_time_and_missing_columns(tmp_path: Path):
    spec={
        'schema_version':'1.0','template':'second_order_mechanics',
        'data':{'split_column':'split','train_values':['train'],'validation_values':['val']},
        'mechanism':{},'knowledge':{},
        'supervision':{'position_column':'q','velocity_column':'v','input_columns':{'u':'u'},'derive_acceleration':{'time_column':'time','method':'gradient'}},
    }
    sp=tmp_path/'device_card.json'; sp.write_text(json.dumps(spec),encoding='utf-8')
    dp=tmp_path/'data.csv'
    with dp.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['q','v','time','split']); w.writerows([[0,0,0,'train'],[0,1,0,'train'],[0,2,1,'val'],[0,3,2,'val']])
    r=inspect_tabular_data(dp,sp)
    assert r.status=='FAIL'
    assert any('missing columns' in x for x in r.errors)
    assert any('not strictly increasing' in x for x in r.errors)


def test_design_bundle_seal_verify_and_predict(tmp_path: Path):
    project=tmp_path/'project'
    init_design(str(project),'mechanics',False)
    spec=json.loads((project/'device_card.json').read_text())
    # Make the smoke fast and deterministic while preserving at least two restarts.
    spec['training']['epochs']=60
    spec['training']['restarts']=[101,102]
    spec['training']['final_seed']=9001
    (project/'device_card.json').write_text(json.dumps(spec),encoding='utf-8')
    out=project/'output'
    result=run_goal_driven_design(project/'device_card.json',project/'data'/'normal_data.csv',out)
    assert result.status in ('SELECTED','FIXED','AMBIGUOUS')
    write_next_actions(out)
    seal_design_bundle(out)
    check=verify_design_bundle(out)
    assert check.status=='PASS', check
    pred=tmp_path/'predictions.csv'
    predict_design_bundle(out,project/'data'/'normal_data.csv',pred)
    header=pred.read_text(encoding='utf-8').splitlines()[0]
    assert 'portfolio_mean__a' in header
    assert 'portfolio_std__a' in header
    # Mutating a sealed bundle must be detected.
    (out/'NEXT_ACTIONS.md').write_text('mutated',encoding='utf-8')
    check2=verify_design_bundle(out)
    assert check2.status=='FAIL'
