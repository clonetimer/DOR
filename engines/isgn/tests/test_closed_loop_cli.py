import json, os, subprocess, sys
from pathlib import Path


def test_closed_loop_cli_plan_and_evaluate(tmp_path):
    before={
      'system_id':'demo',
      'observation':{'status':'OBSERVATION_INSUFFICIENT','blocked_states':['flow'],'summary':{'dt':.002},
        'config':{'fail_derivative_noise_ratio':2.0,'fail_measurement_variation_ratio':2.0},
        'states':[{'state':'flow','status':'FAIL','derivative_noise_ratio':2.5,'sensor_sample_period_s':.001,'variation_to_sensor_floor':50.0}]}}
    after={'system_id':'demo','observation':{'status':'QUALIFIED','blocked_states':[],'summary':{'dt':.001},
      'config':{'fail_derivative_noise_ratio':2.0,'fail_measurement_variation_ratio':2.0},
      'states':[{'state':'flow','status':'PASS','derivative_noise_ratio':.8,'sensor_sample_period_s':.001,'variation_to_sensor_floor':50.0}]}}
    b=tmp_path/'before.json'; af=tmp_path/'after.json'; plan=tmp_path/'plan.json'; ev=tmp_path/'eval.json'
    b.write_text(json.dumps(before)); af.write_text(json.dumps(after))
    env=dict(os.environ); env['PYTHONPATH']=str(Path(__file__).resolve().parents[1]/'src')
    r=subprocess.run([sys.executable,'-m','isgn.cli','plan-closed-loop','--evidence',str(b),'--out',str(plan)],env=env,capture_output=True,text=True)
    assert r.returncode==0, r.stderr
    r=subprocess.run([sys.executable,'-m','isgn.cli','evaluate-closed-loop','--plan',str(plan),'--after',str(af),'--out',str(ev)],env=env,capture_output=True,text=True)
    assert r.returncode==0, r.stderr
    assert json.loads(ev.read_text())['status']=='CLOSED'
