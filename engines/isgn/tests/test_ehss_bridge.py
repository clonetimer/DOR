import csv
import json
from pathlib import Path

import numpy as np
import pytest

from isgn.bridges.ehss import load_ehss_scenario
from isgn.grammar import HierarchicalISGNGraph


MEAS = [
    ("position", "position_m", 0.001),
    ("velocity", "velocity_m_s", 0.001),
    ("pressure_a", "pressure_a_pa", 0.001),
    ("pressure_b", "pressure_b_pa", 0.001),
    ("temperature_a", "temperature_a_k", 0.010),
    ("temperature_b", "temperature_b_k", 0.010),
    ("supply_pressure", "supply_pressure_pa", 0.001),
    ("supply_flow", "supply_flow_m3_s", 0.001),
    ("valve_spool", "valve_spool_m", 0.001),
    ("valve_current", "valve_current_a", 0.001),
    ("pump_speed", "pump_speed_rad_s", 0.002),
]


def _scenario(tmp_path: Path, *, faulted=False, invalid_position=False):
    p=tmp_path/'scenario'; p.mkdir()
    sensors=[]
    for ch, suffix, sp in MEAS:
        sensors.append({"channel":ch,"bias":0.0,"scale":1.0,"noise_std":0.0,"sample_period_s":sp,"delay_s":0.0,"quantization_step":0.0,"dropout_probability":0.0,"dropout_hold_last":True})
    faults=[{"id":"f"}] if faulted else []
    meta={"schema_version":"ehss-diagnosis-dataset/1.3","model_version":"0.5.0","parameter_profile":"diagnosis_extended","experiment_id":"motion","task_id":"multistep","scenario_id":"normal_test" if not faulted else "fault_test","split":"train","seed":7,"faults":faults,"sensors":sensors,"sample_count":40,"finite":True}
    (p/'metadata.json').write_text(json.dumps(meta),encoding='utf-8')
    base=["schema_version","scenario_id","split","experiment_id","task_id","seed","time_s","active_fault_count","command_m","supply_pressure_pa","reservoir_temperature_k","external_load_n"]
    fields=list(base)
    for _, suffix, _ in MEAS:
        c='measurement_'+suffix; fields += [c,c+'_valid',c+'_updated']
    for _, suffix, _ in MEAS: fields.append('truth_'+suffix)
    with (p/'timeseries.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for i in range(40):
            t=0.002*i
            r={"schema_version":meta["schema_version"],"scenario_id":meta["scenario_id"],"split":"train","experiment_id":"motion","task_id":"multistep","seed":"7","time_s":f"{t:.6f}","active_fault_count":"1" if faulted and i>=20 else "0","command_m":str(np.sin(13*t)),"supply_pressure_pa":str(1.2e7+1e5*(i>=10)),"reservoir_temperature_k":str(313.15+5*(i>=20)),"external_load_n":str(100*(i>=30))}
            for j,(ch,suffix,sp) in enumerate(MEAS):
                c='measurement_'+suffix
                value=(j+1)*0.1 + np.sin((j+1)*t)
                r[c]=str(value); r[c+'_valid']='0' if invalid_position and ch=='position' and i==10 else '1'; r[c+'_updated']='1' if sp<=0.002 or i%5==0 else '0'; r['truth_'+suffix]=str(value-0.001)
            w.writerow(r)
    return p


def test_real_contract_shapes_and_pressure_dimension(tmp_path):
    p=_scenario(tmp_path)
    out=load_ehss_scenario(p,view='measured-fast')
    assert out.dataset.X.shape==(1,40,9)
    assert out.dataset.U.shape==(1,39,4)
    assert out.dataset.sample_time_s is not None
    assert out.dataset.sample_time_s.shape == out.dataset.X.shape[:2]
    assert out.contract['production_mechanism_eligible'] is True
    names=[x['name'] for x in out.dataset.schema['states']]
    assert 'temperature_a' not in names and 'pressure_a' in names
    # P2 must be discovery-compatible, not merely Gate-compatible.
    g=HierarchicalISGNGraph.from_schema(out.dataset.schema,out.dataset.dt,seed=1)
    assert len(g.state_specs)==9


def test_full_and_truth_views_are_explicit(tmp_path):
    p=_scenario(tmp_path)
    full=load_ehss_scenario(p,view='measured-full')
    assert full.dataset.X.shape[-1]==11
    assert full.dataset.schema['states'][4]['sensor']['sample_period_s']==0.010
    truth=load_ehss_scenario(p,view='truth-audit')
    assert truth.contract['production_mechanism_eligible'] is False
    assert all(s['observation_role']=='truth' for s in truth.dataset.schema['states'])


def test_bridge_rejects_invalid_selected_measurement(tmp_path):
    p=_scenario(tmp_path,invalid_position=True)
    with pytest.raises(ValueError,match='invalid rows'):
        load_ehss_scenario(p,view='measured-fast')


def test_faulted_scenario_is_blocked_by_default(tmp_path):
    p=_scenario(tmp_path,faulted=True)
    with pytest.raises(ValueError,match='faulted EHSS scenarios are blocked'):
        load_ehss_scenario(p)
    out=load_ehss_scenario(p,allow_faulted=True)
    assert out.dataset.X.shape[0]==1


def test_bridge_accepts_ehss_1ms_writer_jitter(tmp_path):
    p=_scenario(tmp_path)
    csv_path=p/'timeseries.csv'
    rows=list(csv.DictReader(csv_path.open(newline='',encoding='utf-8')))
    fields=list(rows[0].keys())
    # Match the real EHSS 1 ms writer pattern observed in P3: +/-10 ns.
    for i,row in enumerate(rows):
        row['time_s']=f"{0.001*i + (1e-8 if i%2 else 0.0):.12f}"
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    out=load_ehss_scenario(p,view='actuator-core')
    assert abs(out.dataset.dt-0.001) < 2e-8
