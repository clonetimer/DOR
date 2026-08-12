"""P5: frozen M2N design -> model evidence -> FEC Contact Pack.

This example intentionally stops at the FEC handoff. Actual FEC calibration/inference
requires an external FEC Core plus trusted healthy references.
"""
from __future__ import annotations
import tempfile
from pathlib import Path
from m2n.cli import init_design
from m2n import run_goal_driven_design, generate_model_evidence, export_fec_contact_pack

with tempfile.TemporaryDirectory(prefix='m2n_p5_') as td:
    td=Path(td); init_design(str(td/'device'),'graph')
    device=td/'device'
    r=run_goal_driven_design(device/'device_card.json',device/'data'/'normal_data.csv',device/'output')
    assert r.status in {'SELECTED','FIXED','AMBIGUOUS','K3_SELECTED'}
    evidence=generate_model_evidence(device/'output',device/'data'/'normal_data.csv',td/'evidence.csv')
    contacts,contract=export_fec_contact_pack(evidence,td/'contacts.csv',td/'fec_contract.json',window_s=20.0,min_samples=3)
    print('design_status=',r.status)
    print('evidence=',evidence)
    print('contacts=',contacts)
    print('contract=',contract)
    assert contacts.is_file() and contract.is_file()
    print('P5_FEC_HANDOFF_PASS')
