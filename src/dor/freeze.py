from __future__ import annotations

"""DOR 1.0 development-freeze gate.

This is a release decision contract, not a mechanism for overriding missing
evidence.  A PRE-RC may be built with status BLOCKED; it must not be renamed
1.0 final until every required gate is PASS.
"""

from pathlib import Path
from typing import Any
import json

CONTRACT="DOR-DEVELOPMENT-FREEZE-GATE/1.0"
REQUIRED_CORE_GATES=[
    "CONTROL_PLANE_CONTRACTS",
    "CONTEXT_REFERENCE_POLICY",
    "FAULT_SEMANTICS_LOCALIZATION",
    "ROOT_CAUSE_REAL_HARDWARE",
    "IDENTIFICATION_DUAL_ROUTE",
    "MULTITASK_ARBITRATION",
    "REGRESSION_AND_SCHEMA_VERIFICATION",
    "SATELLITE_VERTICAL",
]


def _load(x: str|Path|dict[str,Any]) -> dict[str,Any]:
    if isinstance(x,dict): return x
    return json.loads(Path(x).read_text(encoding='utf-8'))


def evaluate_freeze_gate(evidence: dict[str,Any], vertical: dict[str,Any]) -> dict[str,Any]:
    gates={str(x.get('gate')):x for x in evidence.get('gates',[]) if x.get('gate')}
    results=[]; missing=[]
    for gate in REQUIRED_CORE_GATES:
        if gate=='SATELLITE_VERTICAL':
            passed=vertical.get('status')=='FULLY_VALIDATED'
            reason='satellite v0.9 vertical is fully validated' if passed else f"satellite vertical status={vertical.get('status')}; missing={vertical.get('missing_gates',[])}"
            item={'gate':gate,'status':'PASS' if passed else 'BLOCKED','reason':reason,'evidence':[vertical.get('contract')]}
        else:
            src=gates.get(gate,{})
            passed=str(src.get('status','BLOCKED')).upper()=='PASS'
            item={'gate':gate,'status':'PASS' if passed else 'BLOCKED','reason':str(src.get('reason','required gate evidence missing')),'evidence':src.get('evidence',[])}
        results.append(item)
        if item['status']!='PASS': missing.append(gate)
    ready=not missing
    return {
        'contract':CONTRACT,
        'status':'READY_FOR_1_0_FREEZE' if ready else 'BLOCKED',
        'reason':'all development exit gates passed; 1.0 final packaging is permitted' if ready else 'development freeze is blocked by one or more explicit exit gates',
        'gates':results,
        'blocking_gates':missing,
        'vertical_status':vertical.get('status'),
        'safety':{
            'pre_rc_may_be_blocked':True,
            'blocked_pre_rc_must_not_be_renamed_final':True,
            'missing_satellite_task_evidence_cannot_be_waived_by_cross_domain_success':True,
            'test_performance_cannot_override_contract_gate':True,
            'application_stage_requires_core_contract_stability':True,
        }
    }


def evaluate_freeze_gate_from_files(evidence_path: str|Path, vertical_path: str|Path) -> dict[str,Any]:
    return evaluate_freeze_gate(_load(evidence_path),_load(vertical_path))
