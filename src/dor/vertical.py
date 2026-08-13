from __future__ import annotations

"""DOR v0.9 vertical evidence qualification.

The vertical validator is deliberately not a performance aggregator.  It records
what diagnostic task has actually been exercised, in which domain, and at what
evidence grade.  A strong result in an unrelated domain cannot silently satisfy a
satellite-domain gate, and a source-anchored component fixture cannot be upgraded
to a high-fidelity system simulation merely because its numerical error is small.
"""

from pathlib import Path
from typing import Any
import hashlib, json

CONTRACT = "DOR-VERTICAL-VALIDATION/0.9"
TASKS = ["DETECTION","CLASSIFICATION","LOCALIZATION","ROOT_CAUSE","IDENTIFICATION"]
GRADES = [
    "UNIT_TEST",
    "CONTROLLED_SYNTHETIC",
    "SOURCE_ANCHORED_SYNTHETIC_MECHANISM",
    "HIGH_FIDELITY_SYSTEM_SIMULATION",
    "REAL_HARDWARE_TESTBED",
    "REAL_IN_ORBIT",
]
GRADE_RANK = {g:i for i,g in enumerate(GRADES)}
VALID_CASE_STATES={"PASS","PARTIAL","FAIL","NOT_RUN"}


def _load(x: str|Path|dict[str,Any]) -> dict[str,Any]:
    if isinstance(x,dict): return x
    return json.loads(Path(x).read_text(encoding="utf-8"))


def _sha(path: str|Path|None) -> str|None:
    if path is None: return None
    p=Path(path); h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()


def _grade_ok(actual: str, required: str) -> bool:
    return GRADE_RANK.get(actual,-1) >= GRADE_RANK.get(required,10**9)


def validate_vertical_registry(registry: dict[str,Any]) -> list[str]:
    errors=[]
    if registry.get("target_domain") is None: errors.append("target_domain is required")
    ids=set()
    for i,c in enumerate(registry.get("cases",[])):
        cid=str(c.get("case_id",""))
        if not cid: errors.append(f"case[{i}] missing case_id")
        if cid in ids: errors.append(f"duplicate case_id: {cid}")
        ids.add(cid)
        if str(c.get("task","")).upper() not in TASKS: errors.append(f"{cid}: unsupported task")
        if c.get("evidence_grade") not in GRADES: errors.append(f"{cid}: unsupported evidence_grade")
        if str(c.get("state","PASS")).upper() not in VALID_CASE_STATES: errors.append(f"{cid}: invalid state")
        if not c.get("domain"): errors.append(f"{cid}: domain required")
        if not c.get("provenance"): errors.append(f"{cid}: provenance required")
    return errors


def evaluate_vertical(registry: dict[str,Any]) -> dict[str,Any]:
    errors=validate_vertical_registry(registry)
    target=str(registry.get("target_domain",""))
    requirements={str(k).upper():v for k,v in registry.get("requirements",{}).items()}
    summaries={}
    for task in TASKS:
        req=requirements.get(task)
        if req is None:
            continue
        required_grade=str(req.get("minimum_evidence_grade","UNIT_TEST"))
        require_target=bool(req.get("require_target_domain",True))
        eligible=[]; supporting=[]
        for c in registry.get("cases",[]):
            if str(c.get("task","")).upper()!=task: continue
            state=str(c.get("state","PASS")).upper()
            if state not in {"PASS","PARTIAL"}: continue
            supporting.append(c)
            domain_ok=(str(c.get("domain"))==target) if require_target else True
            if domain_ok and state=="PASS" and _grade_ok(str(c.get("evidence_grade")),required_grade): eligible.append(c)
        best=max((GRADE_RANK.get(str(c.get("evidence_grade")),-1) for c in supporting if (not require_target or str(c.get("domain"))==target)),default=-1)
        summaries[task]={
            "required_grade":required_grade,
            "require_target_domain":require_target,
            "meets_requirement":bool(eligible),
            "qualifying_case_ids":[str(c.get("case_id")) for c in eligible],
            "supporting_case_ids":[str(c.get("case_id")) for c in supporting],
            "best_target_grade":GRADES[best] if best>=0 else None,
        }
    missing=[t for t,x in summaries.items() if not x["meets_requirement"]]
    if errors:
        status="BLOCKED"; reason="vertical registry is invalid"
    elif not missing:
        status="FULLY_VALIDATED"; reason=f"all declared {target} vertical task gates meet their minimum evidence grades"
    else:
        status="PARTIALLY_VALIDATED"; reason="some vertical task gates remain below the declared evidence/domain requirement"
    return {
        "contract":CONTRACT,
        "target_domain":target,
        "status":status,
        "reason":reason,
        "task_summary":summaries,
        "missing_gates":missing,
        "cases":[{
            "case_id":c.get("case_id"),"domain":c.get("domain"),"subsystem":c.get("subsystem"),"task":str(c.get("task","")).upper(),
            "evidence_grade":c.get("evidence_grade"),"state":str(c.get("state","PASS")).upper(),"artifact":c.get("artifact"),
            "artifact_sha256":c.get("artifact_sha256"),"provenance":c.get("provenance"),"limitations":c.get("limitations",[])
        } for c in registry.get("cases",[])],
        "errors":errors,
        "safety":{
            "cross_domain_evidence_cannot_satisfy_target_domain_gate":True,
            "evidence_grade_is_not_performance_score":True,
            "source_anchored_fixture_is_not_high_fidelity_system_simulation":True,
            "real_hardware_testbed_is_not_real_in_orbit":True,
            "missing_task_evidence_is_not_inferred":True,
        }
    }


def evaluate_vertical_from_file(path: str|Path) -> dict[str,Any]:
    return evaluate_vertical(_load(path))
