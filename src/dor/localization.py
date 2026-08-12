from __future__ import annotations

"""Conservative v0.5 fault-localization reasoner.

It localizes *candidate component/fault hypotheses* from provenance-bound,
valid evidence.  It does not infer causal root cause or propagation; those are
v0.6 responsibilities.  The reasoner prefers explicit fault signatures and
preserves alternatives when evidence cannot distinguish hypotheses.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json
import math
import pandas as pd

from .ontology import ontology_index, validate_fault_ontology, validate_mechanism_component_map

LOCALIZATION_CONTRACT = "DOR-LOCALIZATION-RESULT/0.5"
EVIDENCE_VALIDITY = {"VALID", "POWER_LIMITED", "CONTEXT_UNSUPPORTED", "CALIBRATION_UNSUPPORTED", "INVALID"}
EVIDENCE_STATES = {"ABNORMAL", "NORMAL", "HIGH", "LOW", "STUCK", "OFFSET", "NO_RESPONSE", "FAILED_OFF", "OPEN", "CLOSED", "DEGRADED", "CLASS", "MISSING", "OOD"}


def _as_list(x: Any) -> list[Any]:
    if x is None: return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _state_matches(observed: str, expected: str, observed_value: Any=None, expected_value: Any=None) -> bool:
    observed = observed.upper(); expected = expected.upper()
    if observed == expected == "CLASS":
        return str(observed_value) == str(expected_value)
    if observed == expected:
        return True
    if expected == "ABNORMAL" and observed not in {"NORMAL", "MISSING", "OOD"}:
        return True
    return False


def _contradicts(observed: str, expected: str, observed_value: Any=None, expected_value: Any=None) -> bool:
    observed = observed.upper(); expected = expected.upper()
    if observed == expected == "CLASS":
        return str(observed_value) != str(expected_value)
    if observed in {"MISSING", "OOD"}: return False
    if expected == "ABNORMAL" and observed == "NORMAL": return True
    if expected == "NORMAL" and observed not in {"NORMAL"}: return True
    # Specific mutually-exclusive sensor states are contradictory when explicit.
    pairs = {("STUCK", "OFFSET"), ("OFFSET", "STUCK"), ("OPEN", "CLOSED"), ("CLOSED", "OPEN"), ("HIGH", "LOW"), ("LOW", "HIGH")}
    return (observed, expected) in pairs


def _evidence_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("scope_type", "")).upper(), str(row.get("scope_id", ""))


def _read_evidence(path: str | Path) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    required = {"evidence_id", "scope_type", "scope_id", "state", "strength", "validity"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"evidence CSV missing columns: {missing}")
    rows=[]
    for _, r in df.iterrows():
        d={k:(None if (isinstance(v,float) and math.isnan(v)) else v) for k,v in r.to_dict().items()}
        d["scope_type"] = str(d["scope_type"]).upper(); d["state"] = str(d["state"]).upper(); d["validity"] = str(d["validity"]).upper()
        if d["scope_type"] not in {"SENSOR","COMPONENT","MECHANISM","COMMAND"}:
            raise ValueError(f"invalid evidence scope_type: {d['scope_type']}")
        if d["state"] not in EVIDENCE_STATES:
            raise ValueError(f"invalid evidence state: {d['state']}")
        if d["validity"] not in EVIDENCE_VALIDITY:
            raise ValueError(f"invalid evidence validity: {d['validity']}")
        d["strength"] = float(d["strength"])
        if not (0 <= d["strength"] <= 1):
            raise ValueError("evidence strength must be within [0,1]")
        rows.append(d)
    return rows


def _mechanism_bindings(mechanism_map: dict[str, Any] | None) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    if not mechanism_map: return out
    for b in _as_list(mechanism_map.get("bindings")):
        out.setdefault(str(b.get("mechanism_id")), set()).update(str(x) for x in _as_list(b.get("component_ids")))
    return out


def localize_faults(
    ontology: dict[str, Any], evidence_rows: list[dict[str, Any]], *,
    mechanism_map: dict[str, Any] | None = None,
    min_support: float = 0.45,
    retain_margin: float = 0.08,
    max_contradiction_fraction: float = 0.35,
) -> dict[str, Any]:
    ov = validate_fault_ontology(ontology)
    if ov.state != "PASS":
        raise ValueError("invalid ontology: " + "; ".join(ov.errors))
    if mechanism_map:
        mv = validate_mechanism_component_map(mechanism_map, ontology)
        if mv.state != "PASS":
            raise ValueError("invalid mechanism map: " + "; ".join(mv.errors))

    idx=ontology_index(ontology)
    valid=[r for r in evidence_rows if r["validity"]=="VALID"]
    invalid=[r for r in evidence_rows if r["validity"]!="VALID"]
    if not valid:
        return {
            "contract":LOCALIZATION_CONTRACT,"status":"ABSTAIN","reason":"no VALID evidence available for localization",
            "fault_hypotheses":[],"all_scored_hypotheses":[],"component_candidates":[],"candidate_fault_sets":[],"supported_fault_set":[],"ambiguous_exclusion_groups":{},
            "evidence_summary":{"total":len(evidence_rows),"valid":0,"invalid_or_unsupported":len(invalid)},
            "policy":{"min_support":min_support,"retain_margin":retain_margin,"max_contradiction_fraction":max_contradiction_fraction,"candidate_set_cap":32},
            "safety":{"root_cause_claimed":False,"propagation_inferred":False,"invalid_evidence_used_for_scoring":False,"generic_component_or_mechanism_evidence_cannot_by_itself_identify_fault_mode":True,"concurrent_faults_are_not_relabelled_as_alternatives":True},
        }

    by_key: dict[tuple[str,str], list[dict[str,Any]]] = {}
    for r in valid: by_key.setdefault(_evidence_key(r),[]).append(r)
    mech_to_comp=_mechanism_bindings(mechanism_map)

    scored=[]
    for fid,f in idx["faults"].items():
        support=0.0; contradiction=0.0; total_weight=0.0; matched=[]; contrad=[]; missing=[]
        signatures=_as_list(f.get("expected_evidence"))
        for sig in signatures:
            st=str(sig.get("scope_type","SENSOR")).upper(); sid=str(sig.get("scope_id","")); exp=str(sig.get("expected_state","ABNORMAL")).upper(); w=float(sig.get("weight",1.0)); total_weight+=w
            evs=by_key.get((st,sid),[])
            if not evs:
                missing.append({"scope_type":st,"scope_id":sid,"expected_state":exp,"weight":w}); continue
            exp_val=sig.get("expected_value")
            best_match=max((r["strength"] for r in evs if _state_matches(r["state"],exp,r.get("value"),exp_val)), default=0.0)
            best_contra=max((r["strength"] for r in evs if _contradicts(r["state"],exp,r.get("value"),exp_val)), default=0.0)
            if best_match>0:
                support += w*best_match
                matched.extend(str(r["evidence_id"]) for r in evs if _state_matches(r["state"],exp,r.get("value"),exp_val))
            if best_contra>0:
                contradiction += w*best_contra
                contrad.extend(str(r["evidence_id"]) for r in evs if _contradicts(r["state"],exp,r.get("value"),exp_val))
        # explicit contradiction signatures add hard negative evidence
        for sig in _as_list(f.get("contradictions")):
            st=str(sig.get("scope_type","SENSOR")).upper(); sid=str(sig.get("scope_id","")); state=str(sig.get("state","ABNORMAL")).upper(); w=float(sig.get("weight",1.0))
            for r in by_key.get((st,sid),[]):
                if _state_matches(r["state"],state,r.get("value"),sig.get("value")):
                    contradiction += w*r["strength"]; contrad.append(str(r["evidence_id"]))

        # Generic component/mechanism evidence can support *scope localization* but
        # never provides more than 25% of a fault's score because it does not identify the mode.
        target=str(f.get("target_component_id"))
        generic=0.0
        for r in valid:
            if r["scope_type"]=="COMPONENT" and r["scope_id"]==target and r["state"]=="ABNORMAL": generic=max(generic,r["strength"])
            if r["scope_type"]=="MECHANISM" and target in mech_to_comp.get(str(r["scope_id"]),set()) and r["state"]=="ABNORMAL": generic=max(generic,r["strength"])
        denom=max(total_weight,1.0)
        explicit_score=max(0.0,(support-contradiction)/denom)
        scope_bonus=0.25*generic
        score=max(0.0,min(1.0,explicit_score+scope_bonus))
        contra_frac=min(1.0,contradiction/denom)
        scored.append({
            "fault_id":fid,"target_component_id":target,"fault_mode":f.get("fault_mode"),"category":f.get("category"),
            "score":score,"explicit_support":support/denom,"scope_bonus":scope_bonus,"contradiction_fraction":contra_frac,
            "matched_evidence":sorted(set(matched)),"contradicting_evidence":sorted(set(contrad)),"missing_expected":missing,
            "eligible": bool(score>=min_support and contra_frac<=max_contradiction_fraction),
        })

    eligible=[x for x in scored if x["eligible"]]
    if not eligible:
        return {
            "contract":LOCALIZATION_CONTRACT,"status":"UNKNOWN","reason":"valid evidence exists but no declared fault hypothesis meets localization support/contradiction gates",
            "fault_hypotheses":sorted(scored,key=lambda x:x["score"],reverse=True),"component_candidates":[],
            "candidate_fault_sets":[],"supported_fault_set":[],
            "evidence_summary":{"total":len(evidence_rows),"valid":len(valid),"invalid_or_unsupported":len(invalid)},
            "policy":{"min_support":min_support,"retain_margin":retain_margin,"max_contradiction_fraction":max_contradiction_fraction,"candidate_set_cap":32},
            "safety":{"root_cause_claimed":False,"propagation_inferred":False,"invalid_evidence_used_for_scoring":False,"generic_component_or_mechanism_evidence_cannot_by_itself_identify_fault_mode":True,"concurrent_faults_are_not_relabelled_as_alternatives":True},
        }

    # v0.5 explicitly supports concurrent faults.  `exclusion_group` declares
    # mutually-exclusive modes (e.g. weak vs severe leakage on one component).
    # Faults in different groups are allowed to co-exist; this is essential for
    # ADAPT-style double/triple-fault scenarios and avoids turning concurrency
    # into a false ALTERNATIVES result.
    by_group: dict[str,list[dict[str,Any]]] = {}
    for x in eligible:
        f=idx["faults"][x["fault_id"]]
        gid=str(f.get("exclusion_group") or x["fault_id"])
        x["exclusion_group"]=gid
        by_group.setdefault(gid,[]).append(x)

    retained_groups: dict[str,list[dict[str,Any]]] = {}
    for gid,members in by_group.items():
        gbest=max(x["score"] for x in members)
        retained_groups[gid]=sorted([x for x in members if gbest-x["score"]<=retain_margin],key=lambda x:(-x["score"],x["fault_id"]))

    retained=[x for gid in sorted(retained_groups) for x in retained_groups[gid]]
    comp_scores: dict[str,float]={}; comp_faults: dict[str,list[str]]={}
    for x in retained:
        cid=x["target_component_id"]; comp_scores[cid]=max(comp_scores.get(cid,0.0),x["score"]); comp_faults.setdefault(cid,[]).append(x["fault_id"])
    comps=[{"component_id":cid,"score":comp_scores[cid],"retained_fault_ids":sorted(comp_faults[cid])} for cid in sorted(comp_scores,key=lambda c:(-comp_scores[c],c))]

    # Cartesian product across ambiguous exclusion groups; hard cap prevents a
    # combinatorial explosion from being silently presented as certainty.
    candidate_sets=[[]]
    overflow=False
    for gid in sorted(retained_groups):
        choices=retained_groups[gid]
        nxt=[]
        for base in candidate_sets:
            for choice in choices:
                nxt.append(base+[choice["fault_id"]])
                if len(nxt)>32:
                    overflow=True; break
            if overflow: break
        candidate_sets=nxt[:32]
        if overflow: break

    ambiguous_groups={gid:[x["fault_id"] for x in xs] for gid,xs in retained_groups.items() if len(xs)>1}
    if overflow:
        status="ABSTAIN"; reason="candidate fault-set expansion exceeds safety cap; additional evidence or ontology constraints required"
        supported_set=[]
    elif ambiguous_groups:
        status="ALTERNATIVES"; reason="one or more mutually-exclusive fault groups remain ambiguous"
        supported_set=[]
    else:
        status="SUPPORTED"; reason="all retained fault groups are individually resolved; concurrent faults across groups are allowed"
        supported_set=candidate_sets[0] if candidate_sets else []

    return {
        "contract":LOCALIZATION_CONTRACT,"status":status,"reason":reason,
        "fault_hypotheses":retained,"all_scored_hypotheses":sorted(scored,key=lambda x:x["score"],reverse=True),"component_candidates":comps,
        "supported_fault_set":supported_set,"candidate_fault_sets":candidate_sets,"ambiguous_exclusion_groups":ambiguous_groups,
        "evidence_summary":{"total":len(evidence_rows),"valid":len(valid),"invalid_or_unsupported":len(invalid)},
        "policy":{"min_support":min_support,"retain_margin":retain_margin,"max_contradiction_fraction":max_contradiction_fraction,"candidate_set_cap":32},
        "safety":{
            "root_cause_claimed":False,"propagation_inferred":False,"invalid_evidence_used_for_scoring":False,
            "generic_component_or_mechanism_evidence_cannot_by_itself_identify_fault_mode":True,
            "concurrent_faults_are_not_relabelled_as_alternatives":True,
        },
    }


def localize_from_files(ontology_path: str | Path, evidence_path: str | Path, *, mechanism_map_path: str | Path | None=None, **kwargs: Any) -> dict[str,Any]:
    ontology=json.loads(Path(ontology_path).read_text(encoding="utf-8"))
    mechanism_map=json.loads(Path(mechanism_map_path).read_text(encoding="utf-8")) if mechanism_map_path else None
    rows=_read_evidence(evidence_path)
    return localize_faults(ontology,rows,mechanism_map=mechanism_map,**kwargs)

BATCH_LOCALIZATION_CONTRACT = "DOR-LOCALIZATION-BATCH/0.5"


def localize_batch_from_files(
    ontology_path: str | Path, evidence_path: str | Path, *,
    mechanism_map_path: str | Path | None=None, case_column: str="case_id", **kwargs: Any
) -> dict[str,Any]:
    ontology=json.loads(Path(ontology_path).read_text(encoding="utf-8"))
    mechanism_map=json.loads(Path(mechanism_map_path).read_text(encoding="utf-8")) if mechanism_map_path else None
    rows=_read_evidence(evidence_path)
    if not rows or case_column not in rows[0]:
        raise ValueError(f"batch evidence requires {case_column!r} column")
    grouped: dict[str,list[dict[str,Any]]] = {}
    for r in rows:
        cid=str(r.get(case_column))
        grouped.setdefault(cid,[]).append(r)
    cases=[]
    status_counts: dict[str,int]={}
    for cid in sorted(grouped):
        result=localize_faults(ontology,grouped[cid],mechanism_map=mechanism_map,**kwargs)
        status_counts[result["status"]]=status_counts.get(result["status"],0)+1
        cases.append({"case_id":cid,"result":result})
    return {
        "contract":BATCH_LOCALIZATION_CONTRACT,
        "case_column":case_column,
        "n_cases":len(cases),
        "status_counts":status_counts,
        "cases":cases,
        "safety":{"test_truth_used_for_reasoning":False,"root_cause_claimed":False,"propagation_inferred":False},
    }
