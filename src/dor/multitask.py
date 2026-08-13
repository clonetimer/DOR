from __future__ import annotations

"""DOR v0.8 cross-task diagnostic arbitration.

This layer combines already-qualified task outputs without inventing evidence.
It checks logical consistency between Detection, Classification, Localization,
Root Cause, and Identification and returns the strongest defensible diagnostic
scope. Missing optional tasks do not become failures. Contradictory downstream
claims are an ABSTAIN, not an invitation to pick whichever task looks strongest.
"""

from pathlib import Path
from typing import Any
import hashlib, json

CONTRACT = "DOR-MULTITASK-DIAGNOSIS/0.8"
DEPTH_ORDER = ["NONE","DETECTION","CLASSIFICATION","LOCALIZATION","ROOT_CAUSE","IDENTIFICATION"]
VALID_STATES = {"SUPPORTED","ALTERNATIVES","UNKNOWN","ABSTAIN","UNSUPPORTED"}


def _load(x: str | Path | dict[str, Any] | None) -> dict[str, Any] | None:
    if x is None: return None
    if isinstance(x, dict): return x
    return json.loads(Path(x).read_text(encoding="utf-8"))


def _sha(x: str | Path | None) -> str | None:
    if x is None: return None
    p=Path(x); h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def _status(obj: dict[str, Any] | None) -> str:
    if not obj: return "MISSING"
    s=str(obj.get("status", obj.get("diagnosis_status", "UNKNOWN"))).upper()
    return s if s in VALID_STATES else "UNKNOWN"


def _loc_faults(loc: dict[str, Any] | None) -> set[str]:
    if not loc: return set()
    return {str(x.get("fault_id")) for x in loc.get("fault_hypotheses",[]) if x.get("fault_id")}


def _root_faults(root: dict[str, Any] | None) -> set[str]:
    if not root: return set()
    if _status(root)=="SUPPORTED": return set(map(str,root.get("root_cause_set",[])))
    if _status(root)=="ALTERNATIVES":
        out=set()
        for x in root.get("candidate_root_sets",[]): out.update(map(str,x.get("fault_ids",[])))
        return out
    return set()


def _ident_faults(ident: dict[str, Any] | None) -> set[str]:
    if not ident: return set()
    return {str(x.get("fault_id")) for x in ident.get("identifications",[]) if x.get("fault_id") and str(x.get("status","UNKNOWN")).upper() in {"SUPPORTED","ALTERNATIVES"}}


def _class_set(cls: dict[str, Any] | None) -> list[str]:
    if not cls: return []
    raw=cls.get("prediction_set",cls.get("diagnosis_set",cls.get("classes",[])))
    if isinstance(raw,str):
        try: raw=json.loads(raw)
        except Exception: raw=[raw]
    return sorted({str(x) for x in (raw or [])})


def _detection_decision(det: dict[str, Any] | None) -> str | None:
    if not det: return None
    d=str(det.get("diagnosis",det.get("decision",""))).upper()
    return d if d in {"NORMAL","ABNORMAL","OOD","EVIDENCE_INVALID"} else None


def arbitrate_multitask(
    *, detection: dict[str, Any] | None=None, classification: dict[str, Any] | None=None,
    localization: dict[str, Any] | None=None, root_cause: dict[str, Any] | None=None,
    identification: dict[str, Any] | None=None, case_id: str | None=None,
    artifact_bindings: dict[str, str | None] | None=None,
) -> dict[str, Any]:
    supplied={k:v for k,v in {
      "detection":detection,"classification":classification,"localization":localization,
      "root_cause":root_cause,"identification":identification}.items() if v is not None}
    task_states={k:_status(v) for k,v in supplied.items()}
    contradictions=[]; warnings=[]
    det=_detection_decision(detection)
    loc_ids=_loc_faults(localization); root_ids=_root_faults(root_cause); ident_ids=_ident_faults(identification); cls_set=_class_set(classification)

    # Evidence-validity veto at the detection layer. EVIDENCE_INVALID/OOD is not a fault diagnosis.
    if detection and _status(detection)=="SUPPORTED" and det in {"EVIDENCE_INVALID","OOD"}:
        contradictions.append(f"detection={det} cannot support downstream physical fault claims") if (loc_ids or root_ids or ident_ids) else None

    positive_downstream=bool(cls_set or loc_ids or root_ids or ident_ids)
    if detection and _status(detection)=="SUPPORTED" and det=="NORMAL" and positive_downstream:
        contradictions.append("NORMAL detection conflicts with positive classification/localization/root/identification claims")

    if root_cause is not None:
        rs=_status(root_cause); ls=_status(localization)
        if rs in {"SUPPORTED","ALTERNATIVES"}:
            if localization is None or ls not in {"SUPPORTED","ALTERNATIVES"}:
                contradictions.append("root-cause claim lacks a supported/alternative localization scope")
            elif not root_ids.issubset(loc_ids):
                contradictions.append("root-cause fault IDs are not a subset of localized fault hypotheses")

    if identification is not None:
        is_= _status(identification)
        if is_ in {"SUPPORTED","ALTERNATIVES"}:
            allowed = root_ids if root_cause is not None and _status(root_cause) in {"SUPPORTED","ALTERNATIVES"} and root_ids else loc_ids
            if not allowed:
                contradictions.append("identification claim lacks a supported localization/root scope")
            elif not ident_ids.issubset(allowed):
                contradictions.append("identified fault IDs fall outside the retained localization/root scope")
            cand_source=str(identification.get("candidate_source",""))
            if cand_source=="ROOT_CAUSE_SET" and (root_cause is None or _status(root_cause)!="SUPPORTED"):
                contradictions.append("identification says ROOT_CAUSE_SET but no unique supported root-cause set exists")

    # Exact-mode semantic consistency when both localization and identification refer to the same fault.
    if localization and identification:
        loc_modes={str(x.get('fault_id')):str(x.get('fault_mode','')).lower() for x in localization.get('fault_hypotheses',[]) if x.get('fault_id')}
        for x in identification.get('identifications',[]):
            fid=str(x.get('fault_id','')); ch=x.get('identified_change') or {}
            if x.get('status')=="SUPPORTED" and ch.get('exact_mode_identified') is True and fid in loc_modes and ch.get('mode'):
                if str(ch['mode']).lower()!=loc_modes[fid] and loc_modes[fid] not in {"state_fault","abnormal","unknown","unspecified","generic_fault","parameter_drift"}:
                    contradictions.append(f"identified exact mode for {fid} conflicts with localized ontology mode")

    # Optional task abstentions do not erase deeper, independently qualified physical claims.
    # But a localization abstention cannot coexist with root/identification support.
    if localization is not None and _status(localization) in {"ABSTAIN","UNSUPPORTED"} and (root_ids or ident_ids):
        contradictions.append("downstream root/identification claims cannot bypass localization abstention")

    claims={
      "detection":{"status":_status(detection),"decision":det},
      "classification":{"status":_status(classification),"prediction_set":cls_set},
      "localization":{"status":_status(localization),"fault_ids":sorted(loc_ids)},
      "root_cause":{"status":_status(root_cause),"fault_ids":sorted(root_ids)},
      "identification":{"status":_status(identification),"fault_ids":sorted(ident_ids)},
    }

    if contradictions:
        status="ABSTAIN"; depth="NONE"; reason="cross-task semantic contradiction; DOR refuses to choose among inconsistent task outputs"
    else:
        supported_depths=[]
        if detection and _status(detection)=="SUPPORTED" and det in {"NORMAL","ABNORMAL"}: supported_depths.append("DETECTION")
        if classification and _status(classification)=="SUPPORTED" and cls_set: supported_depths.append("CLASSIFICATION")
        if localization and _status(localization)=="SUPPORTED" and loc_ids: supported_depths.append("LOCALIZATION")
        if root_cause and _status(root_cause)=="SUPPORTED" and root_ids: supported_depths.append("ROOT_CAUSE")
        if identification and _status(identification)=="SUPPORTED" and ident_ids: supported_depths.append("IDENTIFICATION")
        depth=max(supported_depths,key=DEPTH_ORDER.index) if supported_depths else "NONE"

        unresolved_alt=False
        # Classification ambiguity is not resolved without an explicit class-to-fault mapping.
        if classification and _status(classification)=="ALTERNATIVES": unresolved_alt=True
        # Localization alternatives can be resolved by a unique supported root set.
        if localization and _status(localization)=="ALTERNATIVES" and not (root_cause and _status(root_cause)=="SUPPORTED" and root_ids): unresolved_alt=True
        if root_cause and _status(root_cause)=="ALTERNATIVES": unresolved_alt=True
        if identification and _status(identification)=="ALTERNATIVES": unresolved_alt=True

        if unresolved_alt:
            status="ALTERNATIVES"; reason="no contradictions, but at least one diagnostic ambiguity remains unresolved at the strongest available scope"
        elif supported_depths:
            status="SUPPORTED"; reason=f"cross-task outputs are consistent; strongest supported diagnostic scope is {depth}"
        else:
            present_states=[_status(x) for x in supplied.values()]
            if present_states and all(s in {"ABSTAIN","UNSUPPORTED"} for s in present_states):
                status="ABSTAIN"; reason="all supplied diagnostic tasks abstained or were unsupported"
            else:
                status="UNKNOWN"; reason="no cross-task contradiction, but no supported diagnostic claim is available"

    return {
      "contract":CONTRACT,"case_id":case_id,"status":status,"reason":reason,"diagnostic_depth":depth,
      "task_states":task_states,"claims":claims,"contradictions":contradictions,"warnings":warnings,
      "artifact_bindings":artifact_bindings or {},
      "safety":{
        "test_truth_allowed":False,
        "detection_is_not_diagnosis":True,
        "classification_is_not_root_cause":True,
        "localization_is_not_root_cause":True,
        "identification_does_not_imply_root_cause":True,
        "root_cause_claimed":bool(root_cause and _status(root_cause)=="SUPPORTED" and root_ids and not contradictions),
        "missing_optional_task_does_not_create_evidence":True,
      }
    }


def arbitrate_multitask_from_files(*, detection_path=None, classification_path=None, localization_path=None, root_cause_path=None, identification_path=None, case_id=None) -> dict[str,Any]:
    paths={"detection":detection_path,"classification":classification_path,"localization":localization_path,"root_cause":root_cause_path,"identification":identification_path}
    objs={k:_load(v) for k,v in paths.items()}
    binds={f"{k}_sha256":_sha(v) for k,v in paths.items() if v is not None}
    return arbitrate_multitask(**objs,case_id=case_id,artifact_bindings=binds)
