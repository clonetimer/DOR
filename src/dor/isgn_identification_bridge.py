from __future__ import annotations

"""Non-invasive DOR adapter over ISGN 2.3 PHM differential APIs.

This module does not modify ISGN Core.  It converts a COMPLETE mechanism
baseline plus normal/authorized-abnormal PublicDataset artifacts into the DOR
v0.7 `DOR-ISGN-IDENTIFICATION-EVIDENCE/0.7` contract.

A parameter-change claim is admitted only when:
  * baseline certificate is COMPLETE,
  * abnormal-window evidence validity is VALID,
  * the ISGN anchored drift is finite/identifiable,
  * its median displacement exceeds the empirical normal absolute envelope,
  * the mechanism slot maps to exactly one component with provenance.
"""

from pathlib import Path
from typing import Any
import hashlib
import json
import math
import numpy as np

from .identification import ISGN_EVIDENCE_CONTRACT
from .ontology import validate_mechanism_component_map

MECHANISM_BASELINE_CONTRACT = "DOR-ISGN-MECHANISM-BASELINE/0.7"


def _sha(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def validate_mechanism_baseline(spec: dict[str, Any]) -> list[str]:
    errors=[]
    if spec.get('contract') != MECHANISM_BASELINE_CONTRACT:
        errors.append(f"contract must be {MECHANISM_BASELINE_CONTRACT}")
    if not isinstance(spec.get('graph'),dict) or not spec.get('graph'):
        errors.append('graph must be a non-empty mapping')
    if not str((spec.get('provenance') or {}).get('source','')).strip():
        errors.append('provenance.source is required')
    return errors


def _map_mechanism(mechanism_map: dict[str, Any], mechanism_id: str) -> tuple[list[str], list[str]]:
    comps=[]; prov=[]
    for b in mechanism_map.get('bindings',[]):
        if str(b.get('mechanism_id')) != mechanism_id: continue
        comps += [str(x) for x in b.get('component_ids',[]) if str(x)]
        if str(b.get('provenance','')).strip(): prov.append(str(b['provenance']))
    return list(dict.fromkeys(comps)), list(dict.fromkeys(prov))


def build_isgn_identification_evidence(
    mechanism_spec: dict[str, Any], normal_ds: Any, abnormal_ds: Any,
    ontology: dict[str, Any], mechanism_map: dict[str, Any], *,
    window_evidence_validity: str = 'VALID', normal_source_sha256: str | None = None,
    abnormal_source_sha256: str | None = None,
) -> dict[str, Any]:
    errors=validate_mechanism_baseline(mechanism_spec)
    mv=validate_mechanism_component_map(mechanism_map,ontology)
    if mv.state!='PASS': errors += list(mv.errors)
    cert=str(mechanism_spec.get('certificate_status','')).upper()
    validity=str(window_evidence_validity).upper()
    base={
      'contract':ISGN_EVIDENCE_CONTRACT,
      'baseline_certificate_status':cert,
      'window_evidence_validity':validity,
      'changes':[],
      'provenance':{
        'mechanism_source':(mechanism_spec.get('provenance') or {}).get('source'),
        'normal_sha256':normal_source_sha256,
        'abnormal_sha256':abnormal_source_sha256,
        'isgn_api':'ISGN 2.3 assess_observation + fit_normal_phm + calibrate_phm_health + anchored_parameter_drift'
      }
    }
    if errors:
        base.update(status='UNSUPPORTED',errors=errors); return base
    if cert!='COMPLETE':
        base.update(status='UNSUPPORTED',errors=['baseline certificate is not COMPLETE']); return base
    if validity!='VALID':
        base.update(status='UNSUPPORTED',errors=['abnormal window evidence validity is not VALID']); return base

    # Lazy import keeps DOR's discrete path independent of ISGN availability.
    from isgn import assess_observation, fit_normal_phm, anchored_parameter_drift, calibrate_phm_health
    normal_obs = assess_observation(normal_ds)
    abnormal_obs = assess_observation(abnormal_ds)
    base['observation_gate'] = {
      'normal': normal_obs.to_dict(),
      'abnormal': abnormal_obs.to_dict(),
      'policy': 'FAIL_CLOSED_ON_OBSERVATION_INSUFFICIENT',
    }
    blocked = [name for name, rep in [('normal', normal_obs), ('abnormal', abnormal_obs)] if str(rep.status).upper() == 'OBSERVATION_INSUFFICIENT']
    if blocked:
        base.update(
          status='UNIDENTIFIABLE',
          errors=[f"ISGN 2.3 observation gate reports OBSERVATION_INSUFFICIENT for {', '.join(blocked)} data; DOR blocks physical identification rather than treating history rescue/timebase failure as mechanism truth"],
        )
        return base
    graph={str(k):[str(x) for x in v] for k,v in mechanism_spec['graph'].items()}
    nominal=fit_normal_phm(normal_ds,graph,operators=mechanism_spec.get('operators') or {},operator_parameters=mechanism_spec.get('operator_parameters') or {})
    cal=calibrate_phm_health(normal_ds,nominal)
    drift,names=anchored_parameter_drift(abnormal_ds,nominal)
    changes=[]; any_ambiguous_map=False; any_unidentifiable=False
    for j,name in enumerate(names):
        # adtheta:<target><-<source>
        slot=name.split(':',1)[1] if ':' in name else name
        if '<-' not in slot: continue
        target,source=slot.split('<-',1); mechanism_id=f'{target}<-{source}'
        vals=np.asarray(drift[:,j],dtype=float); finite=np.isfinite(vals)
        identifiable=bool(np.all(finite) and int(finite.sum())>=1 and name in cal.eligible_features)
        center=float(cal.center.get(name,0.0)); floor=float(cal.normal_abs_floor.get(name,float('inf')))
        median=float(np.median(vals[finite])) if np.any(finite) else float('nan')
        displacement=abs(median-center) if math.isfinite(median) else float('nan')
        raw_supported=bool(identifiable and math.isfinite(displacement) and displacement>floor)
        comps,provs=_map_mechanism(mechanism_map,mechanism_id)
        unique_mapping=len(comps)==1 and bool(provs)
        # Mapping ambiguity matters only when ISGN has actually found a supported
        # physical change.  An unrelated inactive/unidentifiable mechanism slot
        # must not poison a separately supported identification.
        if raw_supported and not unique_mapping:
            any_ambiguous_map=True
        if not identifiable:
            any_unidentifiable=True
        supported=bool(raw_supported and unique_mapping)
        baseline=float(nominal.coefficients[target][source])
        rel=median/max(abs(baseline),1e-12) if math.isfinite(median) else float('nan')
        if comps:
            for comp in comps:
                changes.append({
                  'component_id':comp,'mechanism_id':mechanism_id,'change_type':'PARAMETER_DRIFT',
                  'parameter_name':f'{target}<-{source}','baseline_value':baseline,'delta':median,'relative_delta':rel,
                  'identifiable':identifiable and unique_mapping,'supported':supported,
                  'normal_center':center,'normal_abs_floor':floor,'abnormal_displacement_from_normal_center':displacement,
                  'evidence_id':f'isgn:{name}','provenance':' | '.join(provs) if provs else 'UNPROVENANCED_MECHANISM_COMPONENT_BINDING'
                })
        else:
            changes.append({
              'component_id':'','mechanism_id':mechanism_id,'change_type':'PARAMETER_DRIFT','parameter_name':f'{target}<-{source}',
              'baseline_value':baseline,'delta':median,'relative_delta':rel,'identifiable':False,'supported':False,
              'normal_center':center,'normal_abs_floor':floor,'abnormal_displacement_from_normal_center':displacement,
              'evidence_id':f'isgn:{name}','provenance':'UNMAPPED_MECHANISM_SLOT'
            })
    base['changes']=changes
    supported_changes=[x for x in changes if x.get('supported')]
    if any_ambiguous_map:
        status='ALTERNATIVES'
    elif supported_changes:
        status='SUPPORTED'
    elif any_unidentifiable:
        status='UNIDENTIFIABLE'
    else:
        status='UNSUPPORTED'
    base['status']=status
    base['calibration']={'trajectory_count':cal.trajectory_count,'eligible_features':list(cal.eligible_features),'blocked_features':dict(cal.blocked_features)}
    return base


def build_isgn_identification_evidence_from_files(
    mechanism_spec_path: str | Path, normal_path: str | Path, abnormal_path: str | Path,
    ontology_path: str | Path, mechanism_map_path: str | Path, *, window_evidence_validity: str='VALID'
) -> dict[str, Any]:
    from isgn import load_npz
    spec=json.loads(Path(mechanism_spec_path).read_text()); ont=json.loads(Path(ontology_path).read_text()); mmap=json.loads(Path(mechanism_map_path).read_text())
    normal=load_npz(normal_path); abnormal=load_npz(abnormal_path)
    return build_isgn_identification_evidence(spec,normal,abnormal,ont,mmap,window_evidence_validity=window_evidence_validity,normal_source_sha256=_sha(normal_path),abnormal_source_sha256=_sha(abnormal_path))
