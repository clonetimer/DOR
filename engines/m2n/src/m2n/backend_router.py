from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping
import json
import numpy as np

from .discovery import DynamicsDiscoveryRequest, PySINDyCandidateBackend, BackendUnavailableError
from .symmatika_adapter import SymMatikaExplicitRequest, prepare_symmatika_job, run_symmatika_from_source
from .model_set_inference import ModelComponentSlot, SBMIModelSetRequest, assess_sbmi_feasibility, export_sbmi_contract


@dataclass(frozen=True)
class BackendAction:
    backend: str
    status: str
    slot_id: str | None
    reason: str
    artifact: str | None = None
    def to_dict(self): return asdict(self)


def _col(data: Mapping[str,np.ndarray], name: str) -> np.ndarray:
    if name not in data: raise KeyError(f'backend request missing data column {name!r}')
    return np.asarray(data[name],float)


def run_backend_requests(spec: Mapping[str,Any], data: Mapping[str,np.ndarray], train_idx: np.ndarray, output_dir: str|Path, *, qualification_status: str) -> tuple[BackendAction,...]:
    """Run only explicitly requested backends after M2N qualification.

    Backend results never mutate the architecture plan automatically. Discovery
    outputs remain UNADMITTED proposals; SBMI produces a model-set inference
    contract rather than replacing M2N's finite model semantics.
    """
    requests=spec.get('backends',{}).get('requests',[])
    if not requests: return ()
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    actions=[]
    for i,r in enumerate(requests):
        backend=str(r['backend']).lower(); slot=r.get('slot_id')
        if qualification_status!='READY':
            actions.append(BackendAction(backend,'BLOCKED_BY_QUALIFICATION',slot,f'M2N qualification is {qualification_status}; external backends may not bypass it'))
            continue
        try:
            if backend=='pysindy':
                features=tuple(map(str,r['feature_columns']))
                X=np.column_stack([_col(data,c)[train_idx] for c in features])
                tcol=r.get('time_column')
                t=float(r.get('dt',1.0)) if tcol is None else _col(data,str(tcol))[train_idx]
                xdot=None
                if r.get('derivative_columns'):
                    xdot=np.column_stack([_col(data,c)[train_idx] for c in r['derivative_columns']])
                req=DynamicsDiscoveryRequest(X,t,features,x_dot=xdot,polynomial_degree=int(r.get('polynomial_degree',3)),threshold=float(r.get('threshold',0.05)),alpha=float(r.get('alpha',1e-6)))
                batch=PySINDyCandidateBackend.propose(req)
                p=out/f'pysindy_{i}_proposals.json'; p.write_text(json.dumps(batch.as_dict(),indent=2,ensure_ascii=False),encoding='utf-8')
                actions.append(BackendAction('pysindy','PROPOSALS_PENDING_ADMISSION',slot,'PySINDy ran successfully; proposals must pass M2N physical/units/support/IR admission before joining K2',str(p)))
            elif backend=='symmatika':
                fcols=tuple(map(str,r['feature_columns'])); target=str(r['target_column'])
                X=np.column_stack([_col(data,c)[train_idx] for c in fcols]); y=_col(data,target)[train_idx]
                req=SymMatikaExplicitRequest(X,y,fcols,str(r.get('target_name',target)),population_size=int(r.get('population_size',10000)),max_depth=int(r.get('max_depth',3)))
                source=r.get('source_root')
                if source:
                    batch=run_symmatika_from_source(req,source,out/f'symmatika_{i}')
                    p=out/f'symmatika_{i}_proposals.json'; p.write_text(json.dumps(batch.as_dict(),indent=2,ensure_ascii=False),encoding='utf-8')
                    actions.append(BackendAction('symmatika','PROPOSALS_PENDING_ADMISSION',slot,'SymMatika ran; proposals remain unadmitted',str(p)))
                else:
                    job=prepare_symmatika_job(req,out/f'symmatika_{i}_job')
                    actions.append(BackendAction('symmatika','EXTERNAL_JOB_PREPARED',slot,'No source_root supplied; M2N prepared a reproducible external SymMatika job',job.contract_json))
            else:
                actions.append(BackendAction(backend,'UNSUPPORTED_BACKEND_REQUEST',slot,'No automatic runtime route is defined for this backend in the goal-driven frontend'))
        except BackendUnavailableError as e:
            actions.append(BackendAction(backend,'BACKEND_UNAVAILABLE',slot,str(e)))
    return tuple(actions)


def assess_model_set_routing(slot_families: Mapping[str,tuple[str,...]], output_dir: str|Path, *, exact_enumeration_limit: int=32) -> BackendAction | None:
    varying=[ModelComponentSlot(k,tuple(v)) for k,v in slot_families.items() if len(v)>1]
    if not varying: return None
    req=SBMIModelSetRequest(tuple(varying),exact_enumeration_limit=exact_enumeration_limit)
    a=assess_sbmi_feasibility(req)
    if a.status=='NOT_NEEDED':
        return BackendAction('sbmi','NOT_NEEDED',None,a.reason)
    p=Path(output_dir)/'sbmi_model_set_contract.json'; export_sbmi_contract(req,p)
    return BackendAction('sbmi','MODEL_SET_BACKEND_RECOMMENDED',None,a.reason,str(p))
