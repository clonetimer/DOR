from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

from .api import RecoveryResult, recover_graph, certify_completion
from .backends import recommend_backend, backend_manifest
from .symbolic import export_explicit_relation
from .contracts import PublicDataset
from .observation_gate import assess_observation, ObservationGateReport

@dataclass
class FusionOutcome:
    recovery: Optional[RecoveryResult]
    certificate: Dict[str,Any]
    backend_decision: Dict[str,str]
    next_action: str
    symbolic_task: Optional[Dict[str,Any]] = None
    observation: Optional[Dict[str,Any]] = None
    def report(self) -> Dict[str,Any]:
        return {
            'backend_decision':self.backend_decision,
            'recovery':self.recovery.summary.to_dict() if self.recovery is not None else None,
            'certificate':self.certificate,
            'next_action':self.next_action,
            'symbolic_task':self.symbolic_task,
            'observation':self.observation,
        }

def _next(status:str) -> str:
    return {
        'COMPLETE':'READY_FOR_PHM',
        'PRIOR_CONFLICT':'REVIEW_DECLARED_MECHANISM_PRIOR',
        'UNIDENTIFIABLE':'IMPROVE_EXCITATION_OR_OBSERVATION',
        'GRAMMAR_UNCOVERED':'DISCOVER_LOCAL_RELATION_OR_EXTEND_GRAMMAR',
        'UNCERTIFIED':'CALIBRATE_NUMERICAL_OR_NOISE_COVERAGE_FLOOR',
    }.get(status,'REVIEW_EVIDENCE')

def build_operational_model(train:PublicDataset,val:PublicDataset,*,
                            backend:str='auto',seed:int=101,device:str='cpu',
                            epochs:int=1000,refit_epochs:int=300,
                            constitutive_search:bool=False,
                            known_present:Optional[Dict[str,Iterable[str]]]=None,
                            known_absent:Optional[Dict[str,Iterable[str]]]=None,
                            declared_present:Optional[Dict[str,list[str]]]=None,
                            certification_data:Optional[PublicDataset]=None,
                            coverage_floor:Optional[float]=None,
                            coverage_threshold:float=8.0,
                            backend_options:Optional[Dict[str,Any]]=None,
                            symbolic_gap:Optional[Dict[str,Any]]=None,
                            enforce_observation_gate:bool=True) -> FusionOutcome:
    obs=assess_observation(train) if enforce_observation_gate else None
    if obs is not None and not obs.can_run_discovery:
        cert={'status':'OBSERVATION_INSUFFICIENT','observation':obs.to_dict()}
        return FusionOutcome(None,cert,{'backend':'none','reason':'observation gate blocked mechanism discovery'},
                             'IMPROVE_OBSERVATION_BEFORE_DISCOVERY',None,obs.to_dict())
    decision = recommend_backend(constitutive_search=constitutive_search,
                                 has_partial_prior=bool(known_present or known_absent)) if backend=='auto' else {'backend':backend,'reason':'explicit user selection'}
    chosen=decision['backend']
    r=recover_graph(train,val,seed=seed,device=device,epochs=epochs,refit_epochs=refit_epochs,
                    constitutive_search=constitutive_search,known_present=known_present,known_absent=known_absent,
                    backend=chosen,backend_options=backend_options)
    cert_data=certification_data or val
    cert=certify_completion(r,cert_data,declared_present=declared_present or (known_present or {}),
                            coverage_floor=coverage_floor,coverage_threshold=coverage_threshold,device=device)
    action=_next(cert['status'])
    task=None
    if cert['status']=='GRAMMAR_UNCOVERED' and symbolic_gap:
        inputs=symbolic_gap.get('inputs'); target=symbolic_gap.get('target'); out=symbolic_gap.get('out_csv')
        if inputs and target and out:
            task=export_explicit_relation(cert_data,inputs=inputs,target=target,out_csv=out,
                                          max_samples=int(symbolic_gap.get('max_samples',10000)),seed=int(symbolic_gap.get('seed',0))).to_dict()
            action='RUN_SYMBOLIC_RELATION_DISCOVERY'
    if obs is not None and obs.status=='CONDITIONALLY_QUALIFIED' and action=='READY_FOR_PHM':
        action='READY_FOR_PHM_WITH_OBSERVATION_CAUTION'
    return FusionOutcome(r,cert,decision,action,task,obs.to_dict() if obs is not None else None)
