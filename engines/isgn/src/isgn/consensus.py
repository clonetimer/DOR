from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Any, Tuple
import math

from .api import RecoveryResult, recover_graph, certify_completion
from .contracts import PublicDataset
from .backends.base import CanonicalMechanismModel, rollout_nrmse_canonical
from .api import RecoverySummary
from .observation_gate import assess_observation


def _edges(graph: Dict[str, List[str]]) -> set[tuple[str, str]]:
    return {(s, t) for t, srcs in graph.items() for s in srcs}


def graph_jaccard(a: Dict[str, List[str]], b: Dict[str, List[str]]) -> float:
    ea, eb = _edges(a), _edges(b)
    if not ea and not eb:
        return 1.0
    return len(ea & eb) / max(len(ea | eb), 1)


def graph_exact(a: Dict[str, List[str]], b: Dict[str, List[str]]) -> bool:
    return _edges(a) == _edges(b)


def operator_agreement(a, b) -> float:
    shared=[]
    ea=_edges(a.summary.graph); eb=_edges(b.summary.graph)
    for s,t in sorted(ea & eb):
        oa=a.summary.operators.get(t,{}).get(s,'Identity')
        ob=b.summary.operators.get(t,{}).get(s,'Identity')
        shared.append(oa==ob)
    return 1.0 if not shared else sum(shared)/len(shared)


def coefficient_sign_agreement(a, b, *, floor: float=1e-9) -> float:
    vals=[]
    for s,t in sorted(_edges(a.summary.graph) & _edges(b.summary.graph)):
        ca=float(a.summary.coefficients.get(t,{}).get(s,0.0))
        cb=float(b.summary.coefficients.get(t,{}).get(s,0.0))
        if abs(ca)<=floor or abs(cb)<=floor:
            continue
        vals.append(math.copysign(1,ca)==math.copysign(1,cb))
    return 1.0 if not vals else sum(vals)/len(vals)


def operator_parameter_max_rel_diff(a, b, *, floor: float=1e-8) -> float:
    diffs=[]
    for src,t in sorted(_edges(a.summary.graph) & _edges(b.summary.graph)):
        oa=a.summary.operators.get(t,{}).get(src,'Identity')
        ob=b.summary.operators.get(t,{}).get(src,'Identity')
        if oa!=ob or oa=='Identity':
            continue
        pa=a.summary.operator_parameters.get(t,{}).get(src)
        pb=b.summary.operator_parameters.get(t,{}).get(src)
        if pa is None or pb is None:
            continue
        if isinstance(pa,dict) or isinstance(pb,dict):
            if not isinstance(pa,dict) or not isinstance(pb,dict) or set(pa)!=set(pb):
                diffs.append(float('inf')); continue
            for key in sorted(pa):
                try:
                    xa=float(pa[key]); xb=float(pb[key])
                except (TypeError,ValueError):
                    if pa[key] != pb[key]: diffs.append(float('inf'))
                    continue
                diffs.append(abs(xa-xb)/max(abs(xa),abs(xb),floor))
        else:
            pa=float(pa); pb=float(pb)
            diffs.append(abs(pa-pb)/max(abs(pa),abs(pb),floor))
    return 0.0 if not diffs else max(diffs)



def _transform(z, op, param):
    import numpy as np
    if op == 'Identity': return z
    if op == 'Saturation': return np.clip(z,-float(param),float(param))
    if op == 'DeadZone':
        c=float(param); return np.sign(z)*np.maximum(np.abs(z)-c,0.0)
    if op == 'SmoothStribeck':
        from .friction import smooth_stribeck_transform
        if not isinstance(param,dict): raise ValueError('SmoothStribeck parameter mapping missing')
        return smooth_stribeck_transform(z,param)
    raise ValueError(f'consensus canonical refit does not support operator {op!r}')


def _fit_coefficients_for_template(train: PublicDataset, graph, operators, params, *, ridge: float=1e-10):
    import numpy as np
    states=[v['name'] for v in train.schema['states']]
    inputs=[v['name'] for v in train.schema['inputs']]
    xm=.5*(train.X[:,:-1].astype(float)+train.X[:,1:].astype(float))
    y=((train.X[:,1:].astype(float)-train.X[:,:-1].astype(float))/train.dt)
    coeff={}
    for ti,t in enumerate(states):
        srcs=list(graph.get(t,[])); cols=[]
        for src in srcs:
            if src in states: z=xm[:,:,states.index(src)]
            elif src in inputs: z=train.U[:,:,inputs.index(src)].astype(float)
            elif src=='__constant__': z=np.ones(xm.shape[:2],float)
            else: raise KeyError(src)
            op=operators.get(t,{}).get(src,'Identity'); par=params.get(t,{}).get(src)
            cols.append(_transform(z,op,par).reshape(-1))
        if cols:
            A=np.stack(cols,axis=1); b=y[:,:,ti].reshape(-1)
            scale=np.sqrt(np.mean(A*A,axis=0)); scale=np.maximum(scale,1e-12)
            An=A/scale
            cn=np.linalg.solve(An.T@An+ridge*np.eye(An.shape[1]),An.T@b)
            c=cn/scale
            coeff[t]={src:float(v) for src,v in zip(srcs,c)}
        else: coeff[t]={}
    return coeff


def _parameter_grid(train: PublicDataset, src: str, op: str, current):
    import numpy as np
    states=[v['name'] for v in train.schema['states']]; inputs=[v['name'] for v in train.schema['inputs']]
    if src in inputs: z=np.asarray(train.U[:,:,inputs.index(src)],float)
    elif src in states: z=np.asarray(train.X[:,:-1,states.index(src)],float)
    else: return [current]
    q=max(float(np.quantile(np.abs(z),.995)),1e-8)
    if op=='Saturation': grid=np.linspace(.25*q,.98*q,25)
    elif op=='DeadZone': grid=np.linspace(.02*q,.70*q,25)
    else: return [current]
    vals=[float(x) for x in grid]
    if current is not None: vals.append(float(current))
    return sorted(set(vals))


def refit_consensus_canonical(train: PublicDataset, val: PublicDataset, template: RecoveryResult, *, ridge: float=1e-10) -> RecoveryResult:
    """Common solver-independent refit for a consensus mechanism structure.

    It profiles parameters of finite constitutive operators (Saturation/DeadZone)
    on validation rollout, then refits all linear equation coefficients on train.
    This prevents backend-specific continuous optimization quality from leaking
    into the final mechanism certificate.
    """
    graph={k:list(v) for k,v in template.summary.graph.items()}
    operators={t:{s:template.summary.operators.get(t,{}).get(s,'Identity') for s in srcs} for t,srcs in graph.items()}
    params={t:{s:template.summary.operator_parameters.get(t,{}).get(s) for s in srcs} for t,srcs in graph.items()}
    # Coordinate profile each parametric edge using canonical validation rollout.
    for _ in range(2):
        improved=False
        for t,srcs in graph.items():
            for src in srcs:
                op=operators[t][src]
                if op not in ('Saturation','DeadZone'): continue
                best=None
                for par in _parameter_grid(train,src,op,params[t][src]):
                    trial={tt:dict(row) for tt,row in params.items()}; trial[t][src]=par
                    coeff=_fit_coefficients_for_template(train,graph,operators,trial,ridge=ridge)
                    model=CanonicalMechanismModel(schema=train.schema,dt=train.dt,coefficients=coeff,operators=operators,operator_parameters=trial)
                    err=rollout_nrmse_canonical(model,val)
                    key=(err,par)
                    if best is None or key<best[0]: best=(key,par)
                if best is not None and params[t][src] != best[1]:
                    params[t][src]=best[1]; improved=True
        if not improved: break
    coeff=_fit_coefficients_for_template(train,graph,operators,params,ridge=ridge)
    model=CanonicalMechanismModel(schema=train.schema,dt=train.dt,coefficients=coeff,operators=operators,operator_parameters=params)
    err=rollout_nrmse_canonical(model,val)
    summary=RecoverySummary(graph=graph,operators=operators,coefficients=coeff,val_nrmse=float(err),seed=int(template.summary.seed),
                            constitutive_search=template.summary.constitutive_search,backend='consensus-refit',operator_parameters=params,
                            backend_metadata={'solver':'solver-independent-profile+ridge-refit','source_backends':'structural-consensus'})
    return RecoveryResult(model=model,summary=summary,schema=train.schema,dt=float(train.dt))


@dataclass
class BackendCandidate:
    backend: str
    result: RecoveryResult
    certificate: Dict[str, Any]

@dataclass
class ConsensusReport:
    status: str
    primary_backend: str
    backends: List[str]
    graph_jaccard: float
    graph_exact: bool
    operator_agreement: float
    coefficient_sign_agreement: float
    operator_parameter_max_rel_diff: float
    validation_errors: Dict[str, float]
    certificate_statuses: Dict[str, str]
    disagreement_edges: Dict[str, List[str]]
    reason: str

    def to_dict(self): return asdict(self)

@dataclass
class ConsensusOutcome:
    primary: Optional[RecoveryResult]
    candidates: Dict[str, RecoveryResult]
    certificates: Dict[str, Dict[str, Any]]
    consensus: ConsensusReport
    consensus_certificate: Dict[str, Any]
    next_action: str
    observation: Optional[Dict[str,Any]] = None

    def report(self):
        return {
            'consensus': self.consensus.to_dict(),
            'primary': self.primary.summary.to_dict() if self.primary is not None else None,
            'candidate_summaries': {k:v.summary.to_dict() for k,v in self.candidates.items()},
            'certificates': self.certificates,
            'consensus_certificate': self.consensus_certificate,
            'next_action': self.next_action,
            'observation': self.observation,
        }


def _edge_diff(a:RecoveryResult,b:RecoveryResult):
    ea,eb=_edges(a.summary.graph),_edges(b.summary.graph)
    return {
        f'only_{a.summary.backend}':[f'{s}->{t}' for s,t in sorted(ea-eb)],
        f'only_{b.summary.backend}':[f'{s}->{t}' for s,t in sorted(eb-ea)],
    }


def _default_backends() -> List[str]:
    # Native is always retained as the engineering-system grammar reference.
    # Prefer official PySINDy when installed; reference-sindy is explicit offline QA only.
    try:
        from .backends import backend_status
        st=backend_status()
    except Exception:
        st={}
    return ['native','pysindy'] if st.get('pysindy') else ['native','reference-sindy']


def build_consensus_model(train: PublicDataset, val: PublicDataset, *,
                          backends: Optional[Sequence[str]]=None,
                          seed: int=101, device: str='cpu', epochs: int=1000, refit_epochs: int=300,
                          constitutive_search: bool=False,
                          known_present: Optional[Dict[str,Iterable[str]]]=None,
                          known_absent: Optional[Dict[str,Iterable[str]]]=None,
                          declared_present: Optional[Dict[str,List[str]]]=None,
                          certification_data: Optional[PublicDataset]=None,
                          coverage_floor: Optional[float]=None,
                          coverage_threshold: float=8.0,
                          backend_options: Optional[Dict[str,Dict[str,Any]]]=None,
                          min_graph_jaccard: float=0.8,
                          min_operator_agreement: float=0.8,
                          enforce_observation_gate: bool=True) -> ConsensusOutcome:
    obs=assess_observation(train) if enforce_observation_gate else None
    names=list(backends or _default_backends())
    if obs is not None and not obs.can_run_discovery:
        rep=ConsensusReport('OBSERVATION_INSUFFICIENT','none',names,0.0,False,0.0,0.0,0.0,{}, {}, {},
                            'observation gate blocked all mechanism-discovery backends before search')
        return ConsensusOutcome(None,{}, {},rep,{'status':'OBSERVATION_INSUFFICIENT','observation':obs.to_dict()},
                                'IMPROVE_OBSERVATION_BEFORE_DISCOVERY',obs.to_dict())
    if len(names)<2:
        raise ValueError('consensus requires at least two discovery backends')
    candidates={}; certs={}; cert_data=certification_data or val
    opts=backend_options or {}
    for name in names:
        r=recover_graph(train,val,seed=seed,device=device,epochs=epochs,refit_epochs=refit_epochs,
                        constitutive_search=constitutive_search,known_present=known_present,
                        known_absent=known_absent,backend=name,backend_options=opts.get(name))
        candidates[name]=r
        certs[name]=certify_completion(r,cert_data,declared_present=declared_present or (known_present or {}),
                                       coverage_floor=coverage_floor,coverage_threshold=coverage_threshold,device=device)
    # Pairwise conservative aggregation. For >2 backends take worst agreement.
    pairs=[]
    for i in range(len(names)):
        for j in range(i+1,len(names)):
            a,b=candidates[names[i]],candidates[names[j]]
            pairs.append((graph_jaccard(a.summary.graph,b.summary.graph),graph_exact(a.summary.graph,b.summary.graph),
                          operator_agreement(a,b),coefficient_sign_agreement(a,b),operator_parameter_max_rel_diff(a,b),a,b))
    min_j=min(p[0] for p in pairs); exact=all(p[1] for p in pairs); min_op=min(p[2] for p in pairs); min_sign=min(p[3] for p in pairs); max_param_diff=max(p[4] for p in pairs)
    # Backend-specific coefficient optimizers are diagnostic only.  When structure agrees,
    # perform one common canonical refit before final certification.
    candidate_primary_name=min(names,key=lambda n: float(candidates[n].summary.val_nrmse))
    candidate_primary=candidates[candidate_primary_name]
    cert_statuses={n:certs[n]['status'] for n in names}
    structurally_consistent=min_j>=min_graph_jaccard and min_op>=min_operator_agreement
    consensus_cert={}
    if not structurally_consistent:
        primary_name=candidate_primary_name; primary=candidate_primary
        status='BACKEND_DISAGREEMENT'; action='REVIEW_BACKEND_DISAGREEMENT'
        reason='mechanism graphs/operators disagree beyond configured tolerance; do not auto-certify a unique mechanism model'
    else:
        primary=refit_consensus_canonical(train,val,candidate_primary)
        primary_name='consensus-refit'
        consensus_cert=certify_completion(primary,cert_data,declared_present=declared_present or (known_present or {}),
                                           coverage_floor=coverage_floor,coverage_threshold=coverage_threshold,device=device)
        status='CONSENSUS'
        action={
            'COMPLETE':'READY_FOR_PHM',
            'PRIOR_CONFLICT':'REVIEW_DECLARED_MECHANISM_PRIOR',
            'UNIDENTIFIABLE':'IMPROVE_EXCITATION_OR_OBSERVATION',
            'GRAMMAR_UNCOVERED':'DISCOVER_LOCAL_RELATION_OR_EXTEND_GRAMMAR',
            'UNCERTIFIED':'CALIBRATE_NUMERICAL_OR_NOISE_COVERAGE_FLOOR',
        }.get(consensus_cert.get('status'),'REVIEW_EVIDENCE')
        if len(set(cert_statuses.values()))>1:
            reason='backends agree structurally; backend-specific certificates differ, so final certificate uses solver-independent consensus refit'
        else:
            reason='independent discovery backends agree structurally; final certificate uses solver-independent consensus refit'
    # Aggregate pairwise edge diffs for audit.
    diffs={}
    for _,_,_,_,_,a,b in pairs:
        diffs.update(_edge_diff(a,b))
    rep=ConsensusReport(status,primary_name,names,float(min_j),bool(exact),float(min_op),float(min_sign),float(max_param_diff),
                        {n:float(candidates[n].summary.val_nrmse) for n in names},cert_statuses,diffs,reason)
    if obs is not None and obs.status=='CONDITIONALLY_QUALIFIED' and action=='READY_FOR_PHM':
        action='READY_FOR_PHM_WITH_OBSERVATION_CAUTION'
    return ConsensusOutcome(primary,candidates,certs,rep,consensus_cert,action,obs.to_dict() if obs is not None else None)
