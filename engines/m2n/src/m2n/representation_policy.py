from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Mapping


class RepresentationKind(str, Enum):
    LOCAL_FUNCTION = 'LOCAL_FUNCTION'
    DIFFERENTIAL_DYNAMICS = 'DIFFERENTIAL_DYNAMICS'
    CONSERVATIVE_HAMILTONIAN = 'CONSERVATIVE_HAMILTONIAN'
    FUNCTION_TO_FUNCTION_OPERATOR = 'FUNCTION_TO_FUNCTION_OPERATOR'
    REGULAR_GRID_FIELD_OPERATOR = 'REGULAR_GRID_FIELD_OPERATOR'
    SPATIOTEMPORAL_LATENT_DYNAMICS = 'SPATIOTEMPORAL_LATENT_DYNAMICS'


@dataclass(frozen=True)
class RepresentationEligibility:
    kind: str
    status: str  # ELIGIBLE | CONDITIONALLY_ELIGIBLE | INELIGIBLE
    architecture_pattern: str
    reason: str
    required_conditions: tuple[str,...]=()
    runtime_status: str='CONTRACT_ONLY'

    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class RepresentationPlan:
    requested_kind: str
    eligibility: RepresentationEligibility
    slot_id: str | None = None

    def to_dict(self):
        d={'requested_kind':self.requested_kind,'slot_id':self.slot_id,'eligibility':self.eligibility.to_dict()}
        return d


def _b(m: Mapping[str,Any], key: str, default=False) -> bool:
    return bool(m.get(key,default))


def assess_representation(kind: str|RepresentationKind, semantics: Mapping[str,Any] | None=None) -> RepresentationEligibility:
    k=RepresentationKind(str(kind).upper()) if not isinstance(kind,RepresentationKind) else kind
    s=dict(semantics or {})
    if k is RepresentationKind.LOCAL_FUNCTION:
        return RepresentationEligibility(k.value,'ELIGIBLE','M2N_LOCAL_CONSTRAINED_NN','unknown object is a local finite-dimensional constitutive/function relation',('local inputs observed or reconstructable','output semantics/dimensions declared'),runtime_status='NATIVE')

    if k is RepresentationKind.DIFFERENTIAL_DYNAMICS:
        cond=('state/time semantics declared','differential solver integration is part of the estimation contract','unknown term placement is localized')
        ok=_b(s,'continuous_time',True) and _b(s,'solver_compatible',False)
        return RepresentationEligibility(k.value,'ELIGIBLE' if ok else 'CONDITIONALLY_ELIGIBLE','UDE_STYLE','unknown object is a differential-equation term; use solver-aware hybrid dynamics rather than a pointwise MLP' if ok else 'differential dynamics are requested but solver compatibility has not been qualified',cond,runtime_status='NATIVE_FIXED_STEP_RK4' if ok else 'CONTRACT_ONLY')

    if k is RepresentationKind.CONSERVATIVE_HAMILTONIAN:
        cond=('canonical or reconstructable q/p coordinates','energy-like conserved quantity is physically plausible','dominant dissipation/forcing absent or modeled separately')
        qp=_b(s,'canonical_qp',False) or _b(s,'qp_reconstructable',False)
        conservative=_b(s,'conservative_core',False)
        diss_sep=_b(s,'dissipation_absent',False) or _b(s,'dissipation_separate',False)
        ok=qp and conservative and diss_sep
        return RepresentationEligibility(k.value,'ELIGIBLE' if ok else 'INELIGIBLE','HNN_STYLE','Hamiltonian inductive bias is justified by the declared conservative core' if ok else 'HNN is blocked because canonical observability/conservative-core/dissipation-separation conditions are not all satisfied',cond,runtime_status='CONTRACT_ONLY')

    if k is RepresentationKind.FUNCTION_TO_FUNCTION_OPERATOR:
        cond=('multiple input functions/trajectories available','input function sensors are consistent or explicitly encoded','output query coordinate is defined')
        ok=_b(s,'multiple_input_functions',False) and _b(s,'consistent_input_sensors',False) and _b(s,'output_query_coordinate',False)
        return RepresentationEligibility(k.value,'ELIGIBLE' if ok else 'CONDITIONALLY_ELIGIBLE','DEEPONET_STYLE','operator-valued task satisfies branch/trunk data semantics' if ok else 'operator task is plausible but branch/trunk observation contract is incomplete',cond,runtime_status='CONTRACT_ONLY')

    if k is RepresentationKind.REGULAR_GRID_FIELD_OPERATOR:
        cond=('function-to-function/PDE-field task','regular uniform spatial grid for FFT implementation','many parameter/initial-condition instances rather than one PDE instance')
        ok=_b(s,'field_operator',False) and _b(s,'regular_grid',False) and _b(s,'multiple_instances',False)
        return RepresentationEligibility(k.value,'ELIGIBLE' if ok else 'INELIGIBLE','FNO_STYLE','regular-grid PDE/field operator semantics justify Fourier-operator bias' if ok else 'FNO is blocked unless the task is a repeated field-operator problem on a qualified regular grid',cond,runtime_status='CONTRACT_ONLY')

    if k is RepresentationKind.SPATIOTEMPORAL_LATENT_DYNAMICS:
        cond=('image/field sequence input','physical law is not directly valid in raw observation space','a latent physical dynamics hypothesis is credible','residual visual/detail information must remain separately learnable')
        ok=_b(s,'spatiotemporal_sequence',False) and _b(s,'physics_not_direct_in_observation_space',False) and _b(s,'latent_physics_plausible',False)
        return RepresentationEligibility(k.value,'ELIGIBLE' if ok else 'CONDITIONALLY_ELIGIBLE','PHYDNET_STYLE','latent physical/residual disentanglement is structurally appropriate' if ok else 'spatiotemporal latent physics is not yet sufficiently qualified',cond,runtime_status='CONTRACT_ONLY')
    raise AssertionError(k)


def _slot_requests(spec: Mapping[str,Any]) -> list[tuple[str|None,str,Mapping[str,Any]]]:
    requests=[]
    task=dict(spec.get('task_semantics',{}))
    if task.get('representation_kind'):
        requests.append((None,str(task['representation_kind']),task))
    if spec.get('template')=='second_order_mechanics':
        k=spec.get('knowledge',{}).get('resistance',{})
        if k.get('allow_k3_escape') or k.get('level')=='K3':
            requests.append(('resistance',str(k.get('representation_semantics',task.get('representation_kind','LOCAL_FUNCTION'))),dict(k.get('representation_context',{}))))
    elif spec.get('template')=='generic_mechanism_graph':
        for s in spec.get('knowledge',{}).get('slots',[]):
            if s.get('allow_k3_escape') or s.get('level')=='K3':
                requests.append((str(s['id']),str(s.get('representation_semantics',task.get('representation_kind','LOCAL_FUNCTION'))),dict(s.get('representation_context',{}))))
    if not requests:
        requests.append((None,'LOCAL_FUNCTION',{}))
    # task-level context supplements slot context, slot wins.
    out=[]
    for sid,k,c in requests:
        merged=dict(task); merged.update(c); out.append((sid,k,merged))
    return out


def build_representation_plan(spec: Mapping[str,Any]) -> tuple[RepresentationPlan,...]:
    return tuple(RepresentationPlan(k,assess_representation(k,c),sid) for sid,k,c in _slot_requests(spec))


def k3_representation_for_slot(spec: Mapping[str,Any], slot_id: str) -> RepresentationPlan:
    plans=build_representation_plan(spec)
    for p in plans:
        if p.slot_id==slot_id:
            return p
    return RepresentationPlan('LOCAL_FUNCTION',assess_representation('LOCAL_FUNCTION',{}),slot_id)
