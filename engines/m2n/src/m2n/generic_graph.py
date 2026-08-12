from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .canonical_ir import Dim, VarSpec, CNode, CanonicalIR, validate_canonical
from .knowledge_contract import (
    KnowledgeLevel, KnowledgeSource, MechanismSlot, SlotEvidence,
    NeuralArchitecturePlan, plan_neural_architecture,
)
from .architecture_lowering import FieldAssignment, FamilyPatch, SlotBinding

_DIM_KEYS = {'gain_dim','output_dim','coeff_dim','dt_dim'}


def dim_from_json(obj: Any | None) -> Dim:
    if obj is None:
        return Dim()
    if isinstance(obj, Dim):
        return obj
    if isinstance(obj, Mapping):
        return Dim.of(**obj)
    if isinstance(obj, (list,tuple)):
        return Dim.from_pairs(obj)
    raise TypeError(f'cannot parse dimension from {type(obj).__name__}')


def _params_from_json(params: Mapping[str,Any] | None) -> dict[str,Any]:
    out={}
    for k,v in dict(params or {}).items():
        if k in _DIM_KEYS:
            out[k]=dim_from_json(v)
        else:
            out[k]=v
    return out


def build_generic_canonical_ir(spec: Mapping[str,Any]) -> CanonicalIR:
    """Build a device-independent CanonicalIR from a declarative mechanism graph.

    The graph vocabulary is exactly the frozen CanonicalIR primitive vocabulary.
    Device cards do not gain hidden Python callbacks or arbitrary eval strings.
    """
    m=spec['mechanism']
    inputs=tuple(VarSpec(str(x['name']),dim_from_json(x.get('dim'))) for x in m['inputs'])
    nodes=tuple(CNode(str(n['id']),str(n['op']),tuple(map(str,n.get('inputs',()))),_params_from_json(n.get('params'))) for n in m['nodes'])
    outputs=[]; expected=[]
    for x in m['outputs']:
        name=str(x['name']); outputs.append(name); expected.append((name,dim_from_json(x.get('dim'))))
    meta=(('source_ir','generic_mechanism_graph'),)+tuple((str(k),str(v)) for k,v in m.get('metadata',{}).items())
    ir=CanonicalIR(inputs,nodes,tuple(outputs),tuple(expected),meta)
    validate_canonical(ir)
    return ir


def _normalize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k):(dim_from_json(v) if str(k) in _DIM_KEYS else _normalize_value(v)) for k,v in value.items()}
    if isinstance(value, list):
        return [_normalize_value(v) for v in value]
    return value

def _assignment_value(path: str, value: Any) -> Any:
    tail=path.split('.')[-1]
    if tail in _DIM_KEYS:
        return dim_from_json(value)
    return _normalize_value(value)


def build_generic_problem(
    spec: Mapping[str,Any],
    *,
    data_qualification: str,
    observation_qualification: str,
    reasons: tuple[str,...]=(),
    finite_status_by_slot: Mapping[str,str] | None=None,
) -> tuple[NeuralArchitecturePlan, CanonicalIR, tuple[SlotBinding,...]]:
    skeleton=build_generic_canonical_ir(spec)
    slots=[]; evidence={}; bindings=[]
    finite_status_by_slot=dict(finite_status_by_slot or {})
    for s in spec.get('knowledge',{}).get('slots',[]):
        sid=str(s['id']); level=KnowledgeLevel(str(s['level'])); source=KnowledgeSource(str(s.get('knowledge_source','DECLARED_ENGINEERING')))
        input_dims=tuple(dim_from_json(x) for x in s.get('input_dims',[{}]))
        output_dim=dim_from_json(s.get('output_dim'))
        exact=s.get('exact_family')
        finite=tuple(map(str,s.get('finite_candidates',())))
        slot=MechanismSlot(
            sid,level,str(s.get('role',sid)),input_dims,output_dim,
            exact_family=None if exact is None else str(exact),
            learnable_parameters=tuple(map(str,s.get('learnable_parameters',()))),
            finite_candidates=finite,
            hard_constraints=tuple(map(str,s.get('hard_constraints',()))),
            allow_k3_escape=bool(s.get('allow_k3_escape',False)),
            local_neural_family=s.get('local_neural_family'),
            placement_candidates=tuple(map(str,s.get('placement_candidates',()))),
            knowledge_source=source,
        )
        slots.append(slot)
        finite_status=(finite_status_by_slot.get(sid) or ('PENDING' if level is KnowledgeLevel.K2 else 'NOT_APPLICABLE'))
        evidence[sid]=SlotEvidence(
            data_qualification=data_qualification,
            observation_qualification=observation_qualification,
            hard_mechanism_status=str(s.get('hard_mechanism_status','COMPLETE')),
            finite_status=finite_status,
            placement_control_status=('DECLARED' if s.get('placement_control_declared',False) else 'NOT_REQUIRED'),
            reasons=reasons,
        )

        if level in (KnowledgeLevel.K2,KnowledgeLevel.K3) or (level is KnowledgeLevel.K2 and s.get('allow_k3_escape')):
            fam_map=spec.get('bindings',{}).get(sid,{})
            patches=[]
            required=list(finite)
            if level is KnowledgeLevel.K3:
                required=[slot.local_neural_family]
            elif slot.allow_k3_escape and slot.local_neural_family:
                required=required+[slot.local_neural_family]
            for fam in required:
                if fam not in fam_map:
                    raise ValueError(f'{sid}: missing generic graph binding for family {fam!r}')
                assignments=tuple(FieldAssignment(str(a['path']),_assignment_value(str(a['path']),a['value'])) for a in fam_map[fam])
                patches.append(FamilyPatch(str(fam),assignments))
            bindings.append(SlotBinding(sid,tuple(patches)))
    if not slots:
        # A pure fixed graph still needs one epistemic slot so the planner can emit READY.
        slots=[MechanismSlot('graph',KnowledgeLevel.K0,'fixed declared mechanism graph',(Dim(),),Dim(),exact_family='fixed_graph',knowledge_source=KnowledgeSource.DECLARED_ENGINEERING)]
        evidence['graph']=SlotEvidence(data_qualification='READY',observation_qualification='SUPPORTED',hard_mechanism_status='COMPLETE')
    return plan_neural_architecture(tuple(slots),evidence),skeleton,tuple(bindings)
