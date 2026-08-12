from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Mapping


class BiasRoute(str, Enum):
    OBSERVATIONAL = 'OBSERVATIONAL'
    INDUCTIVE_HARD = 'INDUCTIVE_HARD'
    LEARNING_SOFT = 'LEARNING_SOFT'
    HYBRID = 'HYBRID'
    UNRESOLVED = 'UNRESOLVED'


_HARD_FORMS = {
    'EXACT_EQUATION','EXACT_CONSERVATION','EXACT_INVARIANT','EXACT_BOUNDARY',
    'EXACT_SYMMETRY','DIMENSIONAL_CONSTRAINT','KNOWN_TOPOLOGY','KNOWN_OPERATOR',
}
_SOFT_FORMS = {
    'APPROXIMATE_EQUATION','MONOTONICITY','IRREVERSIBILITY','SIGN_CONSTRAINT',
    'SOFT_CONSERVATION','EMPIRICAL_TREND','REGULARITY','SMOOTHNESS',
}
_OBSERVATIONAL_FORMS = {
    'VIRTUAL_SENSOR','DATA_AUGMENTATION','SIMULATION_DATA','EXPERIMENT_DESIGN',
    'DATA_SELECTION','MULTI_SENSOR_FUSION','MULTI_FIDELITY_DATA','ACTIVE_SAMPLING',
}


@dataclass(frozen=True)
class PhysicsKnowledgeItem:
    id: str
    knowledge_form: str
    trust: str = 'MEDIUM'  # HIGH | MEDIUM | LOW
    requested_route: str | None = None
    notes: str = ''

    @classmethod
    def from_mapping(cls, m: Mapping[str,Any]) -> 'PhysicsKnowledgeItem':
        return cls(
            id=str(m['id']),
            knowledge_form=str(m.get('knowledge_form',m.get('kind',''))).upper(),
            trust=str(m.get('trust','MEDIUM')).upper(),
            requested_route=None if m.get('route') is None else str(m['route']).upper(),
            notes=str(m.get('notes','')),
        )


@dataclass(frozen=True)
class BiasRoutingDecision:
    item_id: str
    route: str
    status: str  # ROUTED | WARN | UNRESOLVED
    reason: str
    action: str

    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class BiasRoutePlan:
    decisions: tuple[BiasRoutingDecision,...]

    @property
    def status(self) -> str:
        if any(x.status=='UNRESOLVED' for x in self.decisions): return 'PARTIAL'
        if any(x.status=='WARN' for x in self.decisions): return 'READY_WITH_WARNINGS'
        return 'READY'

    def to_dict(self):
        return {'status':self.status,'decisions':[x.to_dict() for x in self.decisions]}


def _route_action(route: BiasRoute) -> str:
    return {
        BiasRoute.OBSERVATIONAL:'modify data acquisition/representation/coverage rather than the network topology',
        BiasRoute.INDUCTIVE_HARD:'encode the trusted constraint in the executable architecture/IR so it is satisfied by construction',
        BiasRoute.LEARNING_SOFT:'encode the approximate prior as a loss/constraint/training objective; do not freeze it as exact physics',
        BiasRoute.HYBRID:'combine a hard structural core with an observational and/or soft-learning treatment for the uncertain part',
        BiasRoute.UNRESOLVED:'do not inject this knowledge until its epistemic meaning and intended use are clarified',
    }[route]


def route_physics_knowledge(item: PhysicsKnowledgeItem) -> BiasRoutingDecision:
    if not item.id:
        raise ValueError('physics knowledge item id must be non-empty')
    if item.trust not in {'HIGH','MEDIUM','LOW'}:
        raise ValueError(f'{item.id}: trust must be HIGH, MEDIUM, or LOW')

    if item.requested_route:
        try: route=BiasRoute(item.requested_route)
        except Exception as e: raise ValueError(f'{item.id}: invalid requested route {item.requested_route!r}') from e
        status='WARN' if route is BiasRoute.INDUCTIVE_HARD and item.trust!='HIGH' else 'ROUTED'
        reason='explicit Device Card route'
        if status=='WARN': reason+='; hard architectural enforcement is not automatically justified below HIGH trust'
        return BiasRoutingDecision(item.id,route.value,status,reason,_route_action(route))

    form=item.knowledge_form
    if form in _OBSERVATIONAL_FORMS:
        route=BiasRoute.OBSERVATIONAL; status='ROUTED'; reason=f'{form} changes what information is observed/generated rather than the model law itself'
    elif form in _HARD_FORMS:
        if item.trust=='HIGH':
            route=BiasRoute.INDUCTIVE_HARD; status='ROUTED'; reason=f'{form} is declared high-trust structural knowledge'
        else:
            route=BiasRoute.LEARNING_SOFT; status='WARN'; reason=f'{form} is not HIGH trust; M2N refuses to freeze uncertain physics as an exact architecture constraint'
    elif form in _SOFT_FORMS:
        route=BiasRoute.LEARNING_SOFT; status='ROUTED'; reason=f'{form} is naturally an approximate preference/constraint rather than exact executable physics'
    elif form in {'INCOMPLETE_PHYSICS','LOW_FIDELITY_PHYSICS','PARTIAL_GOVERNING_LAW'}:
        route=BiasRoute.HYBRID; status='ROUTED'; reason=f'{form} should retain the credible physical core while leaving the missing part learnable/qualifiable'
    else:
        route=BiasRoute.UNRESOLVED; status='UNRESOLVED'; reason=f'unrecognized knowledge_form {form!r}; no silent physics injection is allowed'
    return BiasRoutingDecision(item.id,route.value,status,reason,_route_action(route))


def build_bias_route_plan(spec: Mapping[str,Any]) -> BiasRoutePlan:
    items=spec.get('physics_biases',spec.get('bias_policy',{}).get('knowledge_items',()))
    decisions=tuple(route_physics_knowledge(PhysicsKnowledgeItem.from_mapping(x)) for x in items)
    return BiasRoutePlan(decisions)
