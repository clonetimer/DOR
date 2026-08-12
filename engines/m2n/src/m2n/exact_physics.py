from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .canonical_ir import CanonicalIR
from .mechanics_ir import SecondOrderMechanicsIR, lower_second_order_ir
from .peaked_response_ir import SeparablePeakIR

_LEARNED_OPS={'local_neural','signed_local_neural','signed_gain_local_neural','dissipative_local_neural'}
_PARAM_OPS={'positive_gain','signed_gain','signed_constant_like','stribeck_resistance'}
_TOTALIZED_OPS={'sqrt_pos','power_pos','sqrt_pos_diff_scaled','signed_sqrt_diff_scaled','clamp','threshold_gate'}


@dataclass(frozen=True)
class ExactPhysicsContract:
    ir_type: str
    exact_node_ids: tuple[str,...]
    trainable_physics_node_ids: tuple[str,...]
    learned_node_ids: tuple[str,...]
    totalized_or_guarded_node_ids: tuple[str,...]
    domain_notes: tuple[str,...]
    compiled_semantics_exactness: str = 'WITH_RESPECT_TO_DECLARED_IR_FLOATING_POINT_SEMANTICS'
    physical_truth_claim: str = 'NOT_ESTABLISHED_BY_COMPILATION'

    def to_dict(self): return asdict(self)


def _canonical(ir: Any) -> CanonicalIR | None:
    if isinstance(ir,CanonicalIR): return ir
    if isinstance(ir,SecondOrderMechanicsIR): return lower_second_order_ir(ir)[0]
    return None


def audit_exact_physics(ir: Any) -> ExactPhysicsContract:
    c=_canonical(ir)
    if c is not None:
        exact=[]; param=[]; learned=[]; guarded=[]; notes=[]
        for n in c.nodes:
            if n.op in _LEARNED_OPS:
                learned.append(n.id); continue
            if n.op in _PARAM_OPS and (n.params.get('learnable',False) or n.op=='stribeck_resistance'):
                param.append(n.id)
            else:
                exact.append(n.id)
            if n.op in _TOTALIZED_OPS:
                guarded.append(n.id)
                if n.op in {'sqrt_pos','power_pos','sqrt_pos_diff_scaled'}:
                    notes.append(f'{n.id}: positive-domain operation is totalized with clamp/epsilon; exactness is relative to this declared IR semantics, not an unconstrained mathematical sqrt/power')
                elif n.op=='signed_sqrt_diff_scaled':
                    notes.append(f'{n.id}: signed square-root uses abs/sign with epsilon by declared IR semantics')
                else:
                    notes.append(f'{n.id}: {n.op} intentionally changes behavior outside/at a declared guard boundary')
        notes.append('Compilation exactness does not establish that the engineering equation is physically true; KnowledgeSource/validation remain authoritative.')
        return ExactPhysicsContract(type(ir).__name__,tuple(exact),tuple(param),tuple(learned),tuple(guarded),tuple(dict.fromkeys(notes)))
    if isinstance(ir,SeparablePeakIR):
        guarded=[]; notes=[]
        for a in ir.axes:
            if a.name in ir.active_axes and a.transform=='log':
                guarded.append(a.name); notes.append(f'{a.name}: log coordinate is implemented with a positive clamp; physical input should remain in the declared positive validity domain')
        notes.append('SeparablePeakIR is a parametric constitutive family; exactness means exact evaluation of the chosen family, not proof that the family is the true mechanism.')
        return ExactPhysicsContract(type(ir).__name__,(),('amplitude',)+tuple(ir.active_axes),(),tuple(guarded),tuple(notes))
    return ExactPhysicsContract(type(ir).__name__,(),(),(),(),('No exact-physics audit adapter is defined for this IR type; no exactness claim is emitted.',))
