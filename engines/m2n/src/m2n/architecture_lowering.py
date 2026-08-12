from __future__ import annotations

from dataclasses import dataclass, is_dataclass, replace, asdict
from itertools import product
from typing import Any, Iterable, Mapping

from .canonical_ir import CanonicalIR, compile_canonical
from .conservation_ir import ConservationIR, compile_conservation_ir
from .mechanics_ir import SecondOrderMechanicsIR, compile_second_order_ir
from .block_ir import BlockLNLIR, compile_block_lnl_ir
from .valve_ir import FourWayValveIR, compile_four_way_valve_ir
from .peaked_response_ir import SeparablePeakIR, compile_separable_peak_ir
from .knowledge_contract import NeuralArchitecturePlan, SlotPlan


EXECUTABLE_ACTIONS = {
    'EXACT_COMPILE',
    'PARAMETRIC_COMPILE',
    'RESTRICTED_K2_SEARCH',
    'KEEP_FINITE',
    'PROMOTE_LOCAL_K3',
    'LOCAL_K3_CANDIDATE',
}


@dataclass(frozen=True)
class FieldAssignment:
    """One immutable update applied to an IR skeleton.

    ``path`` is a dotted dataclass/tuple path, e.g. ``resistance.law`` or
    ``fluxes.0.exponent``.  Paths are structural and contain no device names.
    """
    path: str
    value: Any

    def validate(self) -> 'FieldAssignment':
        if not self.path or any(not p for p in self.path.split('.')):
            raise ValueError('assignment path must be a non-empty dotted path')
        return self


@dataclass(frozen=True)
class FamilyPatch:
    """How one planner family label maps onto an executable IR skeleton."""
    family: str
    assignments: tuple[FieldAssignment, ...]

    def validate(self) -> 'FamilyPatch':
        if not self.family:
            raise ValueError('family name must be non-empty')
        if not self.assignments:
            raise ValueError(f'{self.family}: at least one assignment is required')
        for x in self.assignments:
            x.validate()
        return self


@dataclass(frozen=True)
class SlotBinding:
    """Binding between one knowledge slot and executable IR alternatives."""
    slot_id: str
    family_patches: tuple[FamilyPatch, ...]

    def validate(self) -> 'SlotBinding':
        if not self.slot_id:
            raise ValueError('slot_id must be non-empty')
        if not self.family_patches:
            raise ValueError(f'{self.slot_id}: at least one family patch is required')
        families=[x.family for x in self.family_patches]
        if len(families) != len(set(families)):
            raise ValueError(f'{self.slot_id}: duplicate family patches')
        for p in self.family_patches:
            p.validate()
        return self

    def patch_for(self, family: str) -> FamilyPatch:
        for p in self.family_patches:
            if p.family == family:
                return p
        raise KeyError(f'{self.slot_id}: no executable binding for family {family!r}')


@dataclass(frozen=True)
class ExecutableArchitectureCandidate:
    id: str
    family_choices: tuple[tuple[str, str], ...]
    ir: Any

    def to_dict(self):
        return {
            'id': self.id,
            'family_choices': dict(self.family_choices),
            'ir_type': type(self.ir).__name__,
        }


@dataclass(frozen=True)
class ExecutableArchitectureSpace:
    planning_status: str
    candidates: tuple[ExecutableArchitectureCandidate, ...]
    fixed_slots: tuple[str, ...]
    variable_slots: tuple[str, ...]

    def to_dict(self):
        return {
            'planning_status': self.planning_status,
            'candidate_count': len(self.candidates),
            'fixed_slots': list(self.fixed_slots),
            'variable_slots': list(self.variable_slots),
            'candidates': [x.to_dict() for x in self.candidates],
        }


def _replace_path(obj: Any, parts: tuple[str, ...], value: Any) -> Any:
    if not parts:
        return value
    head, *tail = parts
    rest=tuple(tail)
    if is_dataclass(obj):
        if not hasattr(obj, head):
            raise ValueError(f'field path does not exist: {head}')
        child=getattr(obj, head)
        return replace(obj, **{head: _replace_path(child, rest, value)})
    if isinstance(obj, tuple):
        try:
            idx=int(head)
        except ValueError as e:
            raise ValueError(f'tuple path element must be an integer, got {head!r}') from e
        if idx < 0 or idx >= len(obj):
            raise ValueError(f'tuple path index out of range: {idx}')
        items=list(obj); items[idx]=_replace_path(items[idx], rest, value)
        return tuple(items)
    if isinstance(obj, list):
        try:
            idx=int(head)
        except ValueError as e:
            raise ValueError(f'list path element must be an integer, got {head!r}') from e
        if idx < 0 or idx >= len(obj):
            raise ValueError(f'list path index out of range: {idx}')
        items=list(obj); items[idx]=_replace_path(items[idx], rest, value)
        return items
    if isinstance(obj, dict):
        if head not in obj and rest:
            raise ValueError(f'dict path key does not exist: {head!r}')
        out=dict(obj)
        if rest:
            out[head]=_replace_path(out[head],rest,value)
        else:
            out[head]=value
        return out
    raise ValueError(f'cannot descend through {type(obj).__name__} at {head!r}')


def apply_family_patch(ir: Any, patch: FamilyPatch) -> Any:
    patch.validate()
    out=ir
    for a in patch.assignments:
        out=_replace_path(out, tuple(a.path.split('.')), a.value)
    return out


def _families_for_plan(
    plan: SlotPlan,
    selected_family_by_slot: Mapping[str, str],
) -> tuple[str, ...]:
    if plan.action in ('EXACT_COMPILE','PARAMETRIC_COMPILE'):
        return ()
    if plan.action in ('RESTRICTED_K2_SEARCH','PROMOTE_LOCAL_K3','LOCAL_K3_CANDIDATE'):
        return tuple(plan.architecture_family)
    if plan.action == 'KEEP_FINITE':
        if plan.slot_id in selected_family_by_slot:
            selected=selected_family_by_slot[plan.slot_id]
            if selected not in plan.architecture_family:
                raise ValueError(f'{plan.slot_id}: selected family {selected!r} is not in retained finite family')
            return (selected,)
        if len(plan.architecture_family)==1:
            return tuple(plan.architecture_family)
        raise ValueError(
            f'{plan.slot_id}: KEEP_FINITE has multiple families; provide selected_family_by_slot '
            'so the retained model can be lowered unambiguously'
        )
    raise ValueError(f'{plan.slot_id}: non-executable planner action {plan.action!r}')


def lower_architecture_plan(
    plan: NeuralArchitecturePlan,
    ir_skeleton: Any,
    bindings: Iterable[SlotBinding],
    *,
    selected_family_by_slot: Mapping[str, str] | None = None,
    max_candidates: int = 64,
) -> ExecutableArchitectureSpace:
    """Lower a READY M2N knowledge plan into concrete executable IR candidates.

    This is the missing bridge between epistemic planning and network
    construction.  K0/K1 are already represented in ``ir_skeleton``.  K2/K3
    slots are the only source of candidate branching and are patched locally.
    """
    plan.validate()
    if plan.planning_status != 'READY':
        raise ValueError(f'architecture plan is {plan.planning_status}; blocked/ambiguous plans must not be compiled')
    if not is_dataclass(ir_skeleton):
        raise TypeError('ir_skeleton must be a frozen/dataclass IR object')
    if max_candidates < 1:
        raise ValueError('max_candidates must be positive')

    selected=dict(selected_family_by_slot or {})
    bs=tuple(bindings)
    for b in bs:
        b.validate()
    bind_map={b.slot_id:b for b in bs}
    if len(bind_map)!=len(bs):
        raise ValueError('duplicate SlotBinding slot ids')

    fixed=[]; variable=[]; axes=[]
    for sp in plan.slot_plans:
        if sp.action not in EXECUTABLE_ACTIONS:
            raise ValueError(f'{sp.slot_id}: planner action {sp.action!r} cannot be lowered')
        families=_families_for_plan(sp,selected)
        if not families:
            fixed.append(sp.slot_id)
            continue
        if sp.slot_id not in bind_map:
            raise ValueError(f'{sp.slot_id}: executable binding is required for {sp.action}')
        # Fail before building any models if planner and IR vocabulary disagree.
        for fam in families:
            bind_map[sp.slot_id].patch_for(fam)
        variable.append(sp.slot_id)
        axes.append((sp.slot_id,families))

    n=1
    for _,fams in axes:
        n*=len(fams)
    if n > max_candidates:
        raise ValueError(f'candidate cross-product {n} exceeds max_candidates={max_candidates}')

    if not axes:
        return ExecutableArchitectureSpace(
            'READY',
            (ExecutableArchitectureCandidate('fixed',(),ir_skeleton),),
            tuple(fixed),(),
        )

    out=[]
    for combo in product(*(fams for _,fams in axes)):
        choice=tuple((axes[i][0],combo[i]) for i in range(len(axes)))
        ir=ir_skeleton
        for slot_id,family in choice:
            ir=apply_family_patch(ir,bind_map[slot_id].patch_for(family))
        cid='__'.join(f'{sid}={fam}' for sid,fam in choice)
        out.append(ExecutableArchitectureCandidate(cid,choice,ir))
    return ExecutableArchitectureSpace('READY',tuple(out),tuple(fixed),tuple(variable))


def compile_executable_ir(ir: Any):
    """Compile a supported M2N IR without device-specific dispatch."""
    if isinstance(ir, CanonicalIR):
        return compile_canonical(ir)
    if isinstance(ir, ConservationIR):
        return compile_conservation_ir(ir)[0]
    if isinstance(ir, SecondOrderMechanicsIR):
        return compile_second_order_ir(ir)[0]
    if isinstance(ir, BlockLNLIR):
        return compile_block_lnl_ir(ir)
    if isinstance(ir, FourWayValveIR):
        return compile_four_way_valve_ir(ir)[0]
    if isinstance(ir, SeparablePeakIR):
        return compile_separable_peak_ir(ir)
    raise TypeError(f'unsupported executable IR type {type(ir).__name__}')


def compile_architecture_space(space: ExecutableArchitectureSpace) -> dict[str, Any]:
    return {c.id: compile_executable_ir(c.ir) for c in space.candidates}
