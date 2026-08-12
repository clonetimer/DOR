from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Mapping, Sequence

from .selection import ArchitectureSelectionPolicy, ArchitectureSelectionPlan, select_architecture
from .architecture_lowering import (
    SlotBinding, ExecutableArchitectureCandidate, ExecutableArchitectureSpace,
    lower_architecture_plan, compile_executable_ir,
)
from .knowledge_contract import NeuralArchitecturePlan


CandidateEvaluator = Callable[[ExecutableArchitectureCandidate, Any], Sequence[float]]


@dataclass(frozen=True)
class DesignSessionResult:
    status: str  # SELECTED | AMBIGUOUS | FIXED
    planning_status: str
    selected_candidate_id: str | None
    selected_family_choices: tuple[tuple[str,str], ...]
    candidate_validation_risks: tuple[tuple[str, tuple[float, ...]], ...]
    selection: ArchitectureSelectionPlan | None
    candidate_space: ExecutableArchitectureSpace

    def to_dict(self):
        return {
            'status': self.status,
            'planning_status': self.planning_status,
            'selected_candidate_id': self.selected_candidate_id,
            'selected_family_choices': dict(self.selected_family_choices),
            'candidate_validation_risks': {k:list(v) for k,v in self.candidate_validation_risks},
            'selection': None if self.selection is None else self.selection.as_dict(),
            'candidate_space': self.candidate_space.to_dict(),
        }


def run_design_session(
    plan: NeuralArchitecturePlan,
    ir_skeleton: Any,
    bindings: Sequence[SlotBinding],
    evaluator: CandidateEvaluator,
    *,
    selected_family_by_slot: Mapping[str,str] | None = None,
    selection_policy: ArchitectureSelectionPolicy = ArchitectureSelectionPolicy(),
    max_candidates: int = 64,
) -> DesignSessionResult:
    """Execute one M2N architecture-design session on normal-data evidence.

    ``evaluator`` is the Estimation Contract: it receives one compiled candidate
    and returns repeated non-negative validation risks.  M2N owns the admissible
    candidate space and the selection policy; the caller owns the task-specific
    estimator/data loader.  This preserves the project boundary that one-step,
    trajectory and output-error objectives are not interchangeable.
    """
    space=lower_architecture_plan(
        plan,ir_skeleton,bindings,
        selected_family_by_slot=selected_family_by_slot,
        max_candidates=max_candidates,
    )
    if len(space.candidates)==1:
        c=space.candidates[0]
        # A fixed K0/K1 or already-resolved finite architecture does not require
        # artificial architecture competition.  Training/fitting occurs later.
        return DesignSessionResult('FIXED',plan.planning_status,c.id,c.family_choices,(),None,space)

    risks=[]
    for c in space.candidates:
        model=compile_executable_ir(c.ir)
        vals=tuple(float(x) for x in evaluator(c,model))
        risks.append((c.id,vals))
    selection=select_architecture(dict(risks),policy=selection_policy)
    if selection.status=='SELECTED':
        chosen=next(c for c in space.candidates if c.id==selection.selected)
        return DesignSessionResult('SELECTED',plan.planning_status,chosen.id,chosen.family_choices,tuple(risks),selection,space)
    return DesignSessionResult('AMBIGUOUS',plan.planning_status,None,(),tuple(risks),selection,space)
