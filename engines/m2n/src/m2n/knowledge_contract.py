from __future__ import annotations
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Iterable

from .canonical_ir import Dim


class KnowledgeSource(str, Enum):
    """Provenance of the knowledge supplied to M2N.

    This is separate from K0/K1/K2/K3.  The K-level says *how much of the
    local mechanism is known*; the source says *why we believe it*.
    Simulator truth marked POSTHOC is never admissible as architecture-selection
    input.
    """
    DECLARED_ENGINEERING = 'DECLARED_ENGINEERING'
    COMMISSIONING_EVIDENCE = 'COMMISSIONING_EVIDENCE'
    FINITE_ENGINEERING_HYPOTHESIS = 'FINITE_ENGINEERING_HYPOTHESIS'
    BLINDED_FOR_VALIDATION = 'BLINDED_FOR_VALIDATION'
    SIMULATOR_TRUTH_POSTHOC = 'SIMULATOR_TRUTH_POSTHOC'


class KnowledgeLevel(str, Enum):
    """Epistemic level of a *local mechanism slot*, not a whole-model score.

    K0: structure/operator/parameter are trusted and fixed.
    K1: mechanism family is trusted, local continuous parameters are uncertain.
    K2: a finite, physically admissible local family is uncertain.
    K3: no credible finite local family is available; a constrained local neural
        constitutive law may be considered after qualification.
    """
    K0 = 'K0'
    K1 = 'K1'
    K2 = 'K2'
    K3 = 'K3'


PLANNING_STATUS = {'READY', 'AMBIGUOUS', 'UNSUPPORTED'}
DATA_QUALIFICATION = {'READY', 'AMBIGUOUS', 'UNSUPPORTED'}
OBSERVATION_QUALIFICATION = {'SUPPORTED', 'PARTIAL', 'UNSUPPORTED'}
FINITE_STATUS = {'NOT_APPLICABLE', 'PENDING', 'RETAINED', 'REJECTED'}
HARD_MECHANISM_STATUS = {'COMPLETE', 'MISSING', 'UNKNOWN'}
PLACEMENT_CONTROL_STATUS = {'NOT_REQUIRED', 'DECLARED', 'MISSING'}


@dataclass(frozen=True)
class MechanismSlot:
    """Knowledge contract for one local physical/engineering relation.

    The contract is intentionally local.  A device may contain K0, K1, K2 and
    K3 slots simultaneously.  M2N must not label the entire device "K3" merely
    because one constitutive law is unknown.
    """
    id: str
    level: KnowledgeLevel | str
    role: str
    input_dims: tuple[Dim, ...]
    output_dim: Dim
    exact_family: str | None = None
    fixed_parameters: tuple[str, ...] = ()
    learnable_parameters: tuple[str, ...] = ()
    finite_candidates: tuple[str, ...] = ()
    hard_constraints: tuple[str, ...] = ()
    allow_k3_escape: bool = False
    local_neural_family: str | None = None
    placement_candidates: tuple[str, ...] = ()
    knowledge_source: KnowledgeSource | str = KnowledgeSource.DECLARED_ENGINEERING

    def normalized_level(self) -> KnowledgeLevel:
        try:
            return self.level if isinstance(self.level, KnowledgeLevel) else KnowledgeLevel(self.level)
        except ValueError as e:
            raise ValueError(f'{self.id}: invalid knowledge level {self.level!r}') from e

    def normalized_source(self) -> KnowledgeSource:
        try:
            return self.knowledge_source if isinstance(self.knowledge_source, KnowledgeSource) else KnowledgeSource(self.knowledge_source)
        except ValueError as e:
            raise ValueError(f'{self.id}: invalid knowledge source {self.knowledge_source!r}') from e

    def validate(self) -> 'MechanismSlot':
        if not self.id.strip():
            raise ValueError('mechanism slot id must be non-empty')
        if not self.role.strip():
            raise ValueError(f'{self.id}: role must be non-empty')
        if not self.input_dims:
            raise ValueError(f'{self.id}: at least one input dimension is required')
        level = self.normalized_level()
        source = self.normalized_source()
        if source is KnowledgeSource.SIMULATOR_TRUTH_POSTHOC:
            raise ValueError(f'{self.id}: simulator truth marked POSTHOC cannot be used as an M2N design input')
        if level is KnowledgeLevel.K0:
            if not self.exact_family:
                raise ValueError(f'{self.id}: K0 requires exact_family')
            if self.learnable_parameters or self.finite_candidates or self.allow_k3_escape or self.local_neural_family:
                raise ValueError(f'{self.id}: K0 cannot contain learnable/search/neural freedom')
        elif level is KnowledgeLevel.K1:
            if not self.exact_family:
                raise ValueError(f'{self.id}: K1 requires a known exact_family')
            if not self.learnable_parameters:
                raise ValueError(f'{self.id}: K1 requires at least one learnable parameter')
            if self.finite_candidates:
                raise ValueError(f'{self.id}: finite structural alternatives belong to K2, not K1')
            if self.local_neural_family:
                raise ValueError(f'{self.id}: K1 cannot silently include a neural branch')
        elif level is KnowledgeLevel.K2:
            if len(self.finite_candidates) < 2:
                raise ValueError(f'{self.id}: K2 requires at least two finite candidates')
            if self.local_neural_family and not self.allow_k3_escape:
                raise ValueError(f'{self.id}: K2 neural escape must be explicitly enabled')
            if self.allow_k3_escape and not self.local_neural_family:
                raise ValueError(f'{self.id}: K2 K3-escape requires local_neural_family')
        elif level is KnowledgeLevel.K3:
            if self.finite_candidates:
                raise ValueError(f'{self.id}: credible finite candidates imply K2 before K3')
            if not self.local_neural_family:
                raise ValueError(f'{self.id}: K3 requires a declared constrained local_neural_family')
            if self.exact_family:
                raise ValueError(f'{self.id}: K3 cannot also declare a trusted exact_family')
        return self


@dataclass(frozen=True)
class SlotEvidence:
    """Evidence that controls whether a knowledge contract may advance.

    ``finite_status`` is relevant to K2.  A K2 slot may promote to K3 only after
    its frozen finite family is REJECTED.  Lack of excitation/observation must
    produce AMBIGUOUS/UNSUPPORTED rather than a neural fallback.
    """
    data_qualification: str = 'READY'
    observation_qualification: str = 'SUPPORTED'
    hard_mechanism_status: str = 'COMPLETE'
    finite_status: str = 'NOT_APPLICABLE'
    placement_control_status: str = 'NOT_REQUIRED'
    support_fraction_outside: float | None = None
    reasons: tuple[str, ...] = ()

    def validate(self) -> 'SlotEvidence':
        checks = (
            (self.data_qualification, DATA_QUALIFICATION, 'data_qualification'),
            (self.observation_qualification, OBSERVATION_QUALIFICATION, 'observation_qualification'),
            (self.hard_mechanism_status, HARD_MECHANISM_STATUS, 'hard_mechanism_status'),
            (self.finite_status, FINITE_STATUS, 'finite_status'),
            (self.placement_control_status, PLACEMENT_CONTROL_STATUS, 'placement_control_status'),
        )
        for value, allowed, name in checks:
            if value not in allowed:
                raise ValueError(f'{name}: {value}')
        if self.support_fraction_outside is not None and not (0.0 <= self.support_fraction_outside <= 1.0):
            raise ValueError('support_fraction_outside must lie in [0,1]')
        return self


@dataclass(frozen=True)
class SlotPlan:
    slot_id: str
    level: str
    action: str
    architecture_family: tuple[str, ...]
    k3_enabled: bool
    planning_status: str
    reasons: tuple[str, ...] = ()

    def validate(self) -> 'SlotPlan':
        if self.planning_status not in PLANNING_STATUS:
            raise ValueError(f'planning_status: {self.planning_status}')
        if self.k3_enabled and self.action not in ('PROMOTE_LOCAL_K3', 'LOCAL_K3_CANDIDATE'):
            raise ValueError('k3_enabled requires a K3 action')
        return self


@dataclass(frozen=True)
class NeuralArchitecturePlan:
    """M2N pre-training architecture plan.

    READY means the *next architecture experiment* is well-posed.  It does not
    mean the final trained architecture has already been validated.
    """
    planning_status: str
    slot_plans: tuple[SlotPlan, ...]
    exact_slots: tuple[str, ...]
    parameterized_slots: tuple[str, ...]
    k2_search_slots: tuple[str, ...]
    k3_candidate_slots: tuple[str, ...]
    reasons: tuple[str, ...] = ()

    def validate(self) -> 'NeuralArchitecturePlan':
        if self.planning_status not in PLANNING_STATUS:
            raise ValueError(f'planning_status: {self.planning_status}')
        for p in self.slot_plans:
            p.validate()
        return self

    def to_dict(self):
        self.validate()
        return asdict(self)


def _blocked_status(ev: SlotEvidence) -> tuple[str, tuple[str, ...]] | None:
    reasons = list(ev.reasons)
    if ev.hard_mechanism_status == 'MISSING':
        return 'AMBIGUOUS', tuple(reasons + ['known hard mechanism must be repaired before architecture search'])
    if ev.hard_mechanism_status == 'UNKNOWN':
        return 'AMBIGUOUS', tuple(reasons + ['hard-mechanism completeness is unresolved'])
    if ev.data_qualification == 'UNSUPPORTED' or ev.observation_qualification == 'UNSUPPORTED':
        return 'UNSUPPORTED', tuple(reasons + ['data/observation contract does not support this physical mechanism slot'])
    if ev.data_qualification == 'AMBIGUOUS' or ev.observation_qualification == 'PARTIAL':
        return 'AMBIGUOUS', tuple(reasons + ['data/observation qualification is insufficient for architecture attribution'])
    return None


def _knowledge_source_notice(slot: MechanismSlot) -> tuple[str, ...]:
    """Human-visible provenance notice carried into every slot plan.

    BLINDED_FOR_VALIDATION means the experimenter knows more mechanism truth
    than M2N is allowed to use in this run.  It is admissible for controlled
    capability tests, but the plan must surface the blinding explicitly so a
    recovered mechanism is not later misreported as genuinely unknown in the
    engineering application.
    """
    if slot.normalized_source() is KnowledgeSource.BLINDED_FOR_VALIDATION:
        return ('BLINDED-FOR-VALIDATION: known mechanism information is deliberately withheld from M2N in this controlled test.',)
    return ()


def plan_mechanism_slot(slot: MechanismSlot, evidence: SlotEvidence | None = None) -> SlotPlan:
    slot.validate()
    ev = (evidence or SlotEvidence()).validate()
    level = slot.normalized_level()

    # K0/K1 are mechanism declarations rather than data-driven architecture
    # searches.  Missing hard mechanism still blocks K1 parameter fitting, but
    # an exact K0 relation can remain compiled while the broader device plan is
    # marked by other unresolved slots.
    if level is KnowledgeLevel.K0:
        return SlotPlan(slot.id, level.value, 'EXACT_COMPILE', (slot.exact_family or '',), False, 'READY',
                        (('trusted hard mechanism is compiled without learning',)+_knowledge_source_notice(slot))).validate()

    blocked = _blocked_status(ev)
    if blocked is not None:
        status, reasons = blocked
        return SlotPlan(slot.id, level.value, 'BLOCKED', (), False, status, reasons+_knowledge_source_notice(slot)).validate()

    if level is KnowledgeLevel.K1:
        return SlotPlan(slot.id, level.value, 'PARAMETRIC_COMPILE', (slot.exact_family or '',), False, 'READY',
                        (('mechanism family fixed; learn only declared continuous parameters',)+_knowledge_source_notice(slot))).validate()

    if level is KnowledgeLevel.K2:
        if ev.finite_status == 'NOT_APPLICABLE':
            raise ValueError(f'{slot.id}: K2 evidence finite_status cannot be NOT_APPLICABLE')
        if ev.finite_status == 'PENDING':
            return SlotPlan(slot.id, level.value, 'RESTRICTED_K2_SEARCH', tuple(slot.finite_candidates), False, 'READY',
                            (('search only the frozen finite local family',)+_knowledge_source_notice(slot))).validate()
        if ev.finite_status == 'RETAINED':
            return SlotPlan(slot.id, level.value, 'KEEP_FINITE', tuple(slot.finite_candidates), False, 'READY',
                            (('validation retained a finite mechanism; K3 remains off',)+_knowledge_source_notice(slot))).validate()
        # REJECTED
        if not slot.allow_k3_escape:
            return SlotPlan(slot.id, level.value, 'FINITE_FAMILY_REJECTED', tuple(slot.finite_candidates), False, 'AMBIGUOUS',
                            (('finite K2 rejected but no declared K3 escape exists',)+_knowledge_source_notice(slot))).validate()
        if len(slot.placement_candidates) > 1 and ev.placement_control_status != 'DECLARED':
            return SlotPlan(slot.id, level.value, 'BLOCKED', (), False, 'AMBIGUOUS',
                            (('K3 promotion requires equal-capacity wrong-placement controls',)+_knowledge_source_notice(slot))).validate()
        return SlotPlan(slot.id, level.value, 'PROMOTE_LOCAL_K3', (slot.local_neural_family or '',), True, 'READY',
                        (('frozen K2 family rejected; local K3 may now compete under validation gates',)+_knowledge_source_notice(slot))).validate()

    # Native K3: only appropriate when no credible finite local family exists.
    if len(slot.placement_candidates) > 1 and ev.placement_control_status != 'DECLARED':
        return SlotPlan(slot.id, level.value, 'BLOCKED', (), False, 'AMBIGUOUS',
                        (('local K3 placement is not attributable without declared placement controls',)+_knowledge_source_notice(slot))).validate()
    return SlotPlan(slot.id, level.value, 'LOCAL_K3_CANDIDATE', (slot.local_neural_family or '',), True, 'READY',
                    (('no credible finite local family declared; constrained local K3 is admissible',)+_knowledge_source_notice(slot))).validate()


def plan_neural_architecture(
    slots: Iterable[MechanismSlot],
    evidence_by_slot: dict[str, SlotEvidence] | None = None,
) -> NeuralArchitecturePlan:
    slots = tuple(slots)
    if not slots:
        raise ValueError('at least one mechanism slot is required')
    ids = [s.id for s in slots]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate mechanism slot ids')
    evidence_by_slot = evidence_by_slot or {}
    unknown = set(evidence_by_slot) - set(ids)
    if unknown:
        raise ValueError(f'evidence supplied for unknown slots: {sorted(unknown)}')

    plans = tuple(plan_mechanism_slot(s, evidence_by_slot.get(s.id)) for s in slots)
    if any(p.planning_status == 'UNSUPPORTED' for p in plans):
        status = 'UNSUPPORTED'
    elif any(p.planning_status == 'AMBIGUOUS' for p in plans):
        status = 'AMBIGUOUS'
    else:
        status = 'READY'

    exact = tuple(p.slot_id for p in plans if p.action == 'EXACT_COMPILE')
    param = tuple(p.slot_id for p in plans if p.action == 'PARAMETRIC_COMPILE')
    k2 = tuple(p.slot_id for p in plans if p.action == 'RESTRICTED_K2_SEARCH')
    k3 = tuple(p.slot_id for p in plans if p.k3_enabled)
    blocking_reasons = tuple(r for p in plans for r in p.reasons if p.planning_status != 'READY')
    blinding_notices = tuple(r for p in plans for r in p.reasons if r.startswith('BLINDED-FOR-VALIDATION:'))
    reasons = tuple(dict.fromkeys(blocking_reasons + blinding_notices))
    return NeuralArchitecturePlan(status, plans, exact, param, k2, k3, reasons).validate()
