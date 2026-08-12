from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum

from .canonical_ir import Dim
from .knowledge_contract import (
    KnowledgeLevel, KnowledgeSource, MechanismSlot, SlotEvidence,
    plan_mechanism_slot,
)


class NoveltyStatus(str, Enum):
    KILLED = 'KILLED_AS_STANDALONE_NOVELTY'
    PARTIAL = 'PARTIALLY_KILLED_OR_DOMAIN_SPECIFIC_PRIOR_ART'
    SURVIVES_SEARCHED_SET = 'NOT_KILLED_BY_SEARCHED_SET'
    INFRASTRUCTURE = 'USEFUL_INFRASTRUCTURE_NOT_NOVELTY_CLAIM'


@dataclass(frozen=True)
class NoveltyClaim:
    id: str
    claim: str
    status: NoveltyStatus
    closest_prior_art: tuple[str, ...]
    disposition: str
    caution: str = ''

    def as_dict(self):
        d=asdict(self); d['status']=self.status.value; return d


def default_claim_ledger() -> tuple[NoveltyClaim, ...]:
    return (
        NoveltyClaim('C1','Compile known symbolic/mechanistic relations into differentiable modules',NoveltyStatus.KILLED,('The Neural Compiler','equation-based differentiable modeling'),'Keep as infrastructure; do not claim as the main novelty.'),
        NoveltyClaim('C2','Embed a local neural component for unknown/missing physics',NoveltyStatus.KILLED,('Universal Differential Equations','APHYNITY','unknown constitutive neural models'),'Keep and strengthen via optional hybrid-learning backends; cite prior art.'),
        NoveltyClaim('C3','Select among a finite physically admissible constitutive/model family using data',NoveltyStatus.PARTIAL,('Constitutive Artificial Neural Networks',),'Keep as cross-domain capability; do not claim the general idea of data-driven physical model selection as new.'),
        NoveltyClaim('C4','Discover governing/symbolic expressions from data',NoveltyStatus.KILLED,('SINDy/PySINDy','SymMatika','symbolic regression'),'Do not rebuild as an M2N novelty. Use discovery backends to propose K2 candidates.'),
        NoveltyClaim('C5','Use explicit knowledge level plus observation/data qualification to decide which architecture operations are admissible',NoveltyStatus.SURVIVES_SEARCHED_SET,('related identifiability/grey-box literature',),'Retain as the primary NKT survivor; requires broader literature review before publication.', 'No exact end-to-end counterpart was found in the bounded NKT-1 source set; this is not proof of global novelty.'),
        NoveltyClaim('C6','Slot-wise K0/K1/K2/K3 adjudication with finite physics before local neuralization',NoveltyStatus.SURVIVES_SEARCHED_SET,('UDE/APHYNITY cover K3 hybridization; CANN covers domain-specific finite selection',),'Retain as a system-level orchestration claim, not as a claim that the individual ingredients are new.', 'Combination novelty remains to be established.'),
        NoveltyClaim('C7','Return SELECTED / finite set / AMBIGUOUS / UNSUPPORTED instead of forcing a single model',NoveltyStatus.SURVIVES_SEARCHED_SET,('selective prediction/abstention ideas broadly exist',),'Keep as a core behavior and evaluation target.', 'Abstention itself is not new; the claim is its architecture-design role under mechanism/observation contracts.'),
        NoveltyClaim('C8','Require equal-capacity wrong-placement controls before attributing a local K3 placement',NoveltyStatus.SURVIVES_SEARCHED_SET,('hybrid modeling ablations broadly exist',),'Keep as a falsification protocol; perform a dedicated literature search before claiming originality.'),
        NoveltyClaim('C9','Knowledge-source provenance (engineering known / blinded validation / posthoc truth) gates architecture claims',NoveltyStatus.INFRASTRUCTURE,('experimental blinding and provenance best practices',),'Keep as research-integrity infrastructure; do not market it as a standalone algorithmic novelty.'),
    )


@dataclass(frozen=True)
class BehavioralNKTResult:
    case: str
    expected_status: str
    expected_action: str | None
    actual_status: str
    actual_action: str | None
    passed: bool


def run_behavioral_nkt() -> tuple[BehavioralNKTResult, ...]:
    d=Dim()
    cases=[]
    def run(name,slot,ev,status,action):
        try:
            p=plan_mechanism_slot(slot,ev)
            a_status=p.planning_status; a_action=p.action
        except Exception as e:
            a_status='ERROR'; a_action=type(e).__name__
        cases.append(BehavioralNKTResult(name,status,action,a_status,a_action,a_status==status and a_action==action))

    run('K1 known family stays parametric', MechanismSlot('s',KnowledgeLevel.K1,'r',(d,),d,exact_family='known',learnable_parameters=('a',)), SlotEvidence(), 'READY','PARAMETRIC_COMPILE')
    run('K2 pending stays finite', MechanismSlot('s',KnowledgeLevel.K2,'r',(d,),d,finite_candidates=('a','b')), SlotEvidence(finite_status='PENDING'), 'READY','RESTRICTED_K2_SEARCH')
    run('K2 retained does not neuralize', MechanismSlot('s',KnowledgeLevel.K2,'r',(d,),d,finite_candidates=('a','b')), SlotEvidence(finite_status='RETAINED'), 'READY','KEEP_FINITE')
    run('ambiguous data blocks search', MechanismSlot('s',KnowledgeLevel.K2,'r',(d,),d,finite_candidates=('a','b')), SlotEvidence(data_qualification='AMBIGUOUS',finite_status='PENDING'), 'AMBIGUOUS','BLOCKED')
    run('unsupported observation rejects search', MechanismSlot('s',KnowledgeLevel.K2,'r',(d,),d,finite_candidates=('a','b')), SlotEvidence(observation_qualification='UNSUPPORTED',finite_status='PENDING'), 'UNSUPPORTED','BLOCKED')
    run('K3 allowed only after finite rejection', MechanismSlot('s',KnowledgeLevel.K2,'r',(d,),d,finite_candidates=('a','b'),allow_k3_escape=True,local_neural_family='local_nn'), SlotEvidence(finite_status='REJECTED'), 'READY','PROMOTE_LOCAL_K3')
    run('missing placement control blocks attribution', MechanismSlot('s',KnowledgeLevel.K2,'r',(d,),d,finite_candidates=('a','b'),allow_k3_escape=True,local_neural_family='local_nn',placement_candidates=('slot','wrong_slot')), SlotEvidence(finite_status='REJECTED',placement_control_status='MISSING'), 'AMBIGUOUS','BLOCKED')
    return tuple(cases)


def nkt2_claim_ledger() -> tuple[NoveltyClaim, ...]:
    """NKT-2 targeted reassessment of the NKT-1 survivor claims.

    NKT-2 is deliberately conservative.  It does not remove capabilities;
    it only changes which capabilities are defensible as standalone research
    novelty after targeted prior-art attacks.
    """
    base={c.id:c for c in default_claim_ledger()}
    revised=[]
    for cid in ('C1','C2','C3','C4'):
        revised.append(base[cid])
    revised.extend((
        NoveltyClaim(
            'C5',
            'Use explicit knowledge level plus observation/data qualification to decide which architecture operations are admissible',
            NoveltyStatus.PARTIAL,
            (
                'classical grey-box modelling taxonomies based on prior physical knowledge',
                'physics-informed grey-box identification from partial observations',
                'hybrid-model identifiability literature',
            ),
            'Keep as the M2N orchestration policy, but do not present the generic idea of adapting model structure to prior knowledge/observability as new. Any research claim must be narrowed to the explicit contract-to-operation policy and evaluated against strong baselines.',
            'NKT-2 did not find an exact implementation of the full M2N contract-to-operation state machine, but the conceptual ingredients are established prior art.',
        ),
        NoveltyClaim(
            'C6',
            'Slot-wise K0/K1/K2/K3 adjudication with finite physics before local neuralization',
            NoveltyStatus.INFRASTRUCTURE,
            (
                'grey-box / hybrid semi-parametric modelling',
                'simulation-based model inference (SBMI) for discrete scientific model components and parameters',
                'UDE/APHYNITY for learned missing physics',
            ),
            'Keep K0/K1/K2/K3 as a useful practitioner taxonomy and software policy. Do not rely on the taxonomy or the finite-before-neural sequence as a primary standalone novelty claim.',
            'SBMI substantially weakens the discrete-component/model-set part; UDE/APHYNITY cover the learned-physics end of the ladder.',
        ),
        NoveltyClaim(
            'C7',
            'Return SELECTED / finite set / AMBIGUOUS / UNSUPPORTED instead of forcing a single model',
            NoveltyStatus.KILLED,
            (
                'simulation-based model inference (SBMI) returns multiple data-consistent model configurations and exposes non-identifiability',
                'partial/set identification literature',
                'abstention/selective prediction literature',
            ),
            'Keep this behavior because it is scientifically important, but treat finite-set/ambiguity output as adopted methodology rather than M2N novelty.',
            'NKT-2 found direct scientific-model prior art for multiple data-consistent configurations and non-identifiable components.',
        ),
        NoveltyClaim(
            'C8',
            'Require equal-capacity wrong-placement controls before attributing a local K3 placement',
            NoveltyStatus.PARTIAL,
            (
                'hybrid-model methods that select which model states/dynamics receive neural correction',
                'automatic structure-aware sparsification/state selection in hybrid neural ODEs',
            ),
            'Keep equal-capacity wrong-placement controls as a strong falsification protocol. Do not claim neural-correction placement selection itself as new.',
            'The exact equal-capacity falsification rule was not found in the targeted set, but correction-location/state selection is prior art.',
        ),
        base['C9'],
    ))
    return tuple(revised)
