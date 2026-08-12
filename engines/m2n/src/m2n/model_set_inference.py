from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import reduce
from operator import mul
from pathlib import Path
import json


@dataclass(frozen=True)
class ModelComponentSlot:
    slot_id: str
    candidates: tuple[str, ...]

    def validate(self):
        if not self.slot_id.strip(): raise ValueError('slot_id must be non-empty')
        if len(self.candidates) < 2: raise ValueError('model-component slot requires at least two candidates')
        if len(set(self.candidates)) != len(self.candidates): raise ValueError('candidate ids must be unique within a slot')
        return self


@dataclass(frozen=True)
class ParameterPrior:
    name: str
    low: float
    high: float
    distribution: str='uniform'

    def validate(self):
        if not self.name.strip(): raise ValueError('parameter prior name must be non-empty')
        if not self.low < self.high: raise ValueError('parameter prior requires low < high')
        if self.distribution not in ('uniform','log_uniform'):
            raise ValueError('v1.3 contract supports uniform/log_uniform parameter priors')
        if self.distribution=='log_uniform' and self.low <= 0:
            raise ValueError('log_uniform requires positive bounds')
        return self


@dataclass(frozen=True)
class SBMIModelSetRequest:
    slots: tuple[ModelComponentSlot, ...]
    parameter_priors: tuple[ParameterPrior, ...] = ()
    simulator_available: bool = True
    randomized_component_simulation: bool = True
    observation_summary_defined: bool = True
    simulation_budget: int | None = None
    exact_enumeration_limit: int = 32

    def validate(self):
        if not self.slots: raise ValueError('at least one model-component slot is required')
        for x in self.slots: x.validate()
        if len({x.slot_id for x in self.slots}) != len(self.slots): raise ValueError('slot ids must be unique')
        for p in self.parameter_priors: p.validate()
        if self.exact_enumeration_limit < 1: raise ValueError('exact_enumeration_limit must be positive')
        if self.simulation_budget is not None and self.simulation_budget < 1: raise ValueError('simulation_budget must be positive')
        return self

    @property
    def configuration_count(self) -> int:
        return reduce(mul,(len(s.candidates) for s in self.slots),1)


@dataclass(frozen=True)
class SBMIFeasibilityAssessment:
    status: str  # NOT_NEEDED | FEASIBLE_CONTRACT | BLOCKED
    configuration_count: int
    recommended_mode: str
    reason: str
    exact_enumeration_limit: int
    important_boundary: str

    def as_dict(self): return asdict(self)


def assess_sbmi_feasibility(request: SBMIModelSetRequest) -> SBMIFeasibilityAssessment:
    request.validate(); n=request.configuration_count
    boundary=('SBMI is simulation-based joint inference over model components and parameters; '
              'M2N categorical slots require an exactly-one-candidate-per-slot encoding/admission layer. '
              'This assessment is an engineering integration heuristic, not a claim from the SBMI paper.')
    if not request.simulator_available:
        return SBMIFeasibilityAssessment('BLOCKED',n,'NONE','SBMI requires a simulator from which component/parameter configurations can be sampled.',request.exact_enumeration_limit,boundary)
    if not request.randomized_component_simulation:
        return SBMIFeasibilityAssessment('BLOCKED',n,'NONE','the simulator cannot currently instantiate randomized component configurations required for simulation-based model inference.',request.exact_enumeration_limit,boundary)
    if not request.observation_summary_defined:
        return SBMIFeasibilityAssessment('BLOCKED',n,'NONE','no observation/summary contract is defined for training and conditioning the inference backend.',request.exact_enumeration_limit,boundary)
    if n <= request.exact_enumeration_limit:
        return SBMIFeasibilityAssessment('NOT_NEEDED',n,'M2N_EXACT_ENUMERATION',f'only {n} finite model configurations exist; exact M2N validation/portfolio is simpler and more transparent than amortized SBMI.',request.exact_enumeration_limit,boundary)
    return SBMIFeasibilityAssessment('FEASIBLE_CONTRACT',n,'EXTERNAL_SBMI_OR_SUCCESSOR',f'{n} configurations exceed the exact-enumeration threshold; a simulation-based model-component posterior could reduce exhaustive model evaluation if simulation budget is adequate.',request.exact_enumeration_limit,boundary)


def export_sbmi_contract(request: SBMIModelSetRequest, output_json: str | Path) -> dict:
    assessment=assess_sbmi_feasibility(request)
    payload={
        'backend_family':'simulation_based_model_inference',
        'status':assessment.status,
        'recommended_mode':assessment.recommended_mode,
        'configuration_count':assessment.configuration_count,
        'component_encoding':{
            'type':'one_hot_binary_mask_per_m2n_slot',
            'constraint':'exactly one candidate is active in each M2N finite-candidate slot',
            'note':'Original SBMI represents distributions over binary model components; this M2N contract adds categorical-slot constraints and therefore requires an adapter-specific encoding layer.',
        },
        'slots':[asdict(s) for s in request.slots],
        'parameter_priors':[asdict(p) for p in request.parameter_priors],
        'simulator_contract':{
            'available':request.simulator_available,
            'randomized_component_simulation':request.randomized_component_simulation,
            'observation_summary_defined':request.observation_summary_defined,
            'simulation_budget':request.simulation_budget,
        },
        'admission_policy':[
            'posterior model configurations remain M2N finite-model evidence, not physical truth',
            'report multimodality/non-identifiability rather than forcing a unique architecture',
            'held-out normal validation remains required before deployment',
            'do not use SBMI to bypass observation/data qualification',
        ],
        'boundary':assessment.important_boundary,
    }
    p=Path(output_json); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    return payload
