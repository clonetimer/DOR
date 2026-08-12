from __future__ import annotations
from dataclasses import dataclass,asdict

ARCHITECTURE_STATUS={'READY','AMBIGUOUS','UNSUPPORTED'}
PREDICTION_STATUS={'SUPPORTED','PARTIAL','UNSUPPORTED','NOT_EVALUATED'}
SEMANTIC_STATUS={'PHYSICAL_SUPPORTED','NORMALIZED_LATENT_ONLY','UNSUPPORTED','NOT_EVALUATED'}
PLACEMENT_STATUS={'SUPPORTED','AMBIGUOUS','NOT_APPLICABLE','UNSUPPORTED','NOT_EVALUATED'}
K3_STATUS={'OFF','CANDIDATE','SUPPORTED','REJECTED','NOT_EVALUATED'}
PLACEMENT_BASIS={'MECHANISM','DATA','MECHANISM_AND_DATA','NOT_APPLICABLE','NOT_EVALUATED'}


@dataclass(frozen=True)
class ArchitectureDecision:
    """Multi-axis M2N decision contract.

    A single READY/AMBIGUOUS bit is too coarse once an architecture can be
    predictively admissible while its hidden coordinates remain gauge-ambiguous.
    The axes below prevent predictive adequacy from being silently promoted into
    physical semantics or K3 placement claims.
    """
    architecture_status: str
    prediction_status: str
    semantic_status: str
    placement_status: str
    k3_status: str
    placement_basis: str='NOT_EVALUATED'
    reasons: tuple[str,...]=()

    def validate(self)->'ArchitectureDecision':
        checks=((self.architecture_status,ARCHITECTURE_STATUS,'architecture_status'),
                (self.prediction_status,PREDICTION_STATUS,'prediction_status'),
                (self.semantic_status,SEMANTIC_STATUS,'semantic_status'),
                (self.placement_status,PLACEMENT_STATUS,'placement_status'),
                (self.k3_status,K3_STATUS,'k3_status'),
                (self.placement_basis,PLACEMENT_BASIS,'placement_basis'))
        for value,allowed,name in checks:
            if value not in allowed: raise ValueError(f'{name}: {value}')
        if self.placement_status=='SUPPORTED' and self.placement_basis in ('NOT_EVALUATED','NOT_APPLICABLE'):
            raise ValueError('supported placement requires mechanism/data basis')
        if self.k3_status=='SUPPORTED' and self.placement_status not in ('SUPPORTED','NOT_APPLICABLE'):
            raise ValueError('supported K3 requires supported/not-applicable placement')
        if self.semantic_status=='PHYSICAL_SUPPORTED' and self.architecture_status!='READY':
            raise ValueError('physical semantics require READY architecture')
        return self

    def to_dict(self):
        self.validate(); return asdict(self)
