from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence
import importlib.util
import importlib
import numpy as np

from .knowledge_contract import KnowledgeLevel, MechanismSlot, SlotEvidence


class BackendUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class DiscoveryGateDecision:
    status: str  # ALLOWED | AMBIGUOUS | UNSUPPORTED | NOT_APPLICABLE
    slot_id: str
    reason: str
    proposal_semantics: str = 'UNADMITTED_K2_PROPOSAL'


@dataclass(frozen=True)
class DynamicsDiscoveryRequest:
    x: np.ndarray
    t: float | np.ndarray
    feature_names: tuple[str, ...]
    x_dot: np.ndarray | None = None
    u: np.ndarray | None = None
    polynomial_degree: int = 3
    threshold: float = 0.05
    alpha: float = 1e-6
    include_bias: bool = True

    def validate(self) -> 'DynamicsDiscoveryRequest':
        x=np.asarray(self.x,dtype=float)
        if x.ndim != 2 or x.shape[0] < 5:
            raise ValueError('x must have shape (samples, states) with at least 5 samples')
        if len(self.feature_names) != x.shape[1]:
            raise ValueError('feature_names must match state columns')
        if not np.all(np.isfinite(x)):
            raise ValueError('x contains non-finite values')
        if self.x_dot is not None:
            xd=np.asarray(self.x_dot,dtype=float)
            if xd.shape != x.shape or not np.all(np.isfinite(xd)):
                raise ValueError('x_dot must be finite and match x shape')
        if self.u is not None:
            u=np.asarray(self.u,dtype=float)
            if u.shape[0] != x.shape[0] or not np.all(np.isfinite(u)):
                raise ValueError('u must be finite and share sample axis with x')
        if self.polynomial_degree < 1:
            raise ValueError('polynomial_degree must be >=1')
        if self.threshold < 0 or self.alpha < 0:
            raise ValueError('threshold/alpha must be non-negative')
        return self


@dataclass(frozen=True)
class CandidateProposal:
    id: str
    expression: str
    target: str
    backend: str
    complexity: int | None = None
    metadata: tuple[tuple[str,str], ...] = ()

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CandidateProposalBatch:
    backend: str
    proposals: tuple[CandidateProposal, ...]
    warning: str

    def as_dict(self):
        return {'backend':self.backend,'proposals':[p.as_dict() for p in self.proposals],'warning':self.warning}


@dataclass(frozen=True)
class ProposalAdmissionCheck:
    proposal_id: str
    family_id: str
    physical_semantics_verified: bool
    units_verified: bool
    support_verified: bool
    executable_binding_verified: bool
    rationale: str

    @property
    def admitted(self) -> bool:
        return all((self.physical_semantics_verified, self.units_verified, self.support_verified, self.executable_binding_verified))


@dataclass(frozen=True)
class AdmittedCandidateProposal:
    proposal: CandidateProposal
    family_id: str
    rationale: str
    status: str = 'ADMITTED_K2_CANDIDATE'

    def as_dict(self):
        return {'proposal': self.proposal.as_dict(), 'family_id': self.family_id, 'rationale': self.rationale, 'status': self.status}


def admit_candidate_proposal(proposal: CandidateProposal, check: ProposalAdmissionCheck) -> AdmittedCandidateProposal:
    if proposal.id != check.proposal_id:
        raise ValueError('proposal_id mismatch')
    if not check.admitted:
        failed=[]
        for name in ('physical_semantics_verified','units_verified','support_verified','executable_binding_verified'):
            if not getattr(check,name): failed.append(name)
        raise ValueError('proposal remains UNADMITTED; failed checks: ' + ', '.join(failed))
    if not check.family_id.strip():
        raise ValueError('family_id must be non-empty')
    if not check.rationale.strip():
        raise ValueError('rationale must explain why the proposal is physically admissible')
    return AdmittedCandidateProposal(proposal, check.family_id, check.rationale)


def gate_candidate_discovery(slot: MechanismSlot, evidence: SlotEvidence) -> DiscoveryGateDecision:
    """Gate external candidate discovery before any backend is called.

    Discovery is not an escape from bad sensing or missing hard mechanism.  It
    may expand a K2 family or attempt a finite symbolic proposal before K3, but
    every returned expression remains an *unadmitted* proposal until an engineer
    gives it a physical/semantic interpretation and an executable M2N binding.
    """
    slot.validate(); evidence.validate()
    level=slot.normalized_level()
    if evidence.hard_mechanism_status != 'COMPLETE':
        return DiscoveryGateDecision('AMBIGUOUS',slot.id,'known hard mechanism is incomplete/unresolved')
    if evidence.data_qualification == 'UNSUPPORTED' or evidence.observation_qualification == 'UNSUPPORTED':
        return DiscoveryGateDecision('UNSUPPORTED',slot.id,'data/observation contract does not support discovery for this slot')
    if evidence.data_qualification == 'AMBIGUOUS' or evidence.observation_qualification == 'PARTIAL':
        return DiscoveryGateDecision('AMBIGUOUS',slot.id,'qualification is insufficient; do not use discovery as a workaround')
    if level in (KnowledgeLevel.K0, KnowledgeLevel.K1):
        return DiscoveryGateDecision('NOT_APPLICABLE',slot.id,'trusted K0/K1 structure must not be replaced by data discovery')
    return DiscoveryGateDecision('ALLOWED',slot.id,'qualified local K2/K3 slot may request finite candidate proposals')


class PySINDyCandidateBackend:
    """Optional PySINDy adapter; PySINDy is not vendored by M2N."""
    backend_id='pysindy'

    @staticmethod
    def runtime_probe() -> tuple[bool, str | None]:
        if importlib.util.find_spec('pysindy') is None:
            return False, 'pysindy package is not installed'
        try:
            importlib.import_module('pysindy')
            return True, None
        except Exception as e:
            return False, f'{type(e).__name__}: {e}'

    @classmethod
    def available(cls) -> bool:
        ok,_=cls.runtime_probe()
        return ok

    @classmethod
    def propose(cls, request: DynamicsDiscoveryRequest) -> CandidateProposalBatch:
        request.validate()
        ok,reason=cls.runtime_probe()
        if not ok:
            raise BackendUnavailableError(
                "PySINDy runtime is not ready. Install the optional dependency with "
                "`pip install 'm2n-toolkit[discovery]'` or install PySINDy 2.1 plus its dependencies. "
                f"Runtime probe: {reason}"
            )
        import pysindy as ps
        lib=ps.PolynomialLibrary(degree=request.polynomial_degree,include_bias=request.include_bias)
        opt=ps.STLSQ(threshold=request.threshold,alpha=request.alpha)
        model=ps.SINDy(feature_library=lib,optimizer=opt)
        kwargs={'t':request.t,'feature_names':list(request.feature_names)}
        if request.x_dot is not None: kwargs['x_dot']=request.x_dot
        if request.u is not None: kwargs['u']=request.u
        model.fit(request.x,**kwargs)
        equations=tuple(str(s) for s in model.equations())
        try: complexity=int(model.complexity)
        except Exception: complexity=None
        proposals=[]
        for i,expr in enumerate(equations):
            target=f'd({request.feature_names[i]})/dt'
            proposals.append(CandidateProposal(
                id=f'pysindy_eq_{i}',expression=expr,target=target,backend=cls.backend_id,
                complexity=complexity,metadata=(('status','UNADMITTED'),)
            ))
        return CandidateProposalBatch(
            backend=cls.backend_id, proposals=tuple(proposals),
            warning='PySINDy output is a proposal only. Do not treat it as physical truth; translate/admit it into a finite M2N K2 family and rerun normal-data qualification.'
        )
