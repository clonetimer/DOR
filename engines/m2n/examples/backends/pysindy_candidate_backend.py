"""PySINDy 2.1 candidate-discovery backend.

Order is fixed:
  M2N qualification -> PySINDy proposal -> proposal admission -> K2 validation.
External discovery never bypasses M2N qualification and its expressions are not
physical truth until explicitly admitted.
"""
import numpy as np
from m2n import (
    Dim, KnowledgeLevel, MechanismSlot, SlotEvidence,
    gate_candidate_discovery, DynamicsDiscoveryRequest,
    PySINDyCandidateBackend, BackendUnavailableError,
    ProposalAdmissionCheck, admit_candidate_proposal,
)

D=Dim()
slot=MechanismSlot(
    id='unknown_local_dynamics', level=KnowledgeLevel.K2, role='candidate expansion demo',
    input_dims=(D,), output_dim=D, finite_candidates=('linear','quadratic'),
)
evidence=SlotEvidence(
    data_qualification='READY', observation_qualification='SUPPORTED',
    hard_mechanism_status='COMPLETE', finite_status='PENDING',
)
gate=gate_candidate_discovery(slot,evidence)
print('discovery gate:',gate.status,'-',gate.reason)

t=np.linspace(0,4,801)
x=np.column_stack([np.exp(-t),0.5*np.exp(-1.7*t)])
xd=np.column_stack([-np.exp(-t),-0.85*np.exp(-1.7*t)])
req=DynamicsDiscoveryRequest(x=x,x_dot=xd,t=t,feature_names=('x','y'),polynomial_degree=2,threshold=0.05,alpha=1e-8)

if gate.status=='ALLOWED':
    try:
        batch=PySINDyCandidateBackend.propose(req)
        for p in batch.proposals:
            print(p.target,'=',p.expression,'[UNADMITTED]')
        # Demonstration only: admit the first proposal after explicit checks.
        p=batch.proposals[0]
        admitted=admit_candidate_proposal(p, ProposalAdmissionCheck(
            proposal_id=p.id,
            family_id='linear_decay_x',
            physical_semantics_verified=True,
            units_verified=True,
            support_verified=True,
            executable_binding_verified=True,
            rationale='demo system is dimensionless and the linear decay binding is explicitly verified',
        ))
        print('admitted:', admitted.family_id, admitted.status)
        print(batch.warning)
    except BackendUnavailableError as e:
        print(e)
        print("Install full PySINDy dependencies, e.g. pip install 'm2n-toolkit[discovery]'.")
