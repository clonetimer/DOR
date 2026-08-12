import numpy as np
import pytest

from m2n import (
    Dim, KnowledgeLevel, MechanismSlot, SlotEvidence,
    list_backends, gate_candidate_discovery,
    DynamicsDiscoveryRequest, PySINDyCandidateBackend, BackendUnavailableError,
    default_claim_ledger, run_behavioral_nkt,
)


def test_backend_registry_contains_pysindy_and_contract_only_backends():
    ids={b.id:b for b in list_backends()}
    assert ids['pysindy'].integration_status=='OPTIONAL_ADAPTER'
    assert ids['pysindy'].code_license=='MIT'
    assert ids['symmatika'].integration_status=='EXTERNAL_ADAPTER'
    assert ids['ude'].integration_status=='CONTRACT_ONLY'
    assert ids['sbmi'].integration_status=='CONTRACT_ONLY'
    assert 'MODEL_COMPONENT_INFERENCE' in [x.value for x in ids['sbmi'].capability]


def test_discovery_gate_cannot_bypass_qualification():
    d=Dim()
    k1=MechanismSlot('k1',KnowledgeLevel.K1,'r',(d,),d,exact_family='gain',learnable_parameters=('g',))
    assert gate_candidate_discovery(k1,SlotEvidence()).status=='NOT_APPLICABLE'
    k2=MechanismSlot('k2',KnowledgeLevel.K2,'r',(d,),d,finite_candidates=('a','b'))
    assert gate_candidate_discovery(k2,SlotEvidence(finite_status='PENDING')).status=='ALLOWED'
    assert gate_candidate_discovery(k2,SlotEvidence(data_qualification='AMBIGUOUS',finite_status='PENDING')).status=='AMBIGUOUS'
    assert gate_candidate_discovery(k2,SlotEvidence(observation_qualification='UNSUPPORTED',finite_status='PENDING')).status=='UNSUPPORTED'


def test_pysindy_missing_dependency_is_explicit(monkeypatch):
    import m2n.discovery as discovery
    monkeypatch.setattr(discovery.importlib.util,'find_spec',lambda name: None)
    req=DynamicsDiscoveryRequest(
        x=np.column_stack([np.linspace(0,1,20),np.linspace(1,2,20)]),
        x_dot=np.column_stack([np.ones(20),np.ones(20)]),
        t=0.1,feature_names=('x','y')
    )
    with pytest.raises(BackendUnavailableError):
        PySINDyCandidateBackend.propose(req)


def test_nkt_claim_ledger_and_behavioral_suite():
    claims=default_claim_ledger()
    assert any(c.id=='C1' and 'KILLED' in c.status.value for c in claims)
    assert any(c.id=='C5' and 'NOT_KILLED' in c.status.value for c in claims)
    results=run_behavioral_nkt()
    assert len(results)>=7
    assert all(r.passed for r in results)


def test_proposal_admission_requires_all_checks():
    from m2n import CandidateProposal, ProposalAdmissionCheck, admit_candidate_proposal
    p=CandidateProposal('p1','-1.0 x','d(x)/dt','pysindy')
    bad=ProposalAdmissionCheck(
        proposal_id='p1', family_id='linear_decay',
        physical_semantics_verified=True, units_verified=True,
        support_verified=False, executable_binding_verified=True,
        rationale='candidate has correct dimensions but support is not established',
    )
    with pytest.raises(ValueError, match='UNADMITTED'):
        admit_candidate_proposal(p,bad)
    good=ProposalAdmissionCheck(
        proposal_id='p1', family_id='linear_decay',
        physical_semantics_verified=True, units_verified=True,
        support_verified=True, executable_binding_verified=True,
        rationale='linear decay is physically admissible and has a verified executable binding',
    )
    a=admit_candidate_proposal(p,good)
    assert a.status=='ADMITTED_K2_CANDIDATE'
    assert a.family_id=='linear_decay'


def test_nkt2_downgrades_survivor_claims():
    from m2n import nkt2_claim_ledger
    c={x.id:x for x in nkt2_claim_ledger()}
    assert c['C5'].status.value.startswith('PARTIALLY')
    assert c['C6'].status.value.startswith('USEFUL_INFRASTRUCTURE')
    assert c['C7'].status.value.startswith('KILLED')
    assert c['C8'].status.value.startswith('PARTIALLY')


def test_symmatika_job_adapter_prepare_parse_and_fake_runtime(tmp_path):
    from m2n import (
        SymMatikaExplicitRequest, prepare_symmatika_job, parse_symmatika_driver_output,
        run_symmatika_from_source, probe_symmatika_source,
    )
    x=np.linspace(-1,1,40)[:,None]
    y=x[:,0]**2
    req=SymMatikaExplicitRequest(x=x,y=y,feature_names=('x',),target_name='y',population_size=100,max_depth=2)
    job=prepare_symmatika_job(req,tmp_path/'job')
    data=np.loadtxt(job.dataset_csv,delimiter=',')
    assert data.shape==(40,2)
    assert 'return_final_candidates' in open(job.driver_cpp,encoding='utf-8').read()
    parsed=parse_symmatika_driver_output('noise\nM2N_SYMMATIKA_BEGIN\nM2N_SYMMATIKA_CANDIDATE\t0.01\tx*x\nM2N_SYMMATIKA_END\n','y')
    assert parsed.proposals[0].expression=='x*x'
    assert parsed.proposals[0].metadata[0]==('status','UNADMITTED')

    # Adapter-harness test: compile against a tiny API-compatible fake source tree.
    # This validates M2N's copy/build/run/parser mechanics, not the upstream SymMatika engine.
    src=tmp_path/'fake_symmatika'; src.mkdir()
    (src/'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.15)\nproject(fake LANGUAGES CXX)\nset(CMAKE_CXX_STANDARD 17)\nadd_executable(symmatika_fake main.cpp)\n',encoding='utf-8')
    (src/'main.cpp').write_text('int main(){return 0;}\n',encoding='utf-8')
    (src/'model.h').write_text(r'''#pragma once
#include <string>
#include <vector>
#include <utility>
struct DataSet { DataSet(std::string,bool,std::vector<std::string>){} };
struct Model {
 Model(const DataSet&,int,int,std::vector<bool>,bool,bool){}
 void run(){}
 std::vector<std::pair<std::string,double>> return_final_candidates() const {return {{"x*x",0.01},{"x",0.2}};}
};
''',encoding='utf-8')
    assert probe_symmatika_source(src).ready
    batch=run_symmatika_from_source(req,src,tmp_path/'runtime',timeout_seconds=30)
    assert [p.expression for p in batch.proposals]==['x*x','x']


def test_sbmi_feasibility_prefers_exact_small_sets_and_contracts_large_sets(tmp_path):
    from m2n import ModelComponentSlot, ParameterPrior, SBMIModelSetRequest, assess_sbmi_feasibility, export_sbmi_contract
    small=SBMIModelSetRequest(slots=(
        ModelComponentSlot('pressure',('direct','line_drop')),
        ModelComponentSlot('friction',('viscous','coulomb','stribeck')),
    ))
    a=assess_sbmi_feasibility(small)
    assert a.configuration_count==6
    assert a.status=='NOT_NEEDED'
    assert a.recommended_mode=='M2N_EXACT_ENUMERATION'

    large=SBMIModelSetRequest(
        slots=tuple(ModelComponentSlot(f's{i}',('a','b','c')) for i in range(4)),
        parameter_priors=(ParameterPrior('gain',0.1,10.0,'log_uniform'),),
        simulation_budget=10000,
    )
    b=assess_sbmi_feasibility(large)
    assert b.configuration_count==81
    assert b.status=='FEASIBLE_CONTRACT'
    payload=export_sbmi_contract(large,tmp_path/'sbmi.json')
    assert payload['component_encoding']['type']=='one_hot_binary_mask_per_m2n_slot'
    assert payload['status']=='FEASIBLE_CONTRACT'

    blocked=SBMIModelSetRequest(slots=(ModelComponentSlot('x',('a','b')),),simulator_available=False)
    assert assess_sbmi_feasibility(blocked).status=='BLOCKED'
