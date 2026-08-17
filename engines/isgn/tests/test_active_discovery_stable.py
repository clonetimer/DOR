import numpy as np
import pytest
from isgn.active_discovery import (
    ParameterParticle, ExperimentCandidate, ParticleSimulationBank, ParameterParticleSMC,
    rank_experiments, predictive_adequacy, admit_proposal_particles,
)


def _bank():
    rng=np.random.default_rng(5); samples={}
    mus={"a0":0.0,"a1":0.4,"b0":3.0,"b1":3.4}
    for pid,mu in mus.items():
        for e,shift in [("weak",0.0),("disc",0.0 if pid.startswith("a") else 2.0)]:
            samples[(pid,e,1)] = rng.normal(mu+shift,0.3,size=(24,1))
    return ParticleSimulationBank(samples,covariance_floor=[0.1])


def test_parameter_particles_and_voi_are_structure_aware():
    ps=[ParameterParticle("a0","A",{"k":0.8}),ParameterParticle("a1","A",{"k":1.2}),
        ParameterParticle("b0","B",{"k":0.8}),ParameterParticle("b1","B",{"k":1.2})]
    post=ParameterParticleSMC(ps,seed=2); bank=_bank()
    rows=rank_experiments(post,bank,[ExperimentCandidate("weak"),ExperimentCandidate("disc")],remaining_budget=1)
    assert rows[0].experiment_id=="disc"
    rec=post.update(bank,"disc",np.array([5.2]),resample_threshold=0.0)
    assert rec.structure_posterior["B"]>0.9
    assert "k_mean" in rec.parameter_summary["B"]
    assert rec.truth_channels_allowed is False


def test_repeat_bank_is_fail_closed():
    bank=_bank()
    try:
        bank.summary("a0","weak",2)
    except KeyError:
        pass
    else:
        raise AssertionError("repeat semantics must not assume covariance / r without a repeat-aware bank")


def test_mopen_admission_is_bounded_candidate_mass_not_evidence():
    post=ParameterParticleSMC([ParameterParticle("a0","A",{"k":1.0})])
    expanded=admit_proposal_particles(post,[ParameterParticle("b0","B",{"k":1.0})],proposal_prior_mass=0.2)
    sp=expanded.structure_posterior()
    assert 0.19 < sp["B"] < 0.21 and 0.79 < sp["A"] < 0.81
    rep=predictive_adequacy(expanded,_bank(),"weak",np.array([0.2]))
    assert rep["truth_channels_allowed"] is False


def test_identifiability_refuses_posterior_concentration_when_alternative_is_equivalent():
    from isgn.active_discovery import assess_structure_identifiability
    rng=np.random.default_rng(17)
    ps=[ParameterParticle('a','A'),ParameterParticle('b','B')]
    post=ParameterParticleSMC(ps,weights=[0.99,0.01])
    samples={}
    for pid in ['a','b']:
        for e in ['e1','e2']:
            # same predictive distribution -> structurally unidentifiable despite concentrated posterior
            samples[(pid,e,1)] = rng.normal(0.0,0.3,size=(40,2))
    bank=ParticleSimulationBank(samples,covariance_floor=[0.1,0.1])
    dec=assess_structure_identifiability(post,bank,[ExperimentCandidate('e1'),ExperimentCandidate('e2')],remaining_budget=2,
                                         equivalence_threshold=1.5,discriminating_threshold=2.0)
    assert dec.status=='UNIDENTIFIABLE'
    assert set(dec.alternative_structures)=={'A','B'}
    assert dec.certification_allowed is False


def test_identifiability_returns_alternatives_when_discriminating_experiment_remains():
    from isgn.active_discovery import assess_structure_identifiability
    rng=np.random.default_rng(19)
    ps=[ParameterParticle('a','A'),ParameterParticle('b','B')]
    post=ParameterParticleSMC(ps,weights=[0.55,0.45])
    samples={}
    samples[('a','weak',1)] = rng.normal(0.0,0.5,size=(40,1))
    samples[('b','weak',1)] = rng.normal(0.1,0.5,size=(40,1))
    samples[('a','disc',1)] = rng.normal(0.0,0.3,size=(40,1))
    samples[('b','disc',1)] = rng.normal(3.0,0.3,size=(40,1))
    bank=ParticleSimulationBank(samples,covariance_floor=[0.1])
    dec=assess_structure_identifiability(post,bank,[ExperimentCandidate('weak'),ExperimentCandidate('disc')],remaining_budget=1,
                                         executed_experiment_ids=['weak'])
    assert dec.status=='ALTERNATIVES'
    assert 'disc' in dec.remaining_discriminating_experiments


def test_live_actuation_is_fail_closed_by_default():
    from isgn.active_discovery import ExperimentExecutionPolicy, admit_experiment_candidate
    c=ExperimentCandidate('live_probe',execution_mode='LIVE',safety_eligible=True,safety_evidence_id='SAFE-1')
    a=admit_experiment_candidate(c,remaining_budget=2)
    assert a.eligible is False and a.status=='BLOCKED_LIVE_NOT_AUTHORIZED'
    p=ExperimentExecutionPolicy(allow_live_actuation=True,approved_live_experiment_ids=('live_probe',))
    a2=admit_experiment_candidate(c,remaining_budget=2,policy=p)
    assert a2.eligible is True


def test_live_actuation_requires_safety_evidence_even_when_enabled():
    from isgn.active_discovery import ExperimentExecutionPolicy, admit_experiment_candidate
    c=ExperimentCandidate('live_probe',execution_mode='LIVE',safety_eligible=True)
    p=ExperimentExecutionPolicy(allow_live_actuation=True)
    a=admit_experiment_candidate(c,remaining_budget=2,policy=p)
    assert a.eligible is False and a.status=='BLOCKED_MISSING_SAFETY_EVIDENCE'


def test_mopen_incomplete_hypothesis_coverage_blocks_unique_identification():
    from isgn.active_discovery import assess_structure_identifiability
    rng=np.random.default_rng(23)
    ps=[ParameterParticle('a','A'),ParameterParticle('b','B')]
    post=ParameterParticleSMC(ps,weights=[0.999,0.001])
    samples={}
    for pid,mu in [('a',0.0),('b',3.0)]:
        samples[(pid,'disc',1)] = rng.normal(mu,0.25,size=(40,1))
    bank=ParticleSimulationBank(samples,covariance_floor=[0.1])
    dec=assess_structure_identifiability(
        post,bank,[ExperimentCandidate('disc')],remaining_budget=1,
        hypothesis_space_coverage_complete=False,
        unresolved_candidate_structures=['C','D'],
    )
    assert dec.status=='HYPOTHESIS_CLASS_INADEQUATE'
    assert dec.top_structure=='A'
    assert dec.certification_allowed is False
    assert 'C' in dec.reason and 'D' in dec.reason


def test_evidence_stability_vetoes_online_top_disagreement():
    from isgn.active_discovery import assess_evidence_stability, assess_structure_identifiability
    rng=np.random.default_rng(29)
    ps=[ParameterParticle('a','A'),ParameterParticle('b','B')]
    prior=ParameterParticleSMC(ps,weights=[0.5,0.5])
    # Online posterior is intentionally confidently wrong; static evidence favors B.
    online=ParameterParticleSMC(ps,weights=[0.99,0.01])
    samples={}
    for e in ['e1','e2','e3']:
        samples[('a',e,1)] = rng.normal(0.0,0.2,size=(60,1))
        samples[('b',e,1)] = rng.normal(3.0,0.2,size=(60,1))
    bank=ParticleSimulationBank(samples,covariance_floor=[0.1])
    obs=[(e,np.array([3.0]),1) for e in ['e1','e2','e3']]
    stab=assess_evidence_stability(online,prior,bank,obs,minimum_loo_same_top_fraction=0.8)
    assert stab.status=='UNSTABLE'
    assert stab.current_top_structure=='A' and stab.robust_top_structure=='B'
    dec=assess_structure_identifiability(online,bank,[ExperimentCandidate('e1'),ExperimentCandidate('e2'),ExperimentCandidate('e3')],
        remaining_budget=1,evidence_stability_status=stab.status)
    assert dec.status=='HYPOTHESIS_CLASS_INADEQUATE'


def test_evidence_stability_passes_consistent_history():
    from isgn.active_discovery import assess_evidence_stability
    rng=np.random.default_rng(31)
    ps=[ParameterParticle('a','A'),ParameterParticle('b','B')]
    prior=ParameterParticleSMC(ps,weights=[0.5,0.5])
    online=ParameterParticleSMC(ps,weights=[0.01,0.99])
    samples={}
    for e in ['e1','e2','e3','e4']:
        samples[('a',e,1)] = rng.normal(0.0,0.2,size=(60,1))
        samples[('b',e,1)] = rng.normal(3.0,0.2,size=(60,1))
    bank=ParticleSimulationBank(samples,covariance_floor=[0.1])
    obs=[(e,np.array([3.0]),1) for e in ['e1','e2','e3','e4']]
    stab=assess_evidence_stability(online,prior,bank,obs,minimum_loo_same_top_fraction=0.8)
    assert stab.status=='STABLE' and stab.robust_top_structure=='B'


@pytest.mark.parametrize('kwargs',[
    {'repeats':0}, {'unit_cost':0}, {'unit_cost':float('nan')}, {'execution_mode':'UNKNOWN'},
])
def test_invalid_experiment_contract_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ExperimentCandidate('bad',**kwargs)


def test_invalid_simulation_bank_and_smc_controls_are_rejected():
    with pytest.raises(ValueError):
        ParticleSimulationBank({})
    with pytest.raises(ValueError):
        ParticleSimulationBank({('a','e',1):np.array([[1.0],[2.0],[np.nan]])})
    post=ParameterParticleSMC([ParameterParticle('a','A')])
    with pytest.raises(ValueError):
        post.update(ParticleSimulationBank({('a','e',1):np.array([[0.0],[0.1],[0.2]])}),'e',np.array([0.1]),resample_threshold=1.1)


def test_invalid_proposal_prior_is_rejected_not_clipped():
    post=ParameterParticleSMC([ParameterParticle('a','A')])
    with pytest.raises(ValueError):
        admit_proposal_particles(post,[ParameterParticle('b','B')],proposal_prior_mass=0.9)
