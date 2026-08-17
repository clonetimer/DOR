from dor.isgn_active_discovery_bridge import build_active_discovery_advisory


def test_active_discovery_posterior_is_advisory_not_diagnosis():
    x=build_active_discovery_advisory({
        'structure_posterior':{'A':0.8,'B':0.2},
        'truth_quarantine':{'held_out_truth_loaded':False,'true_structure_loaded':False,'truth_channels_allowed':False},
        'next_experiment':{'experiment_id':'probe-2'}
    })
    assert x['status']=='ADVISORY_ONLY'
    assert x['candidate_structures'][0]['structure_id']=='A'
    assert x['safety']['diagnostic_claim_allowed'] is False
    assert x['safety']['root_cause_claim_allowed'] is False
    assert x['safety']['mechanism_identity_certified'] is False
    assert x['safety']['proposal_is_evidence'] is False


def test_active_discovery_with_preloaded_truth_is_blocked():
    x=build_active_discovery_advisory({'truth_quarantine':{'held_out_truth_loaded':True}})
    assert x['status']=='BLOCKED'


def test_active_discovery_identifiability_is_exposed_but_never_certified():
    x=build_active_discovery_advisory({
        'structure_posterior':{'A':0.99,'B':0.01},
        'identifiability':{'status':'UNIQUE_SUPPORTED','remaining_discriminating_experiments':[]},
        'evidence_stability':{'status':'STABLE'},
        'truth_quarantine':{'held_out_truth_loaded':False,'true_structure_loaded':False,'truth_channels_allowed':False},
        'hypothesis_space_coverage_complete':True,
    })
    assert x['identifiability_status']=='UNIQUE_SUPPORTED'
    assert x['evidence_stability_status']=='STABLE'
    assert x['safety']['mechanism_identity_certified'] is False
    assert x['safety']['fault_identification_allowed'] is False


def test_missing_quarantine_and_malformed_posterior_fail_closed():
    x=build_active_discovery_advisory({'structure_posterior':{'A':0.8,'B':-0.1}})
    assert x['status']=='BLOCKED'
    assert x['safety']['truth_quarantine_verified'] is False
    assert x['admission_issues']


def test_live_payload_is_blocked_and_never_forwarded():
    x=build_active_discovery_advisory({
        'structure_posterior':{'A':1.0},
        'truth_quarantine':{'held_out_truth_loaded':False,'true_structure_loaded':False,'truth_channels_allowed':False},
        'next_experiment':{
            'experiment_id':'burn-1','execution_mode':'LIVE','repeats':1,'cost':2.0,
            'safety_eligible':True,'payload':{'actuator':'thruster','command':'fire'},
        },
        'predictive_adequacy':{
            'status':'ADEQUATE','truth_channels_allowed':False,'min_mahalanobis':0.5,
            'hidden_truth':'must-not-cross',
        },
    })
    assert x['status']=='BLOCKED'
    assert 'payload' not in x['next_experiment'] and 'command' not in x['next_experiment']
    assert x['next_experiment']['execution_allowed'] is False
    assert x['safety']['raw_experiment_payload_forwarded'] is False
    assert 'hidden_truth' not in x['predictive_adequacy']


def test_unique_status_requires_stability_and_coverage():
    base={
        'structure_posterior':{'A':1.0},
        'truth_quarantine':{'held_out_truth_loaded':False,'true_structure_loaded':False,'truth_channels_allowed':False},
        'identifiability':{'status':'UNIQUE_SUPPORTED'},
    }
    x=build_active_discovery_advisory(base)
    assert x['status']=='BLOCKED'
    assert any('STABLE' in issue for issue in x['admission_issues'])
    assert any('coverage' in issue for issue in x['admission_issues'])


def test_advisory_matches_json_schema():
    import json
    from pathlib import Path
    import pytest
    jsonschema=pytest.importorskip('jsonschema')
    x=build_active_discovery_advisory({
        'structure_posterior':{'A':0.8,'B':0.2},
        'truth_quarantine':{'held_out_truth_loaded':False,'true_structure_loaded':False,'truth_channels_allowed':False},
        'next_experiment':{'experiment_id':'probe-2'},
    })
    schema=json.loads((Path(__file__).parents[1]/'schemas'/'isgn_active_discovery_advisory_1.0.schema.json').read_text())
    jsonschema.validate(x,schema)
