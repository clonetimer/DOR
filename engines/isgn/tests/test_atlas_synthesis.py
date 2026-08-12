from isgn.atlas_synthesis import VariableEvidence, RelationEvidence, synthesize_mechanism_atlas


def test_dynamic_synthesis_keeps_known_state_chain_internal_and_observed_forces_as_drivers():
    vars=[
        VariableEvidence('position','state','READY'), VariableEvidence('velocity','state','CONDITIONAL'),
        VariableEvidence('pressure_a','driver','READY'), VariableEvidence('pressure_b','driver','READY'),
        VariableEvidence('external_load','driver','READY'), VariableEvidence('temperature','state','BLOCKED'),
    ]
    rels=[
        RelationEvidence('r-pos','position',['velocity'],origin='known',bind_states=True),
        RelationEvidence('r-vel','velocity',['velocity','pressure_a','pressure_b','external_load'],origin='consensus'),
    ]
    result=synthesize_mechanism_atlas(system_id='demo',variables=vars,relations=rels)
    assert result.status=='PARTIAL_ATLAS_PROPOSED'
    dyn=[s for s in result.atlas.scopes.values() if s.scope_kind=='dynamic']
    assert len(dyn)==1
    s=dyn[0]
    assert s.target_states==['position','velocity']
    assert set(s.observed_drivers)=={'pressure_a','pressure_b','external_load'}
    assert s.completion_status=='UNCERTIFIED'
    assert any(r.variables==['temperature'] and r.route=='OBSERVATION_REMEDIATION' for r in result.atlas.uncovered_regions)


def test_algebraic_synthesis_builds_interfaces_without_claiming_certification():
    vars=[VariableEvidence(n,'algebraic','READY') for n in ['a','b','c','d']]
    rels=[RelationEvidence('ab','c',['a','b'],kind='algebraic',origin='consensus',terms=['a*b'],coefficients={'a*b':1.0}),
          RelationEvidence('cd','d',['c'],kind='algebraic',origin='consensus',terms=['c'],coefficients={'c':0.5})]
    result=synthesize_mechanism_atlas(system_id='alg',variables=vars,relations=rels)
    assert len(result.atlas.scopes)==2
    assert all(s.completion_status=='UNCERTIFIED' for s in result.atlas.scopes.values())
    assert any(i.variables==['c'] for i in result.atlas.interfaces)


def test_backend_disagreement_does_not_fabricate_scope():
    vars=[VariableEvidence('i','state','READY'),VariableEvidence('w','state','READY')]
    rels=[RelationEvidence('native-i','i',['i'],origin='discovered',confidence=.9,stable=False),
          RelationEvidence('other-i','i',['i','w'],origin='discovered',confidence=.9,stable=False)]
    result=synthesize_mechanism_atlas(system_id='motor',variables=vars,relations=rels)
    assert result.status=='NO_VIABLE_SCOPE'
    assert not result.atlas.scopes
    assert {r.route for r in result.atlas.uncovered_regions}=={'STABLE_RELATION_REQUIRED'}


def test_prediscovery_known_chain_proposes_scope_without_fabricating_missing_relation():
    vars=[
        VariableEvidence('position','state','READY'), VariableEvidence('velocity','state','READY'),
        VariableEvidence('pressure_a','state','READY'), VariableEvidence('pressure_b','state','READY'),
        VariableEvidence('external_load','input','READY'), VariableEvidence('temperature','state','BLOCKED'),
    ]
    # Only kinematics is known.  No velocity derivative row has been discovered.
    rels=[RelationEvidence('known-kinematics','position',['velocity'],origin='known',bind_states=True)]
    result=synthesize_mechanism_atlas(system_id='prediscovery',variables=vars,relations=rels)
    dyn=[s for s in result.atlas.scopes.values() if s.scope_kind=='dynamic']
    assert len(dyn)==1
    scope=dyn[0]
    assert scope.target_states==['position','velocity']
    assert scope.graph=={'position':['velocity']}
    assert scope.metadata['discovery_required_targets']==['velocity']
    assert set(scope.metadata['candidate_driver_pool'])=={'pressure_a','pressure_b','external_load'}
    assert scope.observation_status=='SYNTHESIS_PLANNED'
    assert 'velocity' not in scope.graph  # missing physics remains explicit
