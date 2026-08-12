from isgn.atlas import MechanismAtlas, MechanismScopeRecord, ScopeInterface, UncoveredRegion


def test_atlas_routes_certified_scopes_and_preserves_gaps():
    a=MechanismAtlas('demo')
    a.add_scope(MechanismScopeRecord('mechanical',['x','v'],['pA','pB'],graph={'x':['v'],'v':['pA','pB']},completion_status='COMPLETE',phm_coverage={'faults':['friction']}))
    a.add_scope(MechanismScopeRecord('hydraulic',['pA','pB'],['q'],completion_status='UNCERTIFIED'))
    a.add_uncovered(UncoveredRegion('sensor-offset',['x_sensor'],'OBSERVATION_HEALTH','constant offsets cancel in differential scope'))
    r=a.route_variables(['v','pA','x_sensor'])
    assert [m['scope_id'] for m in r['matches']]==['mechanical']
    assert 'x_sensor' in r['uncovered']
    assert r['route_hints'][0]['route']=='OBSERVATION_HEALTH'


def test_atlas_interface_requires_known_scopes():
    a=MechanismAtlas('demo')
    a.add_scope(MechanismScopeRecord('a',['x']))
    try:
        a.add_interface(ScopeInterface('a','missing',['x']))
    except KeyError:
        pass
    else:
        raise AssertionError('expected KeyError')
