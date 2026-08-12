from isgn.atlas import MechanismAtlas, MechanismScopeRecord, UncoveredRegion
from isgn.evidence_router import HealthEvidenceEvent, route_health_evidence


def _atlas():
    a=MechanismAtlas('demo')
    a.add_scope(MechanismScopeRecord(
        'mechanical',['position','velocity'],observed_drivers=['pressure_a','pressure_b','external_load'],
        graph={'position':['velocity'],'velocity':['velocity','pressure_a','pressure_b','external_load']},
        completion_status='COMPLETE',
        phm_coverage={'evidence_kinds':['parameter_drift','mechanism_residual'],
                      'phenomena':['friction','sensor_delay']}))
    a.add_scope(MechanismScopeRecord('hydraulic',['pressure_a','pressure_b'],observed_drivers=['flow'],completion_status='UNCERTIFIED'))
    a.add_uncovered(UncoveredRegion('position-offset',['position'],'OBSERVATION_HEALTH_OFFSET_SENSITIVE',
                                    'constant offsets cancel in a differential mechanism',
                                    {'phenomena':['sensor_bias']}))
    a.add_uncovered(UncoveredRegion('internal-leakage',['pressure_a','pressure_b'],'HYDRAULIC_MECHANISM_SCOPE',
                                    'mechanical scope receives chamber pressures as measured drivers',
                                    {'phenomena':['internal_leakage']}))
    return a


def test_router_uses_certified_scope_for_in_scope_parameter_evidence():
    r=route_health_evidence(_atlas(),HealthEvidenceEvent('f','parameter_drift',['velocity'],source_scope='mechanical',phenomenon='friction'))
    assert r.decision=='ROUTED_CERTIFIED_SCOPE'
    assert r.routed_scopes==['mechanical']


def test_router_semantic_gap_overrides_generic_variable_overlap():
    r=route_health_evidence(_atlas(),HealthEvidenceEvent('l','mechanism_residual',['pressure_a','pressure_b'],phenomenon='internal_leakage'))
    assert r.decision=='ROUTE_TO_DECLARED_CAPABILITY'
    assert r.route_hints[0]['route']=='HYDRAULIC_MECHANISM_SCOPE'


def test_router_routes_sensor_bias_to_observation_health():
    r=route_health_evidence(_atlas(),HealthEvidenceEvent('b','observation_anomaly',['position'],phenomenon='sensor_bias'))
    assert r.decision=='ROUTE_TO_DECLARED_CAPABILITY'
    assert r.route_hints[0]['route']=='OBSERVATION_HEALTH_OFFSET_SENSITIVE'


def test_router_blocks_uncertified_scope_and_unknowns():
    r=route_health_evidence(_atlas(),HealthEvidenceEvent('h','mechanism_residual',['flow']))
    assert r.decision=='BLOCKED_UNCERTIFIED_SCOPE'
    u=route_health_evidence(_atlas(),HealthEvidenceEvent('u','mechanism_residual',['temperature']))
    assert u.decision=='UNKNOWN_UNCOVERED'
