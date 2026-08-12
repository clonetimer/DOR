from isgn.closed_loop import plan_closed_loop_atlas, evaluate_closed_loop_plan


def _gate(*, dt=.002, blocked=('flow',), flow_noise=2.5, flow_period=.001, variation=100.0):
    return {
        'status':'OBSERVATION_INSUFFICIENT','blocked_states':list(blocked),
        'summary':{'dt':dt},
        'config':{'fail_derivative_noise_ratio':2.0,'fail_measurement_variation_ratio':2.0},
        'states':[{'state':'flow','status':'FAIL','derivative_noise_ratio':flow_noise,
                   'sensor_sample_period_s':flow_period,'variation_to_sensor_floor':variation}],
    }


def test_sampling_gap_recommends_available_sensor_rate_and_closes_on_faster_production_data():
    plan=plan_closed_loop_atlas(system_id='demo',observation=_gate())
    a=plan.actions[0]
    assert a.action_type=='INCREASE_EXPORT_RATE_TO_AVAILABLE_SENSOR_RATE'
    assert a.requested_changes['maximum_export_dt_s']==.001
    assert a.production_safety['truth_channels_allowed'] is False
    after={'observation':_gate(dt=.001,blocked=(),flow_noise=.9,flow_period=.001)}
    ev=evaluate_closed_loop_plan(plan,after=after)
    assert ev.status=='CLOSED'


def test_sensor_resolution_gap_does_not_suggest_truth_substitution():
    plan=plan_closed_loop_atlas(system_id='demo',observation=_gate(variation=.8,flow_noise=1.0,flow_period=.01))
    a=plan.actions[0]
    assert a.action_type=='IMPROVE_SENSOR_OR_PHYSICAL_VARIATION'
    assert a.production_safety['truth_channels_allowed'] is False
    assert 'truth' not in ' '.join(map(str,a.requested_changes.values())).lower()


def test_prediscovery_target_generates_discriminating_experiment_and_consensus_closes_it():
    synthesis={'atlas':{'scopes':{'dyn-q-v':{'graph':{'q':['v']},'metadata':{
        'discovery_required_targets':['v'],'candidate_driver_pool':['force','pressure']}}}}}
    plan=plan_closed_loop_atlas(system_id='demo',synthesis=synthesis)
    a=plan.actions[0]
    assert a.gap_type=='DISCOVERY_REQUIRED'
    assert a.action_type=='DESIGN_DISCRIMINATING_EXCITATION_AND_RUN_CONSENSUS'
    after={'consensus':{'consensus':{'status':'CONSENSUS','graph_jaccard':1.0,'operator_agreement':1.0},
                        'primary':{'graph':{'q':['v'],'v':['force','pressure']}}}}
    assert evaluate_closed_loop_plan(plan,after=after).status=='CLOSED'


def test_backend_disagreement_is_not_auto_selected_and_stays_unresolved_without_new_consensus():
    before={'consensus':{'status':'BACKEND_DISAGREEMENT','graph_jaccard':.33,'operator_agreement':.5,
                         'backends':['a','b'],'disagreement_edges':{'only_a':['x->v'],'only_b':['p->v']}}}
    plan=plan_closed_loop_atlas(system_id='motor',consensus=before)
    assert plan.actions[0].action_type=='ACQUIRE_DISAMBIGUATING_EXCITATION_OR_OBSERVATION'
    assert plan.actions[0].requested_changes['do_not_auto_select_best_error_backend'] is True
    ev=evaluate_closed_loop_plan(plan,after={'system_id':'motor'})
    assert ev.status=='UNRESOLVED'


def test_grammar_gap_requires_reusable_mechanism_and_same_certificate_closes_it():
    before={'status':'GRAMMAR_UNCOVERED','coverage_target':'velocity','coverage_ratio':12.4}
    plan=plan_closed_loop_atlas(system_id='demo',completion=before)
    a=plan.actions[0]
    assert a.action_type=='RUN_REUSABLE_CONSTITUTIVE_LIBRARY_SEARCH'
    assert a.requested_changes['reuse_evidence_required_before_core_grammar_extension'] is True
    ev=evaluate_closed_loop_plan(plan,after={'completion':{'status':'COMPLETE','coverage_target':'velocity','coverage_ratio':1.2,'identifiable':True}})
    assert ev.status=='CLOSED'


def test_multi_gap_plan_prioritizes_observation_before_discovery_but_keeps_both():
    synthesis={'atlas':{'scopes':{'dyn':{'graph':{'q':['v']},'metadata':{'discovery_required_targets':['v'],'candidate_driver_pool':['u']}}}}}
    plan=plan_closed_loop_atlas(system_id='demo',observation=_gate(),synthesis=synthesis)
    assert [a.gap_type for a in plan.actions]==['OBSERVATION_BLOCKED','DISCOVERY_REQUIRED']


def test_observe_compare_state_route_is_consumed_directly_without_audit_truth_becoming_action_input():
    observation={
      'status':'PARTIAL_SCOPE_READY','blocked_states':['temp'],
      'state_routes':[{'state':'temp','eligibility':'BLOCKED','route':'SENSOR_RESOLUTION_OR_EXCITATION_CHANGE_REQUIRED',
        'preferred_candidate':None,'preferred_dt_s':None,'max_direct_export_dt_s':None,
        'reason':'below sensor floor','recommendation':'improve sensor','evidence':{
          'production':{'public':{'dt_s':.01,'status':'WARN','derivative_noise_ratio':1.2,'variation_to_sensor_floor':.9}},
          'audit_only':{'truth':{'dt_s':.01,'status':'PASS','derivative_noise_ratio':.2}}}}]}
    plan=plan_closed_loop_atlas(system_id='demo',observation=observation)
    assert len(plan.actions)==1
    a=plan.actions[0]
    assert a.action_type=='IMPROVE_SENSOR_OR_PHYSICAL_VARIATION'
    assert a.production_safety['truth_channels_allowed'] is False
    assert 'truth' not in str(a.requested_changes).lower()


def test_observe_compare_after_route_can_close_observation_action():
    before={'status':'OBSERVATION_BLOCKED','blocked_states':['x'],'state_routes':[{
      'state':'x','eligibility':'BLOCKED','route':'UNRESOLVED_OBSERVATION_LIMIT','evidence':{'production':{'p':{'dt_s':.01,'status':'FAIL','derivative_noise_ratio':3.0}}}}]}
    plan=plan_closed_loop_atlas(system_id='demo',observation=before)
    after={'system_id':'demo','observation':{'status':'PARTIAL_SCOPE_READY','blocked_states':[], 'state_routes':[{
      'state':'x','eligibility':'CONDITIONAL','route':'FILTERED_OR_WEAK_FRONTEND','preferred_candidate':'p2',
      'evidence':{'production':{'p2':{'dt_s':.01,'status':'WARN','derivative_noise_ratio':.8}}}}]}}
    ev=evaluate_closed_loop_plan(plan,after=after)
    assert ev.status=='CLOSED'


def test_persistent_excitation_failure_generates_safe_excitation_design_and_closes_only_after_gate_passes():
    obs={'status':'OBSERVATION_INSUFFICIENT','blocked_states':[],'states':[],
         'checks':[{'name':'persistent_excitation','status':'FAIL','metrics':{'dynamic_inputs':['u'],'rank_ratio':.25},'reason':'low rank'}]}
    plan=plan_closed_loop_atlas(system_id='demo',observation=obs)
    assert len(plan.actions)==1
    a=plan.actions[0]
    assert a.action_type=='DESIGN_PERSISTENT_EXCITATION'
    assert a.requested_changes['preserve_normal_operating_domain'] is True
    after={'observation':{'checks':[{'name':'persistent_excitation','status':'PASS','metrics':{'dynamic_inputs':['u'],'rank_ratio':1.0}}],
                          'blocked_states':[],'states':[]}}
    assert evaluate_closed_loop_plan(plan,after=after).status=='CLOSED'


def test_source_identifiability_failure_generates_decorrelation_action():
    obs={'status':'OBSERVATION_INSUFFICIENT','blocked_states':[],'states':[],
         'checks':[{'name':'source_identifiability','status':'FAIL','metrics':{'inactive_sources':['load'],'condition_number':1e12}}]}
    plan=plan_closed_loop_atlas(system_id='demo',observation=obs)
    assert plan.actions[0].action_type=='DECORRELATE_OR_ACTIVATE_CANDIDATE_SOURCES'
    assert plan.actions[0].target_variables==['load']


def test_closed_loop_cycle_replans_from_new_evidence_without_merging_old_failures():
    from isgn.closed_loop import run_closed_loop_cycle
    before={'system_id':'demo','observation':_gate()}
    after={'system_id':'demo','observation':_gate(dt=.001,blocked=(),flow_noise=.8,flow_period=.001)}
    cycle=run_closed_loop_cycle(before=before,after=after)
    assert cycle.status=='CYCLE_CLOSED'
    assert cycle.evaluation.status=='CLOSED'
    assert cycle.next_plan.status=='NO_ACTION_NEEDED'
