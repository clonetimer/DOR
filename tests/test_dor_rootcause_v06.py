from dor.localization import localize_faults
from dor.rootcause import infer_root_causes


def base_ontology(provenance=True):
    edge=lambda s,t: {"source":s,"target":t,"relation":"POWER_FLOW", **({"provenance":"unit-test topology"} if provenance else {})}
    return {
      "contract":"DOR-FAULT-ONTOLOGY/0.5","ontology_id":"rc","domain":"test",
      "components":[
        {"component_id":"A","component_type":"source","kind":"PHYSICAL"},
        {"component_id":"B","component_type":"mid","kind":"PHYSICAL"},
        {"component_id":"C","component_type":"load","kind":"PHYSICAL"},
      ],
      "sensors":[],"commands":[],
      "topology_edges":[edge("A","B"),edge("B","C")],
      "faults":[
        {"fault_id":"A_fault","target_component_id":"A","fault_mode":"failed","category":"COMPONENT","exclusion_group":"A_health","expected_evidence":[{"scope_type":"COMPONENT","scope_id":"A","expected_state":"ABNORMAL"}]},
        {"fault_id":"B_fault","target_component_id":"B","fault_mode":"failed","category":"COMPONENT","exclusion_group":"B_health","expected_evidence":[{"scope_type":"COMPONENT","scope_id":"B","expected_state":"ABNORMAL"}]},
      ],
      "provenance":{"sources":["unit-test"]}
    }


def e(eid, comp, state="ABNORMAL", t=1.0, strength=1.0):
    return {"evidence_id":eid,"scope_type":"COMPONENT","scope_id":comp,"state":state,"strength":strength,"validity":"VALID","timestamp":t}


def test_unique_upstream_root_requires_temporal_provenanced_path():
    o=base_ontology(True)
    rows=[e("a","A",t=1),e("b","B",t=2),e("c","C",t=3)]
    loc=localize_faults(o,rows)
    assert loc["status"] == "SUPPORTED" and set(loc["supported_fault_set"])=={"A_fault","B_fault"}
    rc=infer_root_causes(o,loc,rows)
    assert rc["status"]=="SUPPORTED"
    assert rc["root_cause_set"]==["A_fault"]
    assert rc["safety"]["root_cause_claimed"] is True
    assert rc["safety"]["propagation_inferred"] is True


def test_earlier_downstream_anomaly_is_not_falsely_explained_by_later_upstream_fault():
    o=base_ontology(True)
    rows=[e("a","A",t=2),e("b","B",t=1),e("c","C",t=3)]
    loc=localize_faults(o,rows)
    rc=infer_root_causes(o,loc,rows)
    # B occurred before A's fault-specific support, so A cannot explain B.
    # The conservative result is a two-root set rather than relabelling the
    # earliest anomaly itself as the root cause.
    assert rc["status"]=="SUPPORTED"
    assert set(rc["root_cause_set"])=={"A_fault","B_fault"}
    ca=[x for x in rc["root_candidates"] if x["fault_id"]=="A_fault"][0]
    assert ca["temporal_contradictions"]
    assert "b" not in ca["covered_evidence"]
    assert rc["safety"]["earliest_anomaly_is_not_root_cause_rule"] is True


def test_unprovenanced_topology_cannot_support_propagation_claim():
    o=base_ontology(False)
    rows=[e("a","A",t=1),e("b","B",t=2),e("c","C",t=3)]
    loc=localize_faults(o,rows)
    rc=infer_root_causes(o,loc,rows)
    assert rc["status"]=="UNKNOWN"
    assert any(x["unprovenanced_paths"] for x in rc["root_candidates"])
    assert rc["safety"]["root_cause_claimed"] is False


def test_no_timestamps_stays_at_localization_level():
    o=base_ontology(True)
    rows=[e("a","A",t=None),e("b","B",t=None),e("c","C",t=None)]
    loc=localize_faults(o,rows)
    rc=infer_root_causes(o,loc,rows)
    assert rc["status"]=="UNKNOWN"
    assert all(x["root_support_time"] is None for x in rc["root_candidates"])


def test_two_independent_root_causes_are_supported_as_set():
    o=base_ontology(True)
    o["components"] += [
      {"component_id":"X","component_type":"source","kind":"PHYSICAL"},
      {"component_id":"Y","component_type":"load","kind":"PHYSICAL"},
    ]
    o["topology_edges"].append({"source":"X","target":"Y","relation":"POWER_FLOW","provenance":"unit-test topology"})
    o["faults"].append({"fault_id":"X_fault","target_component_id":"X","fault_mode":"failed","category":"COMPONENT","exclusion_group":"X_health","expected_evidence":[{"scope_type":"COMPONENT","scope_id":"X","expected_state":"ABNORMAL"}]})
    rows=[e("a","A",t=1),e("b","B",t=2),e("x","X",t=1),e("y","Y",t=2)]
    loc=localize_faults(o,rows)
    rc=infer_root_causes(o,loc,rows,max_root_set_size=3)
    assert rc["status"]=="SUPPORTED"
    assert set(rc["root_cause_set"])=={"A_fault","X_fault"}
    assert rc["safety"]["concurrent_root_causes_allowed"] is True


def test_equivalent_root_sets_remain_alternatives():
    o={
      "contract":"DOR-FAULT-ONTOLOGY/0.5","ontology_id":"alt","domain":"test",
      "components":[
        {"component_id":"A","component_type":"source","kind":"PHYSICAL"},
        {"component_id":"X","component_type":"source","kind":"PHYSICAL"},
        {"component_id":"C","component_type":"load","kind":"PHYSICAL"},
      ],
      "sensors":[],
      "commands":[{"command_id":"CmdA","target_component_id":"A"},{"command_id":"CmdX","target_component_id":"X"}],
      "topology_edges":[
        {"source":"A","target":"C","relation":"POWER_FLOW","provenance":"unit-test topology"},
        {"source":"X","target":"C","relation":"POWER_FLOW","provenance":"unit-test topology"},
      ],
      "faults":[
        {"fault_id":"A_fault","target_component_id":"A","fault_mode":"failed","category":"COMPONENT","exclusion_group":"A_health","expected_evidence":[{"scope_type":"COMMAND","scope_id":"CmdA","expected_state":"CLOSED"}]},
        {"fault_id":"X_fault","target_component_id":"X","fault_mode":"failed","category":"COMPONENT","exclusion_group":"X_health","expected_evidence":[{"scope_type":"COMMAND","scope_id":"CmdX","expected_state":"CLOSED"}]},
      ],
      "provenance":{"sources":["unit-test"]}
    }
    rows=[
      {"evidence_id":"ca","scope_type":"COMMAND","scope_id":"CmdA","state":"CLOSED","strength":1.0,"validity":"VALID","timestamp":1.0},
      {"evidence_id":"cx","scope_type":"COMMAND","scope_id":"CmdX","state":"CLOSED","strength":1.0,"validity":"VALID","timestamp":1.0},
      e("c","C",t=2.0),
    ]
    loc=localize_faults(o,rows)
    rc=infer_root_causes(o,loc,rows)
    assert rc["status"]=="ALTERNATIVES"
    assert {tuple(x["fault_ids"]) for x in rc["candidate_root_sets"]} == {("A_fault",),("X_fault",)}
    assert rc["safety"]["root_cause_claimed"] is False


def test_localization_alone_is_not_renamed_root_cause():
    o=base_ontology(True)
    rows=[e("a","A",t=1)]
    loc=localize_faults(o,rows)
    rc=infer_root_causes(o,loc,rows)
    assert rc["status"]=="UNKNOWN"
    assert rc["safety"]["topology_only_is_not_sufficient"] is True


def test_feedback_edge_is_not_treated_as_physical_propagation_path():
    o=base_ontology(True)
    o["topology_edges"]=[{"source":"A","target":"B","relation":"FEEDBACK","provenance":"unit-test feedback"}]
    rows=[e("a","A",t=1),e("b","B",t=2)]
    loc=localize_faults(o,rows)
    rc=infer_root_causes(o,loc,rows)
    assert rc["status"]=="UNKNOWN"
    assert rc["safety"]["topology_only_is_not_sufficient"] is True


def test_sensor_health_fault_is_not_silently_relabelled_as_observed_component_fault():
    o={
      "contract":"DOR-FAULT-ONTOLOGY/0.5","ontology_id":"sensor-separation","domain":"test",
      "components":[
        {"component_id":"Plant","component_type":"load","kind":"PHYSICAL"},
        {"component_id":"S1_health","component_type":"sensor-health","kind":"SENSOR"},
      ],
      "sensors":[{"sensor_id":"S1","health_component_id":"S1_health","observes_components":["Plant"]}],
      "commands":[],"topology_edges":[],
      "faults":[
        {"fault_id":"S1_stuck","target_component_id":"S1_health","fault_mode":"stuck","category":"SENSOR","exclusion_group":"S1_health_group","expected_evidence":[{"scope_type":"SENSOR","scope_id":"S1","expected_state":"STUCK"}]},
        {"fault_id":"Plant_fault","target_component_id":"Plant","fault_mode":"failed","category":"COMPONENT","exclusion_group":"Plant_health","expected_evidence":[{"scope_type":"COMPONENT","scope_id":"Plant","expected_state":"ABNORMAL"}]},
      ],
      "provenance":{"sources":["unit-test"]}
    }
    rows=[
      {"evidence_id":"s","scope_type":"SENSOR","scope_id":"S1","state":"STUCK","strength":1.0,"validity":"VALID","timestamp":1.0},
      e("p","Plant",t=2.0),
    ]
    loc=localize_faults(o,rows)
    assert set(loc["supported_fault_set"])=={"S1_stuck","Plant_fault"}
    rc=infer_root_causes(o,loc,rows)
    # No provenance-bound propagation path exists between the sensor-health
    # component and the physical plant.  Root-cause v0.6 must not invent one.
    assert rc["status"]=="UNKNOWN"
    assert rc["safety"]["root_cause_claimed"] is False


def test_unexplained_valid_anomaly_prevents_forced_single_root():
    o=base_ontology(True)
    o["components"].append({"component_id":"Z","component_type":"isolated","kind":"PHYSICAL"})
    rows=[e("a","A",t=1),e("b","B",t=2),e("z","Z",t=2)]
    loc=localize_faults(o,rows)
    rc=infer_root_causes(o,loc,rows,coverage_required=1.0)
    assert rc["status"]=="UNKNOWN"
    assert rc["safety"]["root_cause_claimed"] is False
