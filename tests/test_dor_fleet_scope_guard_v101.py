from dor.control import engine_admissions
from dor.contracts import TaskQualification


def q_ok():
    return TaskQualification("FAULT_CLASSIFICATION", "QUALIFIED", (), (), (), {})


def test_asset_local_keeps_simple_score_bridge_route_available():
    m={"reference_scope":{"kind":"ASSET_LOCAL"},"mechanism":{"status":"NOT_PROVIDED"}}
    a=engine_admissions(m,q_ok())
    assert a[1].status == "RUN"
    assert a[2].status == "RUN"
    assert a[3].status == "RUN"


def test_fleet_group_blocks_simple_score_bridge_route():
    m={"reference_scope":{"kind":"FLEET_GROUP","fleet_group_id":"fleet-1","member_asset_ids":["A","B"]},"mechanism":{"status":"NOT_PROVIDED"}}
    a=engine_admissions(m,q_ok())
    assert a[1].status == "RUN"
    assert a[2].status == "BLOCKED"
    assert a[3].status == "BLOCKED"
    assert "fleet" in " ".join(a[2].reasons).lower()
