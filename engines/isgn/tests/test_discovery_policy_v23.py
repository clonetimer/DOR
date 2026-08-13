from isgn import recommend_discovery_route


def test_sparse_dynamics_requires_derivative_and_library_qualification():
    decision = recommend_discovery_route(
        problem_kind="DYNAMICS", derivative_status="PASS", variable_count=3,
        sparse_library_available=True, backend_availability={"pysindy": True},
    )
    assert decision.status == "READY"
    assert decision.route == "PYSINDY"


def test_noisy_partial_dynamics_routes_to_trajectory_not_derivative_fit():
    decision = recommend_discovery_route(
        problem_kind="DYNAMICS", derivative_status="FAIL", variable_count=2,
        partial_physics=True, sparse_library_available=True,
    )
    assert decision.route == "TRAJECTORY_RESIDUAL"
    assert "credit-assignment" in " ".join(decision.prerequisites)


def test_high_dimensional_symbolic_gap_requires_reduction_and_external_runtime():
    decision = recommend_discovery_route(
        problem_kind="EXPLICIT_RELATION", variable_count=7, grammar_uncovered=True,
        backend_availability={"symmatika": False},
    )
    assert decision.status == "EXTERNAL_RUNTIME_REQUIRED"
    assert decision.route == "PHYSICS_REDUCER_THEN_SYMMATIKA"
    assert "dimensional reduction" in decision.prerequisites[0]


def test_implicit_bridge_does_not_overstate_current_runtime():
    decision = recommend_discovery_route(problem_kind="IMPLICIT_RELATION", variable_count=3)
    assert decision.status == "EXTERNAL_RUNTIME_REQUIRED"
    assert decision.route == "SYMMATIKA_IMPLICIT_UPSTREAM"
