from dor.production_evidence import evaluate_production_evidence, EVIDENCE_CONTRACT


def _base_gates():
    passed = [
        "SOFTWARE_REGRESSION",
        "RELEASE_HYGIENE",
        "REAL_TARGET_DATA",
        "ROLE_SEPARATION",
        "SUBGROUP_REPORTING",
    ]
    return [
        {"gate": g, "status": "PASS", "reason": "fixture", "evidence": [g.lower()]}
        for g in passed
    ]


def test_assisted_poc_can_pass_while_production_remains_blocked():
    result = evaluate_production_evidence({
        "contract": EVIDENCE_CONTRACT,
        "target_use": "operator-assisted anomaly triage",
        "target_domain": "satellite",
        "gates": _base_gates(),
    })
    assert result["profiles"]["ASSISTED_POC"]["status"] == "SUPPORTED"
    assert result["profiles"]["ASSISTED_PRODUCTION"]["status"] == "BLOCKED"
    assert "DISTRIBUTION_SHIFT" in result["profiles"]["ASSISTED_PRODUCTION"]["blocking_gates"]
    assert result["profiles"]["SAFETY_CRITICAL_AUTONOMY"]["status"] == "BLOCKED"


def test_not_run_never_counts_as_pass():
    gates = _base_gates()
    gates[-1] = {"gate": "SUBGROUP_REPORTING", "status": "NOT_RUN", "reason": "not evaluated", "evidence": []}
    result = evaluate_production_evidence({
        "contract": EVIDENCE_CONTRACT,
        "target_use": "fixture",
        "gates": gates,
    })
    assert result["profiles"]["ASSISTED_POC"]["status"] == "BLOCKED"
    assert "SUBGROUP_REPORTING" in result["profiles"]["ASSISTED_POC"]["blocking_gates"]


def test_invalid_contract_blocks_all_profiles():
    result = evaluate_production_evidence({
        "contract": "WRONG",
        "target_use": "fixture",
        "gates": _base_gates(),
    })
    assert result["errors"]
    assert all(v["status"] == "BLOCKED" for v in result["profiles"].values())
