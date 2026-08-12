from __future__ import annotations

import pytest

from fec.extensions.evidence_provenance import (
    EvidenceProvenance,
    ModalityStatus,
    ProviderIdentity,
    SourceDomain,
    SourceKind,
    resolve_evidence_availability,
)


def provider(name: str, version: str = "1") -> ProviderIdentity:
    return ProviderIdentity.build(name, version, configuration={"contract": 1})


def observed(source_id: str, modality: str) -> EvidenceProvenance:
    return EvidenceProvenance(
        source_id, SourceKind.OBSERVED, modality, "hydraulic-rig", "zema", "campaign",
        provider("acq:" + modality), SourceDomain.REAL,
    )


def zema_graph():
    ts1 = observed("sensor:TS1", "TS1")
    ts2 = observed("sensor:TS2", "TS2")
    ts3 = observed("sensor:TS3", "TS3")
    ts4 = observed("sensor:TS4", "TS4")
    fs2 = observed("sensor:FS2", "FS2")
    cp = EvidenceProvenance(
        "virtual:CP:original", SourceKind.VIRTUAL, "CP", "cooler", "zema", "campaign",
        provider("cp_physics", "1"), SourceDomain.REAL,
        parents=("sensor:TS3", "sensor:TS4", "sensor:FS2"),
    )
    ce = EvidenceProvenance(
        "virtual:CE:original", SourceKind.VIRTUAL, "CE", "cooler", "zema", "campaign",
        provider("ce_physics", "1"), SourceDomain.REAL,
        parents=("sensor:TS3", "sensor:TS4", "external:ambient_temperature"),
    )
    bad_ts4 = EvidenceProvenance(
        "virtual:TS4:from_CP", SourceKind.VIRTUAL, "TS4", "cooler", "zema", "campaign",
        provider("ts4_inverse_cp", "1"), SourceDomain.REAL,
        parents=("sensor:TS3", "sensor:FS2", "virtual:CP:original"),
    )
    good_ts4 = EvidenceProvenance(
        "virtual:TS4:poly", SourceKind.LEARNED, "TS4", "cooler", "zema", "campaign",
        provider("ts4_poly", "1"), SourceDomain.REAL,
        parents=("sensor:TS1", "sensor:TS2", "sensor:TS3", "sensor:FS2"),
    )
    cp_failover = EvidenceProvenance(
        "virtual:CP:failover", SourceKind.VIRTUAL, "CP", "cooler", "zema", "campaign",
        provider("cp_physics", "2"), SourceDomain.REAL,
        parents=("sensor:TS3", "virtual:TS4:poly", "sensor:FS2"),
    )
    return (ts1, ts2, ts3, ts4, fs2, cp, ce, bad_ts4, good_ts4, cp_failover)


def full_observed():
    return {"TS1": "observed", "TS2": "observed", "TS3": "observed", "TS4": "observed", "FS2": "observed"}


def test_missing_observed_sensor_cascades_to_virtual_sources_even_if_stale_values_exist():
    m = full_observed(); m["TS4"] = "missing"
    r = resolve_evidence_availability(zema_graph(), m, available_external_sources={"external:ambient_temperature"})
    assert not r.state("sensor:TS4").available
    assert not r.state("virtual:CP:original").available
    assert not r.state("virtual:CE:original").available
    assert "sensor:TS4" in r.state("virtual:CP:original").blocking_sources


def test_unresolved_external_parent_fails_closed_until_explicitly_declared():
    r = resolve_evidence_availability(zema_graph(), full_observed())
    assert not r.state("virtual:CE:original").available
    r2 = resolve_evidence_availability(
        zema_graph(), full_observed(), available_external_sources={"external:ambient_temperature"}
    )
    assert r2.state("virtual:CE:original").available


def test_too_good_inverse_virtual_sensor_is_blocked_when_its_hidden_upstream_target_is_missing():
    m = full_observed(); m["TS4"] = ModalityStatus.MISSING
    r = resolve_evidence_availability(zema_graph(), m, available_external_sources={"external:ambient_temperature"})
    assert not r.state("virtual:CP:original").available
    assert not r.state("virtual:TS4:from_CP").available
    assert "virtual:CP:original" in r.state("virtual:TS4:from_CP").blocking_sources


def test_explicit_alternative_provider_restores_ts4_and_downstream_cp_as_derived():
    m = full_observed(); m["TS4"] = "missing"
    r = resolve_evidence_availability(zema_graph(), m, available_external_sources={"external:ambient_temperature"})
    assert r.state("virtual:TS4:poly").available
    assert r.state("virtual:TS4:poly").effective_status is ModalityStatus.DERIVED
    assert r.state("virtual:CP:failover").available
    availability = r.materialize_modality_availability({
        "TS4": "virtual:TS4:poly",
        "CP": "virtual:CP:failover",
        "TS3": "sensor:TS3",
        "FS2": "sensor:FS2",
    })
    assert dict(availability.states)["TS4"] is ModalityStatus.DERIVED
    assert dict(availability.states)["CP"] is ModalityStatus.DERIVED


def test_reference_contract_uses_transitive_provider_closure_and_rejects_failover_change():
    graph = zema_graph()
    full = resolve_evidence_availability(graph, full_observed(), available_external_sources={"external:ambient_temperature"})
    c_full = full.reference_compatibility(
        {"TS4": "sensor:TS4", "CP": "virtual:CP:original", "TS3": "sensor:TS3", "FS2": "sensor:FS2"},
        source_domain=SourceDomain.REAL,
    )
    m = full_observed(); m["TS4"] = "missing"
    fail = resolve_evidence_availability(graph, m, available_external_sources={"external:ambient_temperature"})
    c_fail = fail.reference_compatibility(
        {"TS4": "virtual:TS4:poly", "CP": "virtual:CP:failover", "TS3": "sensor:TS3", "FS2": "sensor:FS2"},
        source_domain=SourceDomain.REAL,
    )
    assert c_full.modality_signature != c_fail.modality_signature
    assert c_full.provider_fingerprints != c_fail.provider_fingerprints
    assert not c_full.compatible_with(c_fail)


def test_selecting_blocked_source_cannot_build_a_production_compatibility_contract():
    m = full_observed(); m["TS4"] = "missing"
    r = resolve_evidence_availability(zema_graph(), m, available_external_sources={"external:ambient_temperature"})
    with pytest.raises(ValueError, match="selected runtime sources are unavailable"):
        r.reference_compatibility(
            {"TS4": "sensor:TS4", "CP": "virtual:CP:original"}, source_domain=SourceDomain.REAL
        )


def test_cyclic_lineage_fails_closed():
    a = EvidenceProvenance(
        "a", SourceKind.VIRTUAL, "A", "o", "asset", "run", provider("a"), SourceDomain.REAL, parents=("b",)
    )
    b = EvidenceProvenance(
        "b", SourceKind.VIRTUAL, "B", "o", "asset", "run", provider("b"), SourceDomain.REAL, parents=("a",)
    )
    r = resolve_evidence_availability((a, b), {})
    assert not r.valid_dag
    assert not r.state("a").available
    assert not r.state("b").available
