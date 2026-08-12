from __future__ import annotations

from dataclasses import replace

import pytest

from fec.extensions.evidence_provenance import (
    ConstraintType,
    EvidenceProvenance,
    ModalityAvailability,
    ModalityStatus,
    PhysicsKnowledgeProfile,
    ProviderIdentity,
    ReferenceCompatibilityContract,
    SourceDomain,
    SourceKind,
    audit_evidence_lineage,
    provider_fingerprint,
)


def test_provider_fingerprint_is_deterministic_and_version_config_sensitive():
    a = provider_fingerprint("virtual_pressure", "1.0", configuration={"gain": 2, "features": ["q", "xv"]})
    b = provider_fingerprint("virtual_pressure", "1.0", configuration={"features": ["q", "xv"], "gain": 2})
    assert a == b
    assert a != provider_fingerprint("virtual_pressure", "1.1", configuration={"gain": 2, "features": ["q", "xv"]})
    assert a != provider_fingerprint("virtual_pressure", "1.0", configuration={"gain": 3, "features": ["q", "xv"]})


def _provider(name: str, version: str = "1") -> ProviderIdentity:
    return ProviderIdentity.build(name, version, configuration={"schema": 1})


def test_physics_knowledge_is_provenance_not_runtime_vote_and_roundtrips():
    profile = PhysicsKnowledgeProfile(
        observational_bias=("real pressure sensor", "physics simulation augmentation"),
        inductive_bias=("hydraulic connectivity graph",),
        learning_bias=("mass conservation residual",),
        constraint_type=ConstraintType.SOFT,
        known_physics=("continuity",),
        learned_or_unknown_physics=("unmodelled leakage correction",),
        valid_domain=("supply_pressure=100..140bar",),
    )
    p = EvidenceProvenance(
        source_id="contact:flow_residual",
        source_kind=SourceKind.LEARNED,
        modality_id="hydraulic",
        physical_object="valve",
        asset_id="rig-1",
        acquisition_id="run-7",
        provider=_provider("hybrid_residual", "2.3"),
        source_domain=SourceDomain.HYBRID,
        parents=("sensor:pressure", "sensor:spool"),
        physics_knowledge=profile,
    )
    q = EvidenceProvenance.from_dict(p.to_dict())
    assert q == p
    assert q.physics_knowledge.constraint_type is ConstraintType.SOFT


def test_lineage_audit_flags_shared_source_but_never_assigns_equivalence():
    raw = EvidenceProvenance(
        "sensor:pressure", SourceKind.OBSERVED, "pressure", "supply", "rig", "run",
        _provider("pressure_acquisition"), SourceDomain.REAL,
    )
    virtual = EvidenceProvenance(
        "virtual:flow", SourceKind.VIRTUAL, "flow", "valve", "rig", "run",
        _provider("virtual_flow"), SourceDomain.REAL, parents=("sensor:pressure",),
    )
    residual = EvidenceProvenance(
        "contact:pressure_residual", SourceKind.LEARNED, "pressure", "valve", "rig", "run",
        _provider("residual"), SourceDomain.HYBRID, parents=("sensor:pressure", "external:command"),
    )
    report = audit_evidence_lineage((raw, virtual, residual))
    assert report.valid_dag
    assert report.automatic_equivalence_assignments == 0
    pairs = {(x.left_source_id, x.right_source_id): x.shared_upstream_sources for x in report.overlaps}
    assert "sensor:pressure" in pairs[("sensor:pressure", "virtual:flow")]
    assert "sensor:pressure" in pairs[("virtual:flow", "contact:pressure_residual")]
    assert report.unresolved_parents == ("external:command",)


def test_lineage_cycle_is_invalid_metadata_not_silently_flattened():
    a = EvidenceProvenance(
        "a", SourceKind.VIRTUAL, "m1", "o", "asset", "run", _provider("a"), SourceDomain.REAL,
        parents=("b",),
    )
    b = EvidenceProvenance(
        "b", SourceKind.VIRTUAL, "m2", "o", "asset", "run", _provider("b"), SourceDomain.REAL,
        parents=("a",),
    )
    report = audit_evidence_lineage((a, b))
    assert not report.valid_dag
    assert report.cycles
    assert report.overlaps == ()


def test_modality_availability_distinguishes_observed_derived_simulated_and_missing():
    m = ModalityAvailability.from_mapping({
        "pressure": ModalityStatus.OBSERVED,
        "flow": ModalityStatus.DERIVED,
        "temperature": ModalityStatus.SIMULATED,
        "vibration": ModalityStatus.MISSING,
    })
    assert m.presence_mask(("pressure", "flow", "temperature", "vibration")) == (1, 1, 1, 0)
    assert dict(m.signature)["flow"] == "derived"
    assert dict(m.signature)["vibration"] == "missing"


def test_reference_compatibility_is_exact_on_declared_dimensions():
    p1 = _provider("contact_pack", "1")
    p2 = _provider("contact_pack", "2")
    observed = ModalityAvailability.from_mapping({"pressure": "observed", "vibration": "observed"})
    derived = ModalityAvailability.from_mapping({"pressure": "derived", "vibration": "observed"})
    expected = ReferenceCompatibilityContract.from_runtime(
        availability=observed,
        provider_fingerprints={p1.fingerprint},
        source_domain=SourceDomain.REAL,
    )
    same = ReferenceCompatibilityContract.from_runtime(
        availability=observed,
        provider_fingerprints={p1.fingerprint},
        source_domain=SourceDomain.REAL,
    )
    assert expected.compatible_with(same)
    assert not expected.compatible_with(ReferenceCompatibilityContract.from_runtime(
        availability=derived, provider_fingerprints={p1.fingerprint}, source_domain=SourceDomain.REAL
    ))
    assert not expected.compatible_with(ReferenceCompatibilityContract.from_runtime(
        availability=observed, provider_fingerprints={p2.fingerprint}, source_domain=SourceDomain.REAL
    ))
