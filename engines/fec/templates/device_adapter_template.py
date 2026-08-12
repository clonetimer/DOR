"""FEC v2.2 Device Adapter template."""
from __future__ import annotations

from fec.extensions.evidence_provenance import (
    EvidenceProvenance,
    ModalityAvailability,
    ReferenceCompatibilityContract,
)


class DeviceAdapter:
    """Device-side contract. Replace every placeholder with auditable logic."""

    def context_key(self, observation):
        """Return an externally defensible operating/task/phase context key."""
        raise NotImplementedError

    def build_contacts(self, observation):
        """Return a ContactBatch or equivalent X/specs/A/G structure."""
        raise NotImplementedError

    def data_id(self, source):
        """Return a physical-run / acquisition identity used for role separation."""
        raise NotImplementedError

    def trusted_calibration_claims(self, source) -> set[str]:
        """Return externally justified health claims; never infer U from low anomaly score."""
        return set()

    def modality_availability(self, observation) -> ModalityAvailability:
        """Declare OBSERVED / DERIVED / SIMULATED / MISSING for each relevant modality."""
        raise NotImplementedError

    def evidence_provenance(self, observation) -> tuple[EvidenceProvenance, ...]:
        """Return Contact/source lineage. Shared ancestry is audit metadata, not an extra vote."""
        raise NotImplementedError

    def reference_contract(self, observation) -> ReferenceCompatibilityContract:
        """Bind current modality state/provider identities to the admissible Reference Bundle."""
        raise NotImplementedError

    def reference_compatibility(self, observation, active_bundle) -> float:
        """Optional device-specific S_ref scalar after hard reference-contract compatibility passes."""
        return 1.0


# --- v2.2 optional runtime lineage availability hook ---
# Call this BEFORE choosing a Reference Bundle when the Adapter exposes
# virtual/learned/simulated sources.  It deliberately does not auto-select a
# failover provider.
def resolve_runtime_evidence_sources(provenance_graph, observed_modalities, *, external_sources=()):
    from fec.extensions import resolve_evidence_availability
    return resolve_evidence_availability(
        provenance_graph,
        observed_modalities,
        available_external_sources=set(external_sources),
    )
