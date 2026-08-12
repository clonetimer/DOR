"""FEC v2.2 Contact Pack template.

Copy this file into your device project. Device/sensor/physics logic belongs
here; do not place device-specific rules in fec/core/fec_core.py.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from fec.core.fec_core import ContactSpec
from fec.extensions.evidence_provenance import EvidenceProvenance, ModalityAvailability


@dataclass(frozen=True)
class ContactBatch:
    X: np.ndarray
    specs: tuple[ContactSpec, ...]
    A: np.ndarray
    G: np.ndarray
    context_key: str
    data_id: str
    provenance: tuple[EvidenceProvenance, ...]
    modality_availability: ModalityAvailability


def build_contacts(raw, *, context_key: str, data_id: str) -> ContactBatch:
    """Convert device data into physically interpretable Contact columns.

    Required design work for a real adapter:
    - compute Contact values from sensors / residuals / probes;
    - set G=0 when a Contact cannot be computed;
    - set A according to method, acquisition and excitation applicability;
    - assign semantic topology paths and equivalence ids;
    - declare per-source provenance/parents/provider fingerprint;
    - distinguish OBSERVED / DERIVED / SIMULATED / MISSING modality state;
    - keep fault labels out of Contact values and calibration selection.

    Important: shared lineage is a review signal.  Do not automatically mark
    correlated/shared-source Contacts as semantically equivalent.
    """
    raise NotImplementedError
