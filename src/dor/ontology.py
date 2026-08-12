from __future__ import annotations

"""DOR v0.5 fault-semantics contracts.

This module intentionally separates *declared diagnostic semantics* from model
scores.  An ontology is a candidate hypothesis space, not ground truth.  Every
component/fault/sensor/mechanism binding must be provenance-bound and may be
rejected when references are inconsistent.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json

ONTOLOGY_CONTRACT = "DOR-FAULT-ONTOLOGY/0.5"
MECHANISM_MAP_CONTRACT = "DOR-MECHANISM-COMPONENT-MAP/0.5"

ALLOWED_COMPONENT_KINDS = {"PHYSICAL", "SENSOR", "ACTUATOR", "SOFTWARE", "VIRTUAL"}
ALLOWED_FAULT_CATEGORIES = {"COMPONENT", "SENSOR", "COMMAND", "PARAMETRIC", "DISCRETE", "SOFTWARE"}
ALLOWED_RELATIONS = {
    "POWER_FLOW", "FLUID_FLOW", "MECHANICAL", "THERMAL", "CONTROL", "SIGNAL", "FEEDBACK", "CONTAINS", "OBSERVES"
}
ALLOWED_SCOPE_TYPES = {"SENSOR", "COMPONENT", "MECHANISM", "COMMAND"}
ALLOWED_EXPECTED_STATES = {
    "ABNORMAL", "NORMAL", "HIGH", "LOW", "STUCK", "OFFSET", "NO_RESPONSE", "FAILED_OFF", "OPEN", "CLOSED", "DEGRADED", "CLASS"
}


def _as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class OntologyValidation:
    state: str
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_fault_ontology(payload: dict[str, Any]) -> OntologyValidation:
    errors: list[str] = []
    warnings: list[str] = []
    if payload.get("contract") != ONTOLOGY_CONTRACT:
        errors.append(f"contract must be {ONTOLOGY_CONTRACT}")

    components = _as_list(payload.get("components"))
    sensors = _as_list(payload.get("sensors"))
    commands = _as_list(payload.get("commands"))
    edges = _as_list(payload.get("topology_edges"))
    faults = _as_list(payload.get("faults"))

    comp_ids: set[str] = set()
    for c in components:
        cid = str(c.get("component_id", "")).strip()
        if not cid:
            errors.append("component without component_id")
            continue
        if cid in comp_ids:
            errors.append(f"duplicate component_id: {cid}")
        comp_ids.add(cid)
        kind = str(c.get("kind", "PHYSICAL")).upper()
        if kind not in ALLOWED_COMPONENT_KINDS:
            errors.append(f"component {cid} has unsupported kind {kind}")
        parent = c.get("parent_component_id")
        if parent and str(parent) == cid:
            errors.append(f"component {cid} cannot be its own parent")

    sensor_ids: set[str] = set()
    for s in sensors:
        sid = str(s.get("sensor_id", "")).strip()
        if not sid:
            errors.append("sensor without sensor_id")
            continue
        if sid in sensor_ids:
            errors.append(f"duplicate sensor_id: {sid}")
        sensor_ids.add(sid)
        hc = str(s.get("health_component_id", "")).strip()
        if hc and hc not in comp_ids:
            errors.append(f"sensor {sid} health_component_id {hc} not found in components")
        for cid in _as_list(s.get("observes_components")):
            if str(cid) not in comp_ids:
                errors.append(f"sensor {sid} observes unknown component {cid}")

    command_ids: set[str] = set()
    for c in commands:
        cmd = str(c.get("command_id", "")).strip()
        if not cmd:
            errors.append("command without command_id")
            continue
        if cmd in command_ids:
            errors.append(f"duplicate command_id: {cmd}")
        command_ids.add(cmd)
        target = str(c.get("target_component_id", "")).strip()
        if target not in comp_ids:
            errors.append(f"command {cmd} target component {target} not found")

    for e in edges:
        src = str(e.get("source", "")); dst = str(e.get("target", "")); rel = str(e.get("relation", "")).upper()
        if src not in comp_ids or dst not in comp_ids:
            errors.append(f"topology edge references unknown component: {src}->{dst}")
        if rel not in ALLOWED_RELATIONS:
            errors.append(f"topology edge {src}->{dst} has unsupported relation {rel}")

    fault_ids: set[str] = set()
    for f in faults:
        fid = str(f.get("fault_id", "")).strip()
        if not fid:
            errors.append("fault without fault_id")
            continue
        if fid in fault_ids:
            errors.append(f"duplicate fault_id: {fid}")
        fault_ids.add(fid)
        target = str(f.get("target_component_id", "")).strip()
        if target not in comp_ids:
            errors.append(f"fault {fid} target component {target} not found")
        cat = str(f.get("category", "COMPONENT")).upper()
        if cat not in ALLOWED_FAULT_CATEGORIES:
            errors.append(f"fault {fid} has unsupported category {cat}")
        signatures = _as_list(f.get("expected_evidence"))
        if not signatures:
            warnings.append(f"fault {fid} has no expected_evidence; it cannot be positively localized by v0.5")
        for sig in signatures:
            st = str(sig.get("scope_type", "")).upper()
            sid = str(sig.get("scope_id", "")).strip()
            state = str(sig.get("expected_state", "ABNORMAL")).upper()
            if state == "CLASS" and not str(sig.get("expected_value", "")).strip():
                errors.append(f"fault {fid} CLASS signature requires expected_value")
            if st not in ALLOWED_SCOPE_TYPES:
                errors.append(f"fault {fid} signature has unsupported scope_type {st}")
            if state not in ALLOWED_EXPECTED_STATES:
                errors.append(f"fault {fid} signature has unsupported expected_state {state}")
            if st == "SENSOR" and sid not in sensor_ids:
                errors.append(f"fault {fid} signature references unknown sensor {sid}")
            if st == "COMPONENT" and sid not in comp_ids:
                errors.append(f"fault {fid} signature references unknown component {sid}")
            if st == "COMMAND" and sid not in command_ids:
                errors.append(f"fault {fid} signature references unknown command {sid}")
            try:
                w = float(sig.get("weight", 1.0))
                if not (0 < w <= 10):
                    errors.append(f"fault {fid} signature weight out of range: {w}")
            except Exception:
                errors.append(f"fault {fid} signature weight is not numeric")
        for sig in _as_list(f.get("contradictions")):
            st = str(sig.get("scope_type", "")).upper()
            if st not in ALLOWED_SCOPE_TYPES:
                errors.append(f"fault {fid} contradiction has unsupported scope_type {st}")

    # parent references are checked after all IDs are known
    for c in components:
        parent = c.get("parent_component_id")
        if parent and str(parent) not in comp_ids:
            errors.append(f"component {c.get('component_id')} parent {parent} not found")

    prov = payload.get("provenance") or {}
    if not prov.get("sources"):
        warnings.append("ontology has no provenance.sources; production admission should reject untraceable semantics")

    return OntologyValidation(
        state="PASS" if not errors else "FAIL",
        errors=tuple(errors), warnings=tuple(warnings),
        counts={"components": len(components), "sensors": len(sensors), "commands": len(commands), "topology_edges": len(edges), "faults": len(faults)},
    )


def validate_mechanism_component_map(payload: dict[str, Any], ontology: dict[str, Any]) -> OntologyValidation:
    errors: list[str] = []
    warnings: list[str] = []
    if payload.get("contract") != MECHANISM_MAP_CONTRACT:
        errors.append(f"contract must be {MECHANISM_MAP_CONTRACT}")
    comp_ids = {str(c.get("component_id")) for c in _as_list(ontology.get("components"))}
    sensor_ids = {str(s.get("sensor_id")) for s in _as_list(ontology.get("sensors"))}
    bindings = _as_list(payload.get("bindings"))
    mids: set[str] = set()
    for b in bindings:
        mid = str(b.get("mechanism_id", "")).strip()
        if not mid:
            errors.append("mechanism binding without mechanism_id")
            continue
        if mid in mids:
            warnings.append(f"mechanism_id {mid} appears in multiple bindings; this is allowed but must be intentional")
        mids.add(mid)
        for cid in _as_list(b.get("component_ids")):
            if str(cid) not in comp_ids:
                errors.append(f"mechanism {mid} references unknown component {cid}")
        for sid in _as_list(b.get("sensor_ids")):
            if str(sid) not in sensor_ids:
                errors.append(f"mechanism {mid} references unknown sensor {sid}")
        if not str(b.get("provenance", "")).strip():
            warnings.append(f"mechanism {mid} binding lacks provenance")
    return OntologyValidation(
        state="PASS" if not errors else "FAIL", errors=tuple(errors), warnings=tuple(warnings),
        counts={"bindings": len(bindings), "unique_mechanisms": len(mids)},
    )


def ontology_index(ontology: dict[str, Any]) -> dict[str, Any]:
    """Build non-lossy indexes used by v0.5 localization."""
    return {
        "components": {str(x["component_id"]): x for x in _as_list(ontology.get("components"))},
        "sensors": {str(x["sensor_id"]): x for x in _as_list(ontology.get("sensors"))},
        "commands": {str(x["command_id"]): x for x in _as_list(ontology.get("commands"))},
        "faults": {str(x["fault_id"]): x for x in _as_list(ontology.get("faults"))},
        "edges": _as_list(ontology.get("topology_edges")),
    }
