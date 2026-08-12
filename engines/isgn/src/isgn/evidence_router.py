from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .atlas import MechanismAtlas


@dataclass(frozen=True)
class HealthEvidenceEvent:
    """Device-neutral health evidence submitted to a MechanismAtlas.

    `phenomenon` is an optional semantic hint such as ``sensor_offset`` or
    ``internal_leakage``.  The router never infers a fault label from it; it is
    only used to match an explicitly declared uncovered/capability route.
    """
    event_id: str
    evidence_kind: str
    variables: Sequence[str] = ()
    source_scope: Optional[str] = None
    phenomenon: Optional[str] = None
    score: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceRoute:
    event_id: str
    decision: str
    routed_scopes: List[str] = field(default_factory=list)
    route_hints: List[Dict[str, Any]] = field(default_factory=list)
    uncovered_variables: List[str] = field(default_factory=list)
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _declared_list(mapping: Mapping[str, Any], key: str) -> set[str]:
    raw = mapping.get(key, ())
    if isinstance(raw, str):
        return {raw}
    try:
        return {str(x) for x in raw}
    except TypeError:
        return set()


def _scope_supports(scope, event: HealthEvidenceEvent) -> bool:
    cov = scope.phm_coverage or {}
    kinds = _declared_list(cov, "evidence_kinds")
    if kinds and event.evidence_kind not in kinds:
        return False
    phenomena = _declared_list(cov, "phenomena") | _declared_list(cov, "fault_modes")
    if event.phenomenon and phenomena and event.phenomenon not in phenomena:
        return False
    return True


def _uncovered_match(atlas: MechanismAtlas, event: HealthEvidenceEvent) -> List[Dict[str, Any]]:
    requested = set(map(str, event.variables))
    matches: List[Dict[str, Any]] = []
    for region in atlas.uncovered_regions:
        meta = region.metadata or {}
        phenomena = _declared_list(meta, "phenomena") | _declared_list(meta, "fault_modes")
        phenomenon_match = bool(event.phenomenon and event.phenomenon in phenomena)
        overlap = sorted(requested & set(region.variables))
        variable_match = bool(overlap)
        # A semantic route explicitly documented in the atlas is stronger than
        # generic variable overlap.  This is how a measured pressure can be a
        # valid driver for a mechanical scope yet an internal-leakage event is
        # still routed to an uncovered hydraulic scope.
        if phenomenon_match or variable_match:
            matches.append({
                "region_id": region.region_id,
                "variables": overlap,
                "route": region.route,
                "reason": region.reason,
                "phenomenon_match": phenomenon_match,
            })
    matches.sort(key=lambda x: (not x["phenomenon_match"], x["region_id"]))
    return matches


def route_health_evidence(atlas: MechanismAtlas, event: HealthEvidenceEvent) -> EvidenceRoute:
    """Route health evidence without inventing a diagnosis.

    Order of evidence:
      1. Explicit uncovered/capability route that matches `phenomenon`.
      2. Explicit source scope, if certified and PHM-qualified for this evidence.
      3. Certified scope overlap by variables and declared PHM coverage.
      4. Uncertified overlap -> blocked, never silently used.
      5. Declared uncovered route by variable overlap.
      6. UNKNOWN_UNCOVERED.
    """
    variables = list(dict.fromkeys(map(str, event.variables)))
    requested = set(variables)
    gaps = _uncovered_match(atlas, event)
    semantic_gaps = [g for g in gaps if g["phenomenon_match"]]
    if semantic_gaps:
        return EvidenceRoute(
            event.event_id,
            "ROUTE_TO_DECLARED_CAPABILITY",
            route_hints=semantic_gaps,
            uncovered_variables=variables,
            reason="event matches an explicitly declared uncovered/capability route",
            metadata={"evidence_kind": event.evidence_kind, "phenomenon": event.phenomenon},
        )

    if event.source_scope is not None:
        scope = atlas.scopes.get(event.source_scope)
        if scope is None:
            return EvidenceRoute(event.event_id, "UNKNOWN_UNCOVERED", uncovered_variables=variables,
                                 reason=f"source scope {event.source_scope!r} is not present in atlas")
        if not scope.certified:
            return EvidenceRoute(event.event_id, "BLOCKED_UNCERTIFIED_SCOPE", routed_scopes=[scope.scope_id],
                                 uncovered_variables=variables,
                                 reason=f"source scope is {scope.completion_status}, not COMPLETE")
        if not _scope_supports(scope, event):
            return EvidenceRoute(event.event_id, "BLOCKED_OUTSIDE_PHM_COVERAGE", routed_scopes=[scope.scope_id],
                                 uncovered_variables=variables,
                                 reason="source scope is certified but its PHM coverage does not include this evidence")
        return EvidenceRoute(event.event_id, "ROUTED_CERTIFIED_SCOPE", routed_scopes=[scope.scope_id],
                             reason="explicit certified source scope accepts this evidence",
                             metadata={"evidence_kind": event.evidence_kind, "score": event.score})

    certified: List[str] = []
    uncertified: List[str] = []
    covered: set[str] = set()
    for sid, scope in atlas.scopes.items():
        overlap = requested & scope.variables
        if not overlap:
            continue
        if scope.certified and _scope_supports(scope, event):
            certified.append(sid); covered |= overlap
        elif not scope.certified:
            uncertified.append(sid)

    if len(certified) == 1:
        return EvidenceRoute(event.event_id, "ROUTED_CERTIFIED_SCOPE", routed_scopes=certified,
                             route_hints=gaps, uncovered_variables=sorted(requested-covered),
                             reason="exactly one certified PHM-qualified scope overlaps the evidence")
    if len(certified) > 1:
        return EvidenceRoute(event.event_id, "MULTIPLE_CANDIDATE_SCOPES", routed_scopes=sorted(certified),
                             route_hints=gaps, uncovered_variables=sorted(requested-covered),
                             reason="multiple certified scopes overlap; cross-scope arbitration is required")
    if uncertified:
        return EvidenceRoute(event.event_id, "BLOCKED_UNCERTIFIED_SCOPE", routed_scopes=sorted(uncertified),
                             route_hints=gaps, uncovered_variables=variables,
                             reason="only uncertified scopes overlap the evidence")
    if gaps:
        return EvidenceRoute(event.event_id, "ROUTE_TO_DECLARED_CAPABILITY", route_hints=gaps,
                             uncovered_variables=variables,
                             reason="no certified scope covers the evidence; atlas declares a capability route")
    return EvidenceRoute(event.event_id, "UNKNOWN_UNCOVERED", uncovered_variables=variables,
                         reason="no certified scope or declared capability covers this evidence")


def route_health_evidence_batch(atlas: MechanismAtlas, events: Sequence[HealthEvidenceEvent]) -> List[EvidenceRoute]:
    return [route_health_evidence(atlas, e) for e in events]
