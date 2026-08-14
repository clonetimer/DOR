from __future__ import annotations

"""Evidence provenance, lineage, and runtime reference-compatibility contracts.

This module is an Extension/Control-Plane sidecar.  It does not alter scalar FEC
Evidence semantics.  Its purpose is to make the *origin* and *availability* of a
Contact auditable so derived/simulated/learned signals are not silently treated
as independent physical observations and so calibration references can be
invalidated when their modality/provider contract changes.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence
import hashlib
import json


class SourceKind(str, Enum):
    OBSERVED = "observed"
    VIRTUAL = "virtual"
    SIMULATED = "simulated"
    AUGMENTED = "augmented"
    LEARNED = "learned"


class SourceDomain(str, Enum):
    REAL = "real"
    SIMULATION = "simulation"
    HYBRID = "hybrid"


class ModalityStatus(str, Enum):
    OBSERVED = "observed"
    DERIVED = "derived"
    SIMULATED = "simulated"
    MISSING = "missing"


class AssetScopeKind(str, Enum):
    """How narrowly a calibration Reference is allowed to move across assets."""

    ASSET_LOCAL = "asset_local"
    FLEET_GROUP = "fleet_group"


class ConstraintType(str, Enum):
    HARD = "hard"
    SOFT = "soft"
    NONE = "none"


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _normalise_config(obj: Any) -> Any:
    """Convert common config objects to a deterministic JSON-safe structure."""
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            raise ValueError("provider configuration cannot contain non-finite floats")
        return obj
    if isinstance(obj, Mapping):
        return {str(k): _normalise_config(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_normalise_config(x) for x in obj]
    if isinstance(obj, set | frozenset):
        return sorted((_normalise_config(x) for x in obj), key=lambda x: json.dumps(x, sort_keys=True))
    raise TypeError(f"unsupported provider configuration type: {type(obj).__name__}")


def provider_fingerprint(
    name: str,
    version: str,
    *,
    configuration: Mapping[str, Any] | None = None,
    artifact_sha256: str | None = None,
) -> str:
    """Return a deterministic SHA-256 identity for an Evidence provider.

    The fingerprint intentionally includes provider version/configuration and,
    when available, a model/artifact digest.  A provider change should therefore
    invalidate references that explicitly depend on the old fingerprint.
    """
    if not name or not version:
        raise ValueError("provider name and version are required")
    if artifact_sha256 is not None:
        h = artifact_sha256.lower().strip()
        if len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
            raise ValueError("artifact_sha256 must be a 64-character hexadecimal digest")
        artifact_sha256 = h
    payload = {
        "name": str(name),
        "version": str(version),
        "configuration": _normalise_config(configuration or {}),
        "artifact_sha256": artifact_sha256,
    }
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True)
class ProviderIdentity:
    name: str
    version: str
    fingerprint: str

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise ValueError("provider name/version are required")
        if not self.fingerprint.startswith("sha256:") or len(self.fingerprint) != 71:
            raise ValueError("provider fingerprint must use sha256:<64 hex> form")
        if any(c not in "0123456789abcdef" for c in self.fingerprint[7:].lower()):
            raise ValueError("provider fingerprint must be hexadecimal")

    @classmethod
    def build(
        cls,
        name: str,
        version: str,
        *,
        configuration: Mapping[str, Any] | None = None,
        artifact_sha256: str | None = None,
    ) -> "ProviderIdentity":
        return cls(
            str(name),
            str(version),
            provider_fingerprint(name, version, configuration=configuration, artifact_sha256=artifact_sha256),
        )

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version, "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "ProviderIdentity":
        return cls(str(obj["name"]), str(obj["version"]), str(obj["fingerprint"]))


@dataclass(frozen=True)
class PhysicsKnowledgeProfile:
    """How physical knowledge entered a learned/model-derived Contact provider.

    These are provider-provenance descriptors, not additional runtime anomaly
    Evidence.  In particular, a physics loss is not a second Contact vote.
    """

    observational_bias: tuple[str, ...] = ()
    inductive_bias: tuple[str, ...] = ()
    learning_bias: tuple[str, ...] = ()
    constraint_type: ConstraintType = ConstraintType.NONE
    known_physics: tuple[str, ...] = ()
    learned_or_unknown_physics: tuple[str, ...] = ()
    valid_domain: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "observational_bias": list(self.observational_bias),
            "inductive_bias": list(self.inductive_bias),
            "learning_bias": list(self.learning_bias),
            "constraint_type": self.constraint_type.value,
            "known_physics": list(self.known_physics),
            "learned_or_unknown_physics": list(self.learned_or_unknown_physics),
            "valid_domain": list(self.valid_domain),
        }

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "PhysicsKnowledgeProfile":
        return cls(
            observational_bias=tuple(str(x) for x in obj.get("observational_bias", [])),
            inductive_bias=tuple(str(x) for x in obj.get("inductive_bias", [])),
            learning_bias=tuple(str(x) for x in obj.get("learning_bias", [])),
            constraint_type=ConstraintType(str(obj.get("constraint_type", "none"))),
            known_physics=tuple(str(x) for x in obj.get("known_physics", [])),
            learned_or_unknown_physics=tuple(str(x) for x in obj.get("learned_or_unknown_physics", [])),
            valid_domain=tuple(str(x) for x in obj.get("valid_domain", [])),
        )


@dataclass(frozen=True)
class EvidenceProvenance:
    source_id: str
    source_kind: SourceKind
    modality_id: str
    physical_object: str
    asset_id: str
    acquisition_id: str
    provider: ProviderIdentity
    source_domain: SourceDomain
    parents: tuple[str, ...] = ()
    validity_scope: tuple[str, ...] = ()
    physics_knowledge: PhysicsKnowledgeProfile | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        for name in ("source_id", "modality_id", "physical_object", "asset_id", "acquisition_id"):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")
        if self.source_id in self.parents:
            raise ValueError("a provenance source cannot list itself as a parent")
        if len(set(self.parents)) != len(self.parents):
            raise ValueError("provenance parents must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_kind": self.source_kind.value,
            "modality_id": self.modality_id,
            "physical_object": self.physical_object,
            "asset_id": self.asset_id,
            "acquisition_id": self.acquisition_id,
            "provider": self.provider.to_dict(),
            "source_domain": self.source_domain.value,
            "parents": list(self.parents),
            "validity_scope": list(self.validity_scope),
            "physics_knowledge": None if self.physics_knowledge is None else self.physics_knowledge.to_dict(),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "EvidenceProvenance":
        physics = obj.get("physics_knowledge")
        return cls(
            source_id=str(obj["source_id"]),
            source_kind=SourceKind(str(obj["source_kind"])),
            modality_id=str(obj["modality_id"]),
            physical_object=str(obj["physical_object"]),
            asset_id=str(obj["asset_id"]),
            acquisition_id=str(obj["acquisition_id"]),
            provider=ProviderIdentity.from_dict(obj["provider"]),
            source_domain=SourceDomain(str(obj["source_domain"])),
            parents=tuple(str(x) for x in obj.get("parents", [])),
            validity_scope=tuple(str(x) for x in obj.get("validity_scope", [])),
            physics_knowledge=None if physics is None else PhysicsKnowledgeProfile.from_dict(physics),
            notes=str(obj.get("notes", "")),
        )


@dataclass(frozen=True)
class ModalityAvailability:
    """Declared modality state for one inference/calibration configuration."""

    states: tuple[tuple[str, ModalityStatus], ...]

    def __post_init__(self) -> None:
        if not self.states:
            raise ValueError("at least one modality state is required")
        names = [str(k) for k, _ in self.states]
        if any(not x for x in names) or len(set(names)) != len(names):
            raise ValueError("modality ids must be non-empty and unique")
        normalized = tuple(sorted(((str(k), ModalityStatus(v)) for k, v in self.states), key=lambda kv: kv[0]))
        object.__setattr__(self, "states", normalized)

    @classmethod
    def from_mapping(cls, states: Mapping[str, str | ModalityStatus]) -> "ModalityAvailability":
        return cls(tuple((str(k), ModalityStatus(v)) for k, v in states.items()))

    @property
    def signature(self) -> tuple[tuple[str, str], ...]:
        return tuple((k, v.value) for k, v in self.states)

    def presence_mask(self, modality_order: Sequence[str]) -> tuple[int, ...]:
        lookup = dict(self.states)
        missing = [m for m in modality_order if m not in lookup]
        if missing:
            raise KeyError(f"modality order references undeclared modalities: {missing}")
        return tuple(int(lookup[m] is not ModalityStatus.MISSING) for m in modality_order)

    def to_dict(self) -> dict[str, str]:
        return {k: v.value for k, v in self.states}


@dataclass(frozen=True)
class OperatingContextIdentity:
    """Externally declared operating/task context identity for Reference use.

    This is a hard compatibility identity, not a learned context classifier and
    not an anomaly-responsive feature.  Adapters should build it only from
    externally defensible operating/task metadata (for example declared load
    regime, phase, recipe, or DCASE source/target domain).  Values are
    canonicalized so the same declaration has a deterministic fingerprint.
    """

    namespace: str
    attributes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        ns = str(self.namespace)
        if not ns:
            raise ValueError("operating-context namespace is required")
        normalized: list[tuple[str, str]] = []
        names: set[str] = set()
        for key, encoded in self.attributes:
            k = str(key)
            if not k or k in names:
                raise ValueError("operating-context attribute names must be non-empty and unique")
            names.add(k)
            try:
                value = json.loads(str(encoded))
            except Exception as exc:
                raise ValueError(f"invalid canonical operating-context value for {k!r}") from exc
            value = _normalise_config(value)
            enc = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            normalized.append((k, enc))
        if not normalized:
            raise ValueError("at least one operating-context attribute is required")
        object.__setattr__(self, "namespace", ns)
        object.__setattr__(self, "attributes", tuple(sorted(normalized, key=lambda kv: kv[0])))

    @classmethod
    def from_mapping(cls, namespace: str, attributes: Mapping[str, Any]) -> "OperatingContextIdentity":
        normalized = _normalise_config(attributes)
        if not isinstance(normalized, dict):
            raise TypeError("operating-context attributes must be a mapping")
        return cls(
            str(namespace),
            tuple(
                (str(k), json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
                for k, v in normalized.items()
            ),
        )

    @property
    def fingerprint(self) -> str:
        payload = {
            "namespace": self.namespace,
            "attributes": {k: json.loads(v) for k, v in self.attributes},
        }
        return "sha256:" + hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "attributes": {k: json.loads(v) for k, v in self.attributes},
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any] | None) -> "OperatingContextIdentity | None":
        if obj is None:
            return None
        ctx = cls.from_mapping(str(obj["namespace"]), obj.get("attributes", {}))
        declared = obj.get("fingerprint")
        if declared is not None and str(declared) != ctx.fingerprint:
            raise ValueError("operating-context fingerprint does not match canonical attributes")
        return ctx


@dataclass(frozen=True)
class AssetReferenceScope:
    """Explicit asset membership for a reusable calibration Reference.

    The contract is intentionally membership-based rather than schema-based:
    sharing sensors, provider code, or an operating-context label does not by
    itself prove cross-asset exchangeability.  A fleet-group Reference may be
    used only by assets that were explicitly admitted into that group.
    """

    kind: AssetScopeKind
    admitted_asset_ids: frozenset[str]
    fleet_group_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", AssetScopeKind(self.kind))
        ids = frozenset(str(x) for x in self.admitted_asset_ids)
        if not ids or any(not x for x in ids):
            raise ValueError("admitted_asset_ids must contain non-empty asset ids")
        object.__setattr__(self, "admitted_asset_ids", ids)
        if self.kind is AssetScopeKind.ASSET_LOCAL:
            if len(ids) != 1:
                raise ValueError("asset_local scope requires exactly one admitted asset")
            if self.fleet_group_id is not None:
                raise ValueError("asset_local scope must not declare fleet_group_id")
        elif self.kind is AssetScopeKind.FLEET_GROUP:
            if not self.fleet_group_id:
                raise ValueError("fleet_group scope requires fleet_group_id")
            object.__setattr__(self, "fleet_group_id", str(self.fleet_group_id))

    @classmethod
    def asset_local(cls, asset_id: str) -> "AssetReferenceScope":
        return cls(AssetScopeKind.ASSET_LOCAL, frozenset({str(asset_id)}))

    @classmethod
    def fleet_group(cls, fleet_group_id: str, admitted_asset_ids: Iterable[str]) -> "AssetReferenceScope":
        return cls(AssetScopeKind.FLEET_GROUP, frozenset(str(x) for x in admitted_asset_ids), str(fleet_group_id))

    def mismatch_reasons(self, *, asset_id: str | None, fleet_group_id: str | None = None) -> tuple[str, ...]:
        reasons: list[str] = []
        if not asset_id:
            return ("runtime asset_id was not provided",)
        aid = str(asset_id)
        if aid not in self.admitted_asset_ids:
            reasons.append(f"asset_id {aid!r} was not admitted to commissioned reference")
        if self.kind is AssetScopeKind.FLEET_GROUP:
            if fleet_group_id != self.fleet_group_id:
                reasons.append(
                    f"fleet_group_id expected={self.fleet_group_id!r} actual={fleet_group_id!r}"
                )
        return tuple(reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "admitted_asset_ids": sorted(self.admitted_asset_ids),
            "fleet_group_id": self.fleet_group_id,
        }

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any] | None) -> "AssetReferenceScope | None":
        if obj is None:
            return None
        return cls(
            kind=AssetScopeKind(str(obj["kind"])),
            admitted_asset_ids=frozenset(str(x) for x in obj.get("admitted_asset_ids", [])),
            fleet_group_id=None if obj.get("fleet_group_id") is None else str(obj["fleet_group_id"]),
        )


@dataclass(frozen=True)
class ReferenceCompatibilityContract:
    """Reference-side compatibility requirements for runtime use.

    Matching is deliberately exact for declared dimensions.  The FEC control
    plane must not silently route to whichever reference yields the largest
    normality score.
    """

    modality_signature: tuple[tuple[str, str], ...] = ()
    provider_fingerprints: frozenset[str] = frozenset()
    source_domain: SourceDomain | None = None
    asset_scope: AssetReferenceScope | None = None
    operating_context: OperatingContextIdentity | None = None
    runtime_asset_id: str | None = None
    runtime_fleet_group_id: str | None = None

    def __post_init__(self) -> None:
        sig = tuple(sorted(((str(k), ModalityStatus(v).value) for k, v in self.modality_signature), key=lambda kv: kv[0]))
        if len({k for k, _ in sig}) != len(sig):
            raise ValueError("modality_signature contains duplicate modality ids")
        object.__setattr__(self, "modality_signature", sig)
        fps = frozenset(str(x) for x in self.provider_fingerprints)
        if any(
            not x.startswith("sha256:")
            or len(x) != 71
            or any(c not in "0123456789abcdef" for c in x[7:].lower())
            for x in fps
        ):
            raise ValueError("provider_fingerprints must use sha256:<64 hex> form")
        object.__setattr__(self, "provider_fingerprints", fps)
        if self.source_domain is not None:
            object.__setattr__(self, "source_domain", SourceDomain(self.source_domain))
        if self.asset_scope is not None and not isinstance(self.asset_scope, AssetReferenceScope):
            raise TypeError("asset_scope must be an AssetReferenceScope or None")
        if self.operating_context is not None and not isinstance(self.operating_context, OperatingContextIdentity):
            raise TypeError("operating_context must be an OperatingContextIdentity or None")
        if self.runtime_asset_id is not None:
            if not str(self.runtime_asset_id):
                raise ValueError("runtime_asset_id must be non-empty when provided")
            object.__setattr__(self, "runtime_asset_id", str(self.runtime_asset_id))
        if self.runtime_fleet_group_id is not None:
            if not str(self.runtime_fleet_group_id):
                raise ValueError("runtime_fleet_group_id must be non-empty when provided")
            object.__setattr__(self, "runtime_fleet_group_id", str(self.runtime_fleet_group_id))

    @classmethod
    def commissioned(
        cls,
        *,
        availability: ModalityAvailability | None = None,
        provider_fingerprints: Iterable[str] = (),
        source_domain: SourceDomain | str | None = None,
        asset_scope: AssetReferenceScope | None = None,
        operating_context: OperatingContextIdentity | None = None,
    ) -> "ReferenceCompatibilityContract":
        """Build the persisted requirements of a commissioned Reference.

        Runtime asset identity is deliberately separate from the commissioned
        scope.  This prevents a state snapshot from accidentally becoming tied
        to whichever asset happened to execute the commissioning command.
        """
        return cls(
            modality_signature=() if availability is None else availability.signature,
            provider_fingerprints=frozenset(str(x) for x in provider_fingerprints),
            source_domain=None if source_domain is None else SourceDomain(source_domain),
            asset_scope=asset_scope,
            operating_context=operating_context,
        )

    @classmethod
    def from_runtime(
        cls,
        *,
        availability: ModalityAvailability | None = None,
        provider_fingerprints: Iterable[str] = (),
        source_domain: SourceDomain | str | None = None,
        operating_context: OperatingContextIdentity | None = None,
        asset_id: str | None = None,
        fleet_group_id: str | None = None,
    ) -> "ReferenceCompatibilityContract":
        return cls(
            modality_signature=() if availability is None else availability.signature,
            provider_fingerprints=frozenset(str(x) for x in provider_fingerprints),
            source_domain=None if source_domain is None else SourceDomain(source_domain),
            operating_context=operating_context,
            runtime_asset_id=None if asset_id is None else str(asset_id),
            runtime_fleet_group_id=None if fleet_group_id is None else str(fleet_group_id),
        )

    def mismatch_reasons(self, runtime: "ReferenceCompatibilityContract") -> tuple[str, ...]:
        reasons: list[str] = []
        if self.modality_signature and runtime.modality_signature != self.modality_signature:
            reasons.append(
                f"modality_signature expected={self.modality_signature!r} actual={runtime.modality_signature!r}"
            )
        if self.provider_fingerprints and runtime.provider_fingerprints != self.provider_fingerprints:
            reasons.append("provider_fingerprints do not exactly match commissioned reference")
        if self.source_domain is not None and runtime.source_domain != self.source_domain:
            reasons.append(
                f"source_domain expected={self.source_domain.value!r} "
                f"actual={None if runtime.source_domain is None else runtime.source_domain.value!r}"
            )
        if self.operating_context is not None:
            if runtime.operating_context is None:
                reasons.append("runtime operating_context was not provided")
            elif runtime.operating_context.fingerprint != self.operating_context.fingerprint:
                reasons.append(
                    "operating_context expected="
                    f"{self.operating_context.fingerprint!r} actual={runtime.operating_context.fingerprint!r}"
                )
        elif runtime.operating_context is not None:
            reasons.append(
                "commissioned reference has no operating_context binding; recommission or omit the new "
                "runtime context only under an explicit legacy policy"
            )
        if self.asset_scope is not None:
            reasons.extend(
                self.asset_scope.mismatch_reasons(
                    asset_id=runtime.runtime_asset_id,
                    fleet_group_id=runtime.runtime_fleet_group_id,
                )
            )
        return tuple(reasons)

    def compatible_with(self, runtime: "ReferenceCompatibilityContract") -> bool:
        return not self.mismatch_reasons(runtime)

    def to_dict(self) -> dict[str, Any]:
        return {
            "modality_signature": [list(x) for x in self.modality_signature],
            "provider_fingerprints": sorted(self.provider_fingerprints),
            "source_domain": None if self.source_domain is None else self.source_domain.value,
            "asset_scope": None if self.asset_scope is None else self.asset_scope.to_dict(),
            "operating_context": None if self.operating_context is None else self.operating_context.to_dict(),
            # runtime identity is intentionally not persisted as part of the commissioned reference.
        }

    def to_runtime_dict(self) -> dict[str, Any]:
        """Diagnostic/report representation including actual runtime asset identity."""
        out = self.to_dict()
        out["runtime_asset_id"] = self.runtime_asset_id
        out["runtime_fleet_group_id"] = self.runtime_fleet_group_id
        return out

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any] | None) -> "ReferenceCompatibilityContract | None":
        if obj is None:
            return None
        domain = obj.get("source_domain")
        return cls(
            modality_signature=tuple((str(x[0]), str(x[1])) for x in obj.get("modality_signature", [])),
            provider_fingerprints=frozenset(str(x) for x in obj.get("provider_fingerprints", [])),
            source_domain=None if domain is None else SourceDomain(str(domain)),
            asset_scope=AssetReferenceScope.from_dict(obj.get("asset_scope")),
            operating_context=OperatingContextIdentity.from_dict(obj.get("operating_context")),
        )


@dataclass(frozen=True)
class LineageOverlap:
    left_source_id: str
    right_source_id: str
    shared_upstream_sources: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceLineageAudit:
    overlaps: tuple[LineageOverlap, ...]
    cycles: tuple[tuple[str, ...], ...]
    unresolved_parents: tuple[str, ...]
    valid_dag: bool
    automatic_equivalence_assignments: int = 0


def audit_evidence_lineage(provenance: Sequence[EvidenceProvenance]) -> EvidenceLineageAudit:
    """Audit shared ancestry without changing Contact equivalence/topology.

    Shared lineage is a *review signal*, not proof of semantic equivalence.  This
    function therefore never assigns an equivalence id automatically.
    """
    by_id: dict[str, EvidenceProvenance] = {}
    for p in provenance:
        if p.source_id in by_id:
            raise ValueError(f"duplicate provenance source_id: {p.source_id!r}")
        by_id[p.source_id] = p

    unresolved = sorted({parent for p in provenance for parent in p.parents if parent not in by_id})
    visiting: list[str] = []
    visit_set: set[str] = set()
    done: set[str] = set()
    cycles: list[tuple[str, ...]] = []

    def walk_cycle(node: str) -> None:
        if node in done:
            return
        if node in visit_set:
            idx = visiting.index(node)
            cyc = tuple(visiting[idx:] + [node])
            if cyc not in cycles:
                cycles.append(cyc)
            return
        visit_set.add(node)
        visiting.append(node)
        for parent in by_id[node].parents:
            if parent in by_id:
                walk_cycle(parent)
        visiting.pop()
        visit_set.remove(node)
        done.add(node)

    for sid in by_id:
        walk_cycle(sid)

    # Ancestors are still useful when no cycle exists.  On cyclic metadata,
    # return the cycle and suppress overlap inference rather than recurse forever.
    if cycles:
        return EvidenceLineageAudit((), tuple(cycles), tuple(unresolved), False, 0)

    memo: dict[str, frozenset[str]] = {}

    def ancestors(node: str) -> frozenset[str]:
        if node in memo:
            return memo[node]
        out = {node}
        for parent in by_id[node].parents:
            out.add(parent)
            if parent in by_id:
                out.update(ancestors(parent))
        memo[node] = frozenset(out)
        return memo[node]

    ids = list(by_id)
    overlaps: list[LineageOverlap] = []
    for i, left in enumerate(ids):
        a = ancestors(left)
        for right in ids[i + 1 :]:
            shared = tuple(sorted(a & ancestors(right)))
            if shared:
                overlaps.append(LineageOverlap(left, right, shared))

    return EvidenceLineageAudit(tuple(overlaps), (), tuple(unresolved), True, 0)


@dataclass(frozen=True)
class SourceAvailabilityState:
    """Resolved runtime availability for one provenance source.

    `effective_status` describes the modality semantics of the source that can
    actually be consumed at runtime.  An unavailable source is always reported
    as MISSING, even when a stale/precomputed value happens to exist on disk.
    """

    source_id: str
    modality_id: str
    asset_id: str
    available: bool
    effective_status: ModalityStatus
    provider_fingerprints: tuple[str, ...] = ()
    blocking_sources: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "modality_id": self.modality_id,
            "asset_id": self.asset_id,
            "available": self.available,
            "effective_status": self.effective_status.value,
            "provider_fingerprints": list(self.provider_fingerprints),
            "blocking_sources": list(self.blocking_sources),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class EvidenceAvailabilityResolution:
    """Fail-closed runtime resolution of a provenance DAG.

    The resolver does not choose replacement providers automatically.  It only
    determines whether each declared source can be produced from the currently
    available observed modalities and explicitly available external sources.
    A caller must explicitly select which source supplies each modality.
    """

    states: tuple[SourceAvailabilityState, ...]
    valid_dag: bool
    unresolved_parents: tuple[str, ...] = ()
    cycles: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        ids = [s.source_id for s in self.states]
        if len(ids) != len(set(ids)):
            raise ValueError("availability resolution contains duplicate source ids")

    @property
    def by_source(self) -> dict[str, SourceAvailabilityState]:
        return {s.source_id: s for s in self.states}

    def state(self, source_id: str) -> SourceAvailabilityState:
        try:
            return self.by_source[str(source_id)]
        except KeyError as exc:
            raise KeyError(f"source not present in availability resolution: {source_id!r}") from exc

    def materialize_modality_availability(
        self,
        selected_sources: Mapping[str, str | None],
    ) -> ModalityAvailability:
        """Build an effective modality signature from explicitly selected sources.

        Selection is intentionally explicit: FEC must never silently pick a
        replacement source because it happens to preserve normality.  A `None`
        source means the modality is intentionally missing.  Selecting a blocked
        source also materializes as MISSING, allowing a compatibility gate to
        reject the configuration safely.
        """
        if not selected_sources:
            raise ValueError("selected_sources must be non-empty")
        by_id = self.by_source
        states: dict[str, ModalityStatus] = {}
        for modality_id, source_id in selected_sources.items():
            modality = str(modality_id)
            if not modality:
                raise ValueError("selected modality ids must be non-empty")
            if source_id is None:
                states[modality] = ModalityStatus.MISSING
                continue
            sid = str(source_id)
            if sid not in by_id:
                raise KeyError(f"selected source not present in resolution: {sid!r}")
            state = by_id[sid]
            if state.modality_id != modality:
                raise ValueError(
                    f"selected source {sid!r} provides modality {state.modality_id!r}, "
                    f"not {modality!r}"
                )
            states[modality] = state.effective_status if state.available else ModalityStatus.MISSING
        return ModalityAvailability.from_mapping(states)

    def reference_compatibility(
        self,
        selected_sources: Mapping[str, str | None],
        *,
        source_domain: SourceDomain | str | None,
        additional_provider_fingerprints: Iterable[str] = (),
        require_selected_available: bool = True,
        asset_id: str | None = None,
        fleet_group_id: str | None = None,
        operating_context: OperatingContextIdentity | None = None,
    ) -> ReferenceCompatibilityContract:
        """Construct a compatibility contract from resolved lineage.

        Provider fingerprints include the selected provider and all resolved
        upstream providers.  This closes the gap where a downstream provider is
        unchanged but an upstream virtual/learned provider has been replaced.
        """
        availability = self.materialize_modality_availability(selected_sources)
        by_id = self.by_source
        fps = {str(x) for x in additional_provider_fingerprints}
        unavailable: list[str] = []
        selected_assets: set[str] = set()
        for source_id in selected_sources.values():
            if source_id is None:
                continue
            state = by_id[str(source_id)]
            selected_assets.add(state.asset_id)
            if not state.available:
                unavailable.append(state.source_id)
            fps.update(state.provider_fingerprints)
        if require_selected_available and unavailable:
            raise ValueError(f"selected runtime sources are unavailable: {sorted(unavailable)}")
        if len(selected_assets) > 1:
            raise ValueError(
                "selected runtime sources span multiple asset_ids: " + repr(sorted(selected_assets))
            )
        derived_asset_id = next(iter(selected_assets), None)
        if asset_id is not None and derived_asset_id is not None and str(asset_id) != derived_asset_id:
            raise ValueError(
                f"explicit runtime asset_id {str(asset_id)!r} does not match selected-source asset_id "
                f"{derived_asset_id!r}"
            )
        runtime_asset_id = derived_asset_id if asset_id is None else str(asset_id)
        return ReferenceCompatibilityContract.from_runtime(
            availability=availability,
            provider_fingerprints=fps,
            source_domain=source_domain,
            operating_context=operating_context,
            asset_id=runtime_asset_id,
            fleet_group_id=fleet_group_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid_dag": self.valid_dag,
            "unresolved_parents": list(self.unresolved_parents),
            "cycles": [list(x) for x in self.cycles],
            "states": [x.to_dict() for x in self.states],
        }


def resolve_evidence_availability(
    provenance: Sequence[EvidenceProvenance],
    observed_modalities: Mapping[str, str | ModalityStatus],
    *,
    available_external_sources: Iterable[str] = (),
    disabled_sources: Iterable[str] = (),
) -> EvidenceAvailabilityResolution:
    """Resolve source availability from lineage and physical modality state.

    Rules are deliberately conservative:

    * OBSERVED sources require their modality to be explicitly OBSERVED.
    * VIRTUAL/LEARNED/AUGMENTED/SIMULATED sources require every declared
      parent to be available (logical AND).
    * unresolved parents are unavailable unless explicitly listed in
      `available_external_sources`.
    * a cyclic lineage fails closed; no source in the cycle graph is usable.
    * derived values already materialized in a file/cache do not remain valid
      when an upstream source needed to produce them is unavailable.

    This is a Control-Plane resolver, not an imputation algorithm and not a
    provider selector.
    """
    audit = audit_evidence_lineage(provenance)
    by_id: dict[str, EvidenceProvenance] = {p.source_id: p for p in provenance}
    observed = {str(k): ModalityStatus(v) for k, v in observed_modalities.items()}
    external = frozenset(str(x) for x in available_external_sources)
    disabled = frozenset(str(x) for x in disabled_sources)

    if not audit.valid_dag:
        states = tuple(
            SourceAvailabilityState(
                p.source_id,
                p.modality_id,
                p.asset_id,
                False,
                ModalityStatus.MISSING,
                (),
                tuple(sorted(set(p.parents))),
                ("lineage graph is cyclic; availability resolution failed closed",),
            )
            for p in provenance
        )
        return EvidenceAvailabilityResolution(states, False, audit.unresolved_parents, audit.cycles)

    memo: dict[str, SourceAvailabilityState] = {}

    def resolve(sid: str) -> SourceAvailabilityState:
        if sid in memo:
            return memo[sid]
        p = by_id[sid]
        reasons: list[str] = []
        blocking: set[str] = set()
        parent_fps: set[str] = set()

        if sid in disabled:
            reasons.append("source/provider explicitly disabled")

        for parent in p.parents:
            if parent in by_id:
                parent_state = resolve(parent)
                if not parent_state.available:
                    blocking.add(parent)
                    blocking.update(parent_state.blocking_sources)
                else:
                    parent_fps.update(parent_state.provider_fingerprints)
            elif parent in external:
                # External source is declared available but has no FEC provider
                # identity in this provenance graph.
                continue
            else:
                blocking.add(parent)
                reasons.append(f"unresolved parent not declared available: {parent}")

        if p.source_kind is SourceKind.OBSERVED:
            declared = observed.get(p.modality_id)
            if declared is None:
                reasons.append(f"observed modality not declared at runtime: {p.modality_id}")
            elif declared is not ModalityStatus.OBSERVED:
                reasons.append(
                    f"observed source requires modality status 'observed'; actual={declared.value!r}"
                )
        # Non-observed providers are available by default only when their
        # lineage is satisfied and the provider has not been explicitly disabled.

        available = not reasons and not blocking
        if available:
            if p.source_kind is SourceKind.OBSERVED:
                effective = ModalityStatus.OBSERVED
            elif p.source_kind is SourceKind.SIMULATED:
                effective = ModalityStatus.SIMULATED
            else:
                effective = ModalityStatus.DERIVED
            fps = tuple(sorted(parent_fps | {p.provider.fingerprint}))
        else:
            effective = ModalityStatus.MISSING
            fps = ()
            if blocking and not reasons:
                reasons.append("one or more required upstream sources are unavailable")

        state = SourceAvailabilityState(
            source_id=p.source_id,
            modality_id=p.modality_id,
            asset_id=p.asset_id,
            available=available,
            effective_status=effective,
            provider_fingerprints=fps,
            blocking_sources=tuple(sorted(blocking)),
            reasons=tuple(reasons),
        )
        memo[sid] = state
        return state

    states = tuple(resolve(p.source_id) for p in provenance)
    return EvidenceAvailabilityResolution(states, True, audit.unresolved_parents, ())
