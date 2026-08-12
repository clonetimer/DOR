from __future__ import annotations

"""Durable, non-executable storage for commissioned FEC Reference Bundles.

The v1.6 ReferenceBundleManager made leaf + root calibration atomic in memory.
This module closes the restart gap for the stable production path by storing
validated active/retired bundles as canonical JSON with an integrity digest.

Security / reliability properties:
- no pickle / eval / dynamic code loading;
- explicit schema version;
- SHA-256 integrity check over canonical payload bytes;
- atomic replace on save;
- only the stable CoreEmpiricalLeafProvider + EmpiricalRootBackend are accepted;
- shadow/rejected candidates are intentionally not persisted and must be
  restaged/revalidated after restart.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Hashable, Mapping
import hashlib
import json
import os
import tempfile

import numpy as np

from fec.core.fec_core import ContactSpec, EmpiricalReference
from .conformal_backend import EmpiricalRootBackend
from .production_calibration import CalibrationReferenceMeta
from .evidence_provenance import ReferenceCompatibilityContract
from .reference_bundle import (
    CoreEmpiricalLeafProvider,
    ManagedReferenceBundle,
    ReferenceBundleManager,
    ReferenceBundleMeta,
)

SCHEMA_VERSION = "fec.reference_bundle_store.v1"


class ReferenceBundleStoreError(RuntimeError):
    pass


class ReferenceBundleIntegrityError(ReferenceBundleStoreError):
    pass


def _canonical_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _json_scalar(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    return x


def _encode_key(key: Hashable) -> Mapping[str, Any]:
    key = _json_scalar(key)
    if key is None:
        return {"type": "none", "value": None}
    if isinstance(key, bool):
        return {"type": "bool", "value": key}
    if isinstance(key, int) and not isinstance(key, bool):
        return {"type": "int", "value": key}
    if isinstance(key, float):
        if not np.isfinite(key):
            raise TypeError("reference key float must be finite")
        return {"type": "float", "value": key}
    if isinstance(key, str):
        return {"type": "str", "value": key}
    if isinstance(key, tuple):
        return {"type": "tuple", "value": [_encode_key(v) for v in key]}
    raise TypeError(f"unsupported persistent reference key type: {type(key).__name__}")


def _decode_key(obj: Mapping[str, Any]) -> Hashable:
    t = obj.get("type")
    v = obj.get("value")
    if t == "none":
        return None
    if t == "bool":
        return bool(v)
    if t == "int":
        return int(v)
    if t == "float":
        f = float(v)
        if not np.isfinite(f):
            raise ReferenceBundleStoreError("stored reference key float is non-finite")
        return f
    if t == "str":
        return str(v)
    if t == "tuple":
        if not isinstance(v, list):
            raise ReferenceBundleStoreError("tuple key payload must be a list")
        return tuple(_decode_key(x) for x in v)
    raise ReferenceBundleStoreError(f"unsupported stored reference key type: {t!r}")


def _find_root_reference(manager: ReferenceBundleManager, key: Hashable, reference_id: str):
    root = manager._root
    current = root._active.get(key)
    if current is not None and current.meta.reference_id == reference_id:
        return current
    for ref in root._retired.get(key, []):
        if ref.meta.reference_id == reference_id:
            return ref
    raise ReferenceBundleStoreError(f"root reference {reference_id!r} not found for bundle key {key!r}")


def _serialize_bundle(manager: ReferenceBundleManager, bundle: ManagedReferenceBundle) -> Mapping[str, Any]:
    if not isinstance(manager.leaf_provider, CoreEmpiricalLeafProvider):
        raise ReferenceBundleStoreError(
            f"durable store supports CoreEmpiricalLeafProvider only; got {type(manager.leaf_provider).__name__}"
        )
    leaf = bundle.leaf_reference
    if not isinstance(leaf, EmpiricalReference):
        raise ReferenceBundleStoreError(
            f"durable store supports EmpiricalReference leaf state only; got {type(leaf).__name__}"
        )
    root_id = f"{bundle.meta.bundle_id}:root"
    root_ref = _find_root_reference(manager, bundle.meta.key, root_id)
    if not isinstance(root_ref.backend, EmpiricalRootBackend):
        raise ReferenceBundleStoreError(
            f"durable store supports EmpiricalRootBackend only; got {type(root_ref.backend).__name__}"
        )
    if root_ref.backend._scores is None:
        raise ReferenceBundleStoreError("empirical root backend is not fitted")
    meta = bundle.meta
    return {
        "meta": {
            "bundle_id": meta.bundle_id,
            "key": _encode_key(meta.key),
            "provenance": meta.provenance,
            "dependency_set": sorted(meta.dependency_set),
            "leaf_data_ids": sorted(meta.leaf_data_ids),
            "root_data_ids": sorted(meta.root_data_ids),
            "scope": meta.scope,
            "notes": meta.notes,
            "compatibility": None if meta.compatibility is None else meta.compatibility.to_dict(),
        },
        "leaf_provider": manager.leaf_provider.name,
        "leaf_reference": {
            "center": np.asarray(leaf.center, dtype=float).tolist(),
            "deviations": [np.asarray(x, dtype=float).tolist() for x in leaf.deviations],
        },
        "specs": [
            {"key": s.key, "path": list(s.path), "equivalence_id": s.equivalence_id}
            for s in bundle.specs
        ],
        "A": np.asarray(bundle.A, dtype=float).tolist(),
        "G": np.asarray(bundle.G, dtype=float).tolist(),
        "operators": None if bundle.operators is None else list(bundle.operators),
        "expected_contact_count": int(bundle.expected_contact_count),
        "lifecycle_state": bundle.lifecycle_state,
        "health_status": bundle.health_status,
        "generation": int(bundle.generation),
        "power_limited": bool(bundle.power_limited),
        "root": {
            "backend": root_ref.backend.name,
            "scores": np.asarray(root_ref.backend._scores, dtype=float).tolist(),
            "health_status": root_ref.health_status,
            "generation": int(root_ref.generation),
            "validation_supported_fraction": root_ref.validation_supported_fraction,
        },
    }


def export_reference_bundle_manager(manager: ReferenceBundleManager) -> Mapping[str, Any]:
    """Return a JSON-safe snapshot of active/retired commissioned bundles.

    Shadow/rejected candidates are deliberately excluded. They have not become
    production state and must be restaged/revalidated after restart.
    """
    root = manager._root
    settings = {
        "alpha": root.alpha,
        "min_sentinel_samples": root.min_sentinel_samples,
        "min_supported_fraction": root.min_supported_fraction,
        "sentinel_block_size": root.sentinel_block_size,
        "sentinel_bootstrap_reps": root.sentinel_bootstrap_reps,
        "random_state": root.random_state,
        "require_disjoint_sentinel": root.require_disjoint_sentinel,
    }
    entries = []
    for key in manager._active.keys() | manager._retired.keys():
        entries.append({
            "key": _encode_key(key),
            "retired": [_serialize_bundle(manager, b) for b in manager._retired.get(key, [])],
            "active": None if key not in manager._active else _serialize_bundle(manager, manager._active[key]),
        })
    entries.sort(key=lambda x: json.dumps(x["key"], sort_keys=True, ensure_ascii=False))
    return {
        "schema_version": SCHEMA_VERSION,
        "manager_settings": settings,
        "leaf_provider": manager.leaf_provider.name,
        "bundles": entries,
        "history": list(manager.history),
    }


def save_reference_bundle_manager(manager: ReferenceBundleManager, path: str | os.PathLike[str]) -> str:
    """Atomically save commissioned bundle state and return payload SHA-256."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = export_reference_bundle_manager(manager)
    digest = _sha256_payload(payload)
    envelope = {
        "format": "FEC Reference Bundle Store",
        "schema_version": SCHEMA_VERSION,
        "payload_sha256": digest,
        "payload": payload,
    }
    data = json.dumps(envelope, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return digest


def _deserialize_bundle(obj: Mapping[str, Any]) -> tuple[ManagedReferenceBundle, EmpiricalRootBackend, int, float | None]:
    m = obj["meta"]
    key = _decode_key(m["key"])
    meta = ReferenceBundleMeta(
        bundle_id=str(m["bundle_id"]),
        key=key,
        provenance=str(m["provenance"]),
        dependency_set=frozenset(str(x) for x in m["dependency_set"]),
        leaf_data_ids=frozenset(str(x) for x in m["leaf_data_ids"]),
        root_data_ids=frozenset(str(x) for x in m["root_data_ids"]),
        scope=str(m.get("scope", "leaf+root")),
        notes=str(m.get("notes", "")),
        compatibility=ReferenceCompatibilityContract.from_dict(m.get("compatibility")),
    )
    lr = obj["leaf_reference"]
    center = np.asarray(lr["center"], dtype=float)
    deviations = tuple(np.asarray(x, dtype=float) for x in lr["deviations"])
    if center.ndim != 1 or len(deviations) != len(center):
        raise ReferenceBundleStoreError("invalid persisted leaf reference dimensions")
    if any(x.ndim != 1 or len(x) < 2 for x in deviations):
        raise ReferenceBundleStoreError("invalid persisted leaf deviation arrays")
    if not np.all(np.isfinite(center)) or any(not np.all(np.isfinite(x)) for x in deviations):
        raise ReferenceBundleStoreError("persisted leaf reference must be finite")
    specs = tuple(
        ContactSpec(str(s["key"]), tuple(str(x) for x in s.get("path", [])), s.get("equivalence_id"))
        for s in obj["specs"]
    )
    A = np.asarray(obj["A"], dtype=float)
    G = np.asarray(obj["G"], dtype=float)
    if len(A) != len(specs) or len(G) != len(specs):
        raise ReferenceBundleStoreError("persisted A/G dimensions do not match specs")
    leaf = EmpiricalReference(center=center, deviations=deviations)
    bundle = ManagedReferenceBundle(
        meta=meta,
        leaf_reference=leaf,
        specs=specs,
        A=A,
        G=G,
        operators=None if obj.get("operators") is None else tuple(str(x) for x in obj["operators"]),
        expected_contact_count=int(obj["expected_contact_count"]),
        lifecycle_state=str(obj.get("lifecycle_state", "active")),
        health_status=str(obj.get("health_status", "unknown")),
        generation=int(obj.get("generation", 0)),
        validated_for_promotion=False,
        power_limited=bool(obj.get("power_limited", False)),
    )
    root = obj["root"]
    if root.get("backend") != "empirical":
        raise ReferenceBundleStoreError(f"unsupported persisted backend: {root.get('backend')!r}")
    backend = EmpiricalRootBackend().fit(np.asarray(root["scores"], dtype=float), specs, A=A, G=G)
    return bundle, backend, int(root.get("generation", 0)), root.get("validation_supported_fraction")


def load_reference_bundle_manager(path: str | os.PathLike[str]) -> ReferenceBundleManager:
    """Load and integrity-check a commissioned ReferenceBundleManager."""
    source = Path(path)
    try:
        envelope = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReferenceBundleStoreError(f"cannot read reference bundle store: {exc}") from exc
    if envelope.get("schema_version") != SCHEMA_VERSION:
        raise ReferenceBundleStoreError(
            f"unsupported reference bundle store schema: {envelope.get('schema_version')!r}"
        )
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise ReferenceBundleStoreError("reference bundle store payload is missing")
    actual = _sha256_payload(payload)
    expected = str(envelope.get("payload_sha256", ""))
    if not expected or actual != expected:
        raise ReferenceBundleIntegrityError(
            f"reference bundle store integrity check failed: expected={expected!r}, actual={actual!r}"
        )
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ReferenceBundleStoreError("payload schema version mismatch")
    if payload.get("leaf_provider") != CoreEmpiricalLeafProvider.name:
        raise ReferenceBundleStoreError(f"unsupported leaf provider: {payload.get('leaf_provider')!r}")
    settings = dict(payload.get("manager_settings", {}))
    manager = ReferenceBundleManager(leaf_provider=CoreEmpiricalLeafProvider(), backend_factory=EmpiricalRootBackend, **settings)
    max_bundle_generation = 0
    max_root_generation = 0
    for entry in payload.get("bundles", []):
        key = _decode_key(entry["key"])
        chain = list(entry.get("retired", []))
        if entry.get("active") is not None:
            chain.append(entry["active"])
        loaded: list[ManagedReferenceBundle] = []
        for bobj in chain:
            bundle, backend, stored_root_generation, stored_supported_fraction = _deserialize_bundle(bobj)
            if bundle.meta.key != key:
                raise ReferenceBundleStoreError("bundle entry key does not match bundle metadata key")
            root_meta = CalibrationReferenceMeta(
                reference_id=f"{bundle.meta.bundle_id}:root",
                key=key,
                provenance=bundle.meta.provenance,
                dependency_set=bundle.meta.dependency_set,
                scope=bundle.meta.scope,
                notes=f"restored atomic reference bundle; leaf_provider={manager.leaf_provider.name}; {bundle.meta.notes}".strip(),
                reference_data_id="|".join(sorted(bundle.meta.reference_data_ids)),
            )
            manager._root.install_active(
                root_meta,
                backend,
                bundle.specs,
                A=bundle.A,
                G=bundle.G,
                health_status=bundle.health_status,
            )
            r = manager._root._active[key]
            r.generation = stored_root_generation
            r.validation_supported_fraction = stored_supported_fraction
            max_root_generation = max(max_root_generation, stored_root_generation)
            loaded.append(bundle)
            max_bundle_generation = max(max_bundle_generation, bundle.generation)
        if not loaded:
            continue
        active_expected = entry.get("active") is not None
        if active_expected:
            manager._active[key] = loaded[-1]
            manager._active[key].lifecycle_state = "active"
            retired = loaded[:-1]
        else:
            # A key without an active bundle is not a valid commissioned state.
            raise ReferenceBundleStoreError(f"persisted key {key!r} has no active bundle")
        manager._retired[key] = retired
        for b in retired:
            b.lifecycle_state = "retired"
        # install_active already built the root retired stack in the same order.
    manager._generation = max_bundle_generation
    manager._root._generation = max_root_generation
    history = payload.get("history", [])
    if isinstance(history, list):
        manager._history = list(history)
    manager._history.append({"event": "load_persistent_state", "path": str(source), "payload_sha256": actual})
    return manager
