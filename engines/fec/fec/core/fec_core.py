from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
from typing import Iterable, Sequence
import numpy as np

EPS = 1e-12

@dataclass(frozen=True)
class EmpiricalReference:
    center: np.ndarray
    deviations: tuple[np.ndarray, ...]

@dataclass(frozen=True)
class ContactSpec:
    key: str
    path: tuple[str, ...] = ()
    equivalence_id: str | None = None


@dataclass(frozen=True)
class RootCalibrationReference:
    scores: np.ndarray
    topology_signature: tuple
    scope: str
    dependency_set: frozenset[str]
    provenance: str

@dataclass
class EvidenceState:
    root_E: np.ndarray
    A_mean_present: float
    C_presence: float
    C_effective: float
    n_expected_contacts: int
    n_present_contacts: int
    n_equivalence_classes: int


def fit_reference(X: np.ndarray) -> EmpiricalReference:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError('fit_reference expects a 2D array with at least two rows')
    c = np.median(X, axis=0)
    dev = tuple(np.sort(np.abs(X[:, j] - c[j])) for j in range(X.shape[1]))
    return EmpiricalReference(c, dev)


def evidence(X: np.ndarray, ref: EmpiricalReference) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(ref.deviations):
        raise ValueError('evidence dimensionality mismatch')
    D = np.abs(X - ref.center)
    E = np.empty_like(D)
    for j, r in enumerate(ref.deviations):
        rank = np.searchsorted(r, D[:, j], side='left')
        p = (1.0 + (len(r) - rank)) / (len(r) + 1.0)
        E[:, j] = -np.log(p)
    return E


def empirical_upper_p(reference_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    r = np.sort(np.asarray(reference_scores, dtype=float))
    s = np.asarray(scores, dtype=float)
    if r.ndim != 1 or len(r) < 2:
        raise ValueError('reference_scores must be a 1D array with at least two values')
    rank = np.searchsorted(r, s, side='left')
    return (1.0 + (len(r) - rank)) / (len(r) + 1.0)


def support_p(reference_distances: np.ndarray, distances: np.ndarray) -> np.ndarray:
    return empirical_upper_p(reference_distances, distances)


def _combine(E: np.ndarray, A: np.ndarray, G: np.ndarray, operator: str) -> tuple[np.ndarray, float, float]:
    E = np.asarray(E, dtype=float)
    A = np.asarray(A, dtype=float)
    G = np.asarray(G, dtype=float)
    present = G > 0
    if E.ndim != 2 or E.shape[1] != len(A) or len(A) != len(G):
        raise ValueError('message shapes are inconsistent')
    if not np.any(present):
        return np.zeros(E.shape[0]), 0.0, 0.0
    w = np.where(present, np.clip(A, 0.0, 1.0), 0.0)
    if w.sum() <= EPS:
        return np.zeros(E.shape[0]), float(np.mean(A[present])), 1.0
    if operator == 'mean':
        z = (E * w).sum(axis=1) / w.sum()
    elif operator == 'lme2':
        tau = 2.0
        m = E[:, present].max(axis=1, keepdims=True)
        Ep = E[:, present]
        wp = w[present]
        z = tau * np.log((np.exp((Ep - m) / tau) * wp).sum(axis=1) / wp.sum()) + m[:, 0]
    elif operator == 'median':
        # weighted median is intentionally not implemented; plain median is valid only when A is uniform.
        if np.ptp(w[present]) > 1e-12:
            raise ValueError('median requires uniform applicability')
        z = np.median(E[:, present], axis=1)
    else:
        raise KeyError(operator)
    # A is propagated as a separate parent message; normalized local weighting must not consume and forget it.
    parent_A = float(np.mean(A[present]))
    return z, parent_A, 1.0


def _collapse_equivalence(
    E: np.ndarray,
    specs: Sequence[ContactSpec],
    A: np.ndarray,
    G: np.ndarray,
) -> tuple[np.ndarray, list[ContactSpec], np.ndarray, np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for j, s in enumerate(specs):
        eq = s.equivalence_id if s.equivalence_id is not None else f'__unique__:{j}:{s.key}'
        groups[eq].append(j)
    outE, outS, outA, outG = [], [], [], []
    for eq, js in groups.items():
        # Equivalent contacts are one semantic contact. Replication must not increase influence.
        arr = E[:, js]
        outE.append(arr.mean(axis=1))
        base = specs[js[0]]
        if any(specs[k].path != base.path for k in js):
            raise ValueError('equivalent contacts must share the same topology path')
        outS.append(ContactSpec(key=base.key, path=base.path, equivalence_id=eq))
        outA.append(float(np.mean(A[js])))
        outG.append(float(np.max(G[js])))
    return np.column_stack(outE), outS, np.asarray(outA), np.asarray(outG)


def coordinate(
    E: np.ndarray,
    specs: Sequence[ContactSpec],
    *,
    A: np.ndarray | None = None,
    G: np.ndarray | None = None,
    operators: Sequence[str] | None = None,
    expected_contact_count: int | None = None,
) -> EvidenceState:
    E = np.asarray(E, dtype=float)
    if E.ndim != 2 or E.shape[1] != len(specs):
        raise ValueError('E/spec dimensionality mismatch')
    n = len(specs)
    A0 = np.ones(n, dtype=float) if A is None else np.clip(np.asarray(A, dtype=float), 0.0, 1.0)
    G0 = np.ones(n, dtype=float) if G is None else (np.asarray(G, dtype=float) > 0).astype(float)
    if len(A0) != n or len(G0) != n:
        raise ValueError('A/G dimensionality mismatch')
    expected = int(expected_contact_count if expected_contact_count is not None else n)
    if expected <= 0:
        raise ValueError('expected_contact_count must be positive')
    C_presence = float(np.sum(G0) / expected)
    C_presence = float(np.clip(C_presence, 0.0, 1.0))
    C_effective = float(np.sum(G0 * A0) / expected)
    C_effective = float(np.clip(C_effective, 0.0, 1.0))
    A_mean = float(np.mean(A0[G0 > 0])) if np.any(G0 > 0) else 0.0

    curE, curS, curA, curG = _collapse_equivalence(E, specs, A0, G0)
    n_eq = len(curS)
    max_depth = max((len(s.path) for s in curS), default=0)
    ops = tuple(operators) if operators is not None else tuple('mean' for _ in range(max_depth + 1))
    if len(ops) != max_depth + 1:
        raise ValueError(f'operators must have length {max_depth + 1}')

    # Recursively consume path elements from leaf-side to root-side.
    for level in range(max_depth):
        buckets: dict[str, list[int]] = defaultdict(list)
        passthrough = []
        for j, s in enumerate(curS):
            if len(s.path) > 0:
                buckets[s.path[0]].append(j)
            else:
                passthrough.append(j)
        nextE, nextS, nextA, nextG = [], [], [], []
        for label, js in sorted(buckets.items()):
            z, a, g = _combine(curE[:, js], curA[js], curG[js], ops[level])
            remaining = curS[js[0]].path[1:]
            if any(curS[k].path[1:] != remaining for k in js):
                raise ValueError('children grouped at one node must share the same remaining path')
            nextE.append(z); nextS.append(ContactSpec(key=f'node:{level}:{label}', path=remaining))
            nextA.append(a); nextG.append(g)
        for j in passthrough:
            nextE.append(curE[:, j]); nextS.append(curS[j]); nextA.append(curA[j]); nextG.append(curG[j])
        curE = np.column_stack(nextE) if nextE else np.zeros((E.shape[0], 0))
        curS = nextS; curA = np.asarray(nextA); curG = np.asarray(nextG)

    if curE.shape[1] == 0:
        root = np.zeros(E.shape[0])
    else:
        root, _, _ = _combine(curE, curA, curG, ops[-1])
    return EvidenceState(
        root_E=root,
        A_mean_present=A_mean,
        C_presence=C_presence,
        C_effective=C_effective,
        n_expected_contacts=expected,
        n_present_contacts=int(np.sum(G0)),
        n_equivalence_classes=n_eq,
    )


def topology_signature(specs: Sequence[ContactSpec], A: np.ndarray | None = None, G: np.ndarray | None = None) -> tuple:
    n = len(specs)
    A0 = np.ones(n) if A is None else np.asarray(A, float)
    G0 = np.ones(n) if G is None else np.asarray(G, float)
    return tuple((s.path, s.equivalence_id, round(float(A0[i]), 8), int(G0[i] > 0)) for i, s in enumerate(specs))


def make_root_reference(
    scores: np.ndarray,
    specs: Sequence[ContactSpec],
    *,
    A: np.ndarray | None = None,
    G: np.ndarray | None = None,
    scope: str = 'root',
    dependency_set: Iterable[str] = (),
    provenance: str = 'trusted-normal',
) -> RootCalibrationReference:
    return RootCalibrationReference(
        scores=np.asarray(scores, dtype=float).copy(),
        topology_signature=topology_signature(specs, A, G),
        scope=str(scope),
        dependency_set=frozenset(str(x) for x in dependency_set),
        provenance=str(provenance),
    )


def calibrated_p(
    ref: RootCalibrationReference,
    scores: np.ndarray,
    specs: Sequence[ContactSpec],
    *,
    A: np.ndarray | None = None,
    G: np.ndarray | None = None,
) -> np.ndarray:
    sig = topology_signature(specs, A, G)
    if sig != ref.topology_signature:
        raise ValueError('root calibration topology mismatch')
    return empirical_upper_p(ref.scores, scores)


def calibration_admission(trusted_claims: Iterable[str], ref: RootCalibrationReference) -> bool:
    claims = frozenset(str(x) for x in trusted_claims)
    return ref.dependency_set.issubset(claims)
