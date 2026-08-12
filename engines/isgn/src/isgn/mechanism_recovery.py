"""Core utilities for recovering missing mechanism terms from partial physics.

The key contract is deliberately small::

    observed dynamics = known mechanism + missing mechanism

Known terms are kept fixed. Only the residual is searched over a finite,
interpretable term library. This is complementary to full-equation SINDy and
external symbolic-regression backends; it is not an implementation of UDE or
APHYNITY.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Mapping, Sequence
import re
import numpy as np
from sklearn.linear_model import Lasso

from .contracts import PublicDataset


_TERM_RE = re.compile(r"^(sin|cos|square|cube)\(([^(),]+)\)$")
_PROD_RE = re.compile(r"^product\(([^(),]+),([^(),]+)\)$")


@dataclass
class MissingTermRecoveryResult:
    target: str
    known_terms: Dict[str, float]
    selected_missing_terms: List[str]
    missing_coefficients: Dict[str, float]
    full_coefficients: Dict[str, float]
    validation_residual_nrmse: float
    alpha: float
    candidate_terms: List[str]
    backend: str = "residual-lasso"
    metadata: Dict[str, object] | None = None

    def to_dict(self):
        return asdict(self)


def _variables(ds: PublicDataset) -> Dict[str, np.ndarray]:
    ds.validate()
    out: Dict[str, np.ndarray] = {}
    xm = 0.5 * (np.asarray(ds.X[:, :-1], float) + np.asarray(ds.X[:, 1:], float))
    for j, meta in enumerate(ds.schema.get("states", [])):
        out[str(meta["name"])] = xm[:, :, j].reshape(-1)
    for j, meta in enumerate(ds.schema.get("inputs", [])):
        out[str(meta["name"])] = np.asarray(ds.U[:, :, j], float).reshape(-1)
    return out


def evaluate_term(ds: PublicDataset, term: str) -> np.ndarray:
    """Evaluate a small device-agnostic mechanism term on midpoint samples.

    Supported syntax: ``x``, ``sin(x)``, ``cos(x)``, ``square(x)``,
    ``cube(x)``, ``product(x,y)``, and ``__constant__``.
    """
    term = str(term).strip()
    vars = _variables(ds)
    if term == "__constant__":
        n = ds.X.shape[0] * (ds.X.shape[1] - 1)
        return np.ones(n, dtype=float)
    if term in vars:
        return vars[term]
    m = _TERM_RE.match(term)
    if m:
        op, name = m.group(1), m.group(2).strip()
        if name not in vars:
            raise KeyError(f"unknown variable {name!r} in term {term!r}")
        x = vars[name]
        if op == "sin": return np.sin(x)
        if op == "cos": return np.cos(x)
        if op == "square": return x * x
        if op == "cube": return x * x * x
    m = _PROD_RE.match(term)
    if m:
        a, b = m.group(1).strip(), m.group(2).strip()
        if a not in vars or b not in vars:
            raise KeyError(f"unknown variable in product term {term!r}")
        return vars[a] * vars[b]
    raise ValueError(f"unsupported mechanism term syntax: {term!r}")


def target_derivative(ds: PublicDataset, target: str) -> np.ndarray:
    names = [str(x["name"]) for x in ds.schema.get("states", [])]
    if target not in names:
        raise KeyError(f"target {target!r} is not a state")
    j = names.index(target)
    return ((np.asarray(ds.X[:, 1:, j], float) - np.asarray(ds.X[:, :-1, j], float)) / float(ds.dt)).reshape(-1)


def _known_prediction(ds: PublicDataset, known_terms: Mapping[str, float]) -> np.ndarray:
    n = ds.X.shape[0] * (ds.X.shape[1] - 1)
    out = np.zeros(n, dtype=float)
    for term, coeff in known_terms.items():
        out += float(coeff) * evaluate_term(ds, str(term))
    return out


def recover_missing_terms(
    train: PublicDataset,
    validation: PublicDataset,
    *,
    target: str,
    known_terms: Mapping[str, float] | None = None,
    candidate_terms: Sequence[str],
    alphas: Iterable[float] = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2),
    support_threshold: float = 1e-5,
    complexity_penalty: float = 5e-4,
    max_iter: int = 50000,
    tol: float = 1e-9,
) -> MissingTermRecoveryResult:
    """Recover only the dynamics unexplained by the supplied known mechanism.

    The physical ``known_terms`` are never re-optimized. Candidate selection is
    performed in a scale-normalized sparse regression space and coefficients are
    mapped back to physical units.
    """
    train.validate(); validation.validate()
    known = {str(k): float(v) for k, v in (known_terms or {}).items()}
    candidates = [str(x) for x in candidate_terms]
    if not candidates:
        raise ValueError("candidate_terms must not be empty")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate_terms must be unique")
    overlap = sorted(set(candidates) & set(known))
    if overlap:
        raise ValueError(f"candidate terms overlap fixed known terms: {overlap}")

    y = target_derivative(train, target) - _known_prediction(train, known)
    vy = target_derivative(validation, target) - _known_prediction(validation, known)
    A = np.stack([evaluate_term(train, t) for t in candidates], axis=1)
    V = np.stack([evaluate_term(validation, t) for t in candidates], axis=1)
    good = np.isfinite(A).all(axis=1) & np.isfinite(y)
    vgood = np.isfinite(V).all(axis=1) & np.isfinite(vy)
    A, y, V, vy = A[good], y[good], V[vgood], vy[vgood]
    if len(A) < max(20, 2 * len(candidates)) or len(V) < 10:
        raise ValueError("insufficient finite samples for missing-term recovery")

    xs = np.sqrt(np.mean(A * A, axis=0)); xs = np.maximum(xs, 1e-12)
    ys = max(float(np.std(y)), 1e-12)
    An, Vn, yn = A / xs, V / xs, y / ys
    best = None
    for alpha in alphas:
        m = Lasso(alpha=float(alpha), fit_intercept=False, max_iter=int(max_iter), tol=float(tol), selection="cyclic")
        m.fit(An, yn)
        coef_n = np.asarray(m.coef_, float)
        pred = Vn @ coef_n * ys
        nrmse = float(np.sqrt(np.mean((vy - pred) ** 2)) / (np.std(vy) + 1e-12))
        support = np.abs(coef_n) >= float(support_threshold)
        score = nrmse + float(complexity_penalty) * int(support.sum())
        key = (score, int(support.sum()), float(alpha))
        if best is None or key < best[0]:
            best = (key, float(alpha), coef_n, nrmse, support)
    _, alpha, coef_n, nrmse, support = best
    coef = coef_n * ys / xs
    missing = {candidates[i]: float(coef[i]) for i in range(len(candidates)) if bool(support[i])}
    full = dict(known); full.update(missing)
    return MissingTermRecoveryResult(
        target=str(target), known_terms=known,
        selected_missing_terms=list(missing), missing_coefficients=missing,
        full_coefficients=full, validation_residual_nrmse=float(nrmse),
        alpha=float(alpha), candidate_terms=candidates,
        metadata={
            "known_terms_frozen": True,
            "feature_scaling": "rms",
            "target_scaling": "std",
            "train_samples": int(len(A)),
            "validation_samples": int(len(V)),
            "interpretation": "missing-dynamics residual isolation; not UDE/APHYNITY implementation",
        },
    )
