from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence
import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class DirichletOpinion:
    """Opinion-valued Contact for optional class-level fusion.

    This object is deliberately *not* constructed from scalar FEC anomaly E.
    An Adapter must provide non-negative class evidence with a justified class
    semantics. For K classes, alpha = evidence + 1, belief = evidence / sum(alpha),
    and uncertainty = K / sum(alpha).
    """
    evidence: np.ndarray
    source: str = ''

    def __post_init__(self):
        e = np.asarray(self.evidence, dtype=float)
        if e.ndim != 1 or len(e) < 2:
            raise ValueError('evidence must be a 1D vector with at least two classes')
        if not np.all(np.isfinite(e)) or np.any(e < 0):
            raise ValueError('Dirichlet evidence must be finite and non-negative')
        object.__setattr__(self, 'evidence', e.copy())

    @property
    def n_classes(self) -> int:
        return int(len(self.evidence))

    @property
    def alpha(self) -> np.ndarray:
        return self.evidence + 1.0

    @property
    def strength(self) -> float:
        return float(np.sum(self.alpha))

    @property
    def belief(self) -> np.ndarray:
        return self.evidence / self.strength

    @property
    def probability(self) -> np.ndarray:
        return self.alpha / self.strength

    @property
    def uncertainty(self) -> float:
        return float(self.n_classes / self.strength)


@dataclass(frozen=True)
class OpinionFusionResult:
    opinion: DirichletOpinion
    source_discount: np.ndarray
    pairwise_conflict: np.ndarray


def _validate_opinions(opinions: Sequence[DirichletOpinion]) -> int:
    if len(opinions) == 0:
        raise ValueError('at least one opinion is required')
    K = opinions[0].n_classes
    if any(o.n_classes != K for o in opinions):
        raise ValueError('all opinions must use the same class frame')
    return K


def pairwise_degree_of_conflict(a: DirichletOpinion, b: DirichletOpinion) -> float:
    """Conflict degree used by the AISTATS-2025 DBF family.

    Conflict combines total-variation distance between projected class
    probabilities with the confidence of both opinions.
    """
    if a.n_classes != b.n_classes:
        raise ValueError('class frames differ')
    probability_conflict = 0.5 * float(np.abs(a.probability - b.probability).sum())
    confidence = (1.0 - a.uncertainty) * (1.0 - b.uncertainty)
    return float(np.clip(probability_conflict * confidence, 0.0, 1.0))


def discounted_belief_fusion(
    opinions: Sequence[DirichletOpinion],
    *,
    conflict_power: float = 3.0,
) -> OpinionFusionResult:
    """Order-invariant conflict-discounted fusion for opinion-valued Contacts.

    Independent NumPy adaptation of the discounted-belief idea described by
    Bezirganyan et al. (AISTATS 2025). Each source is discounted according to
    its pairwise conflict with all sources; discounted belief mass is moved to
    uncertainty, then the discounted evidence vectors are averaged.

    This extension is intentionally outside FEC's scalar-E Core. It is valid
    only when the Adapter already has class-level non-negative evidence.
    """
    K = _validate_opinions(opinions)
    if conflict_power <= 0:
        raise ValueError('conflict_power must be > 0')
    M = len(opinions)
    if M == 1:
        return OpinionFusionResult(opinions[0], np.ones(1), np.zeros((1, 1)))

    conflict = np.zeros((M, M), dtype=float)
    for i in range(M):
        for j in range(i + 1, M):
            d = pairwise_degree_of_conflict(opinions[i], opinions[j])
            conflict[i, j] = conflict[j, i] = d

    # Generalized-product agreement. Self-conflict is zero and thus neutral.
    discount = np.ones(M, dtype=float)
    p = float(conflict_power)
    for i in range(M):
        terms = np.clip(1.0 - np.power(conflict[i] + 1e-6, p), EPS, 1.0)
        discount[i] = float(np.prod(np.power(terms, 1.0 / p)))

    discounted_evidence = []
    for o, d in zip(opinions, discount):
        b = o.belief * d
        u = o.uncertainty * d + (1.0 - d)
        # Convert opinion back to Dirichlet evidence: e = K*b/u.
        e = K * b / max(u, EPS)
        discounted_evidence.append(e)
    fused = np.mean(np.vstack(discounted_evidence), axis=0)
    return OpinionFusionResult(
        DirichletOpinion(fused, source='discounted_belief_fusion'),
        discount,
        conflict,
    )


def dempster_shafer_pair(a: DirichletOpinion, b: DirichletOpinion) -> DirichletOpinion:
    """Two-source Dempster-style combination for Dirichlet opinions.

    Kept as a baseline/catalog operator, not as the FEC default. High conflict
    can make the normalization denominator very small; callers should prefer a
    conflict-discounting operator when that behavior is undesirable.
    """
    K = _validate_opinions((a, b))
    ba, bb = a.belief, b.belief
    ua, ub = a.uncertainty, b.uncertainty
    outer = np.outer(ba, bb)
    C = float(outer.sum() - np.trace(outer))
    denom = 1.0 - C
    if denom <= EPS:
        raise ValueError('Dempster combination undefined at near-total conflict')
    fused_b = (ba * bb + ba * ub + bb * ua) / denom
    fused_u = (ua * ub) / denom
    e = K * fused_b / max(float(fused_u), EPS)
    return DirichletOpinion(e, source='dempster_shafer_pair')


def opinion_from_scalar_anomaly_evidence(*args, **kwargs):
    """Deliberate hard stop against arbitrary scalar-E -> belief conversion."""
    raise TypeError(
        'FEC scalar anomaly Evidence E is not a class-level Dirichlet opinion. '
        'Provide an Adapter with justified non-negative class evidence instead.'
    )
