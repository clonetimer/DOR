from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
from typing import Sequence
import numpy as np
from fec.core.fec_core import ContactSpec

@dataclass(frozen=True)
class RedundancyPair:
    i: int
    j: int
    key_i: str
    key_j: str
    abs_correlation: float

@dataclass(frozen=True)
class ContactBankAuditResult:
    branch_counts: dict[str, int]
    redundancy_pairs: tuple[RedundancyPair, ...]
    max_branch_count: int
    min_branch_count: int


def audit_contact_bank(X_normal: np.ndarray, specs: Sequence[ContactSpec], *, corr_threshold: float = 0.995) -> ContactBankAuditResult:
    """Audit a Contact bank for capacity imbalance and near-duplicate normal behaviour.

    Inspired by expert-indicator generation/selection literature. High
    correlation is only a review flag; it MUST NOT automatically create an
    `equivalence_id`, because statistical correlation is not semantic identity.
    """
    X=np.asarray(X_normal,float)
    if X.ndim != 2 or X.shape[1] != len(specs):
        raise ValueError('X_normal/spec dimensionality mismatch')
    if not (0 <= corr_threshold <= 1):
        raise ValueError('corr_threshold must be in [0,1]')
    labels=[s.path[0] if s.path else '__root__' for s in specs]
    counts=Counter(labels)
    # Constant columns are not treated as correlated; they need Eligibility review.
    C=np.full((X.shape[1],X.shape[1]),np.nan)
    std=np.std(X,axis=0)
    good=std>1e-12
    if np.sum(good)>=2:
        C[np.ix_(good,good)] = np.corrcoef(X[:,good],rowvar=False)
    pairs=[]
    for i in range(X.shape[1]):
        for j in range(i+1,X.shape[1]):
            c=C[i,j]
            if np.isfinite(c) and abs(c)>=corr_threshold:
                pairs.append(RedundancyPair(i,j,specs[i].key,specs[j].key,float(abs(c))))
    vals=list(counts.values()) or [0]
    return ContactBankAuditResult(dict(counts),tuple(pairs),max(vals),min(vals))
