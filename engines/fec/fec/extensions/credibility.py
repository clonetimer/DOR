from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CredibilityOverlay:
    """Optional per-sample credibility supplied by an external model.

    `A_contract` and `Q_credibility` are deliberately kept separate.
    TMC/C2MF-like learned credibility is useful prior art, but FEC must not
    silently redefine learned credibility as method-contract Applicability.
    """
    A_contract: np.ndarray
    Q_credibility: np.ndarray

    def __post_init__(self):
        a = np.asarray(self.A_contract, dtype=float)
        q = np.asarray(self.Q_credibility, dtype=float)
        if a.shape != q.shape:
            raise ValueError('A_contract and Q_credibility must have same shape')
        if np.any((a < 0) | (a > 1)) or np.any((q < 0) | (q > 1)):
            raise ValueError('A and Q must lie in [0,1]')

    @property
    def product_weight(self) -> np.ndarray:
        return np.asarray(self.A_contract, float) * np.asarray(self.Q_credibility, float)


def combine_contract_and_credibility(A_contract, Q_credibility, *, mode: str = 'separate'):
    """Return an optional effective weight without erasing semantic provenance.

    - `separate` (default): returns (A, Q); caller/policy sees both.
    - `product`: returns A*Q for an explicitly opted-in fusion experiment.

    The default intentionally avoids turning FEC A into an attention score.
    """
    a = np.clip(np.asarray(A_contract, dtype=float), 0.0, 1.0)
    q = np.clip(np.asarray(Q_credibility, dtype=float), 0.0, 1.0)
    if a.shape != q.shape:
        raise ValueError('A_contract and Q_credibility must have same shape')
    if mode == 'separate':
        return a, q
    if mode == 'product':
        return a * q
    raise KeyError(mode)
