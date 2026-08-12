from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np

from .class_evidence import ClassEvidenceBatch


@dataclass(frozen=True)
class EDLTrainingReport:
    epochs: int
    train_accuracy: float
    final_loss: float
    seed: int


class EvidentialMLPProvider:
    """Small PyTorch evidential classifier producing genuine class evidence.

    The network emits non-negative evidence via softplus and alpha=evidence+1.
    Training uses the expected Dirichlet cross-entropy (digamma) with an
    annealed KL-to-uniform penalty, following the standard EDL formulation
    popularized by Sensoy, Kaplan & Kandemir (NeurIPS 2018).

    This provider belongs to the optional diagnostic layer, not the Frozen FEC
    anomaly-E Core.
    """
    def __init__(self, class_frame: Sequence[str], *, hidden: int = 32, seed: int = 0, source: str = 'edl_mlp'):
        self.class_frame = tuple(str(x) for x in class_frame)
        if len(self.class_frame) < 2:
            raise ValueError('at least two classes are required')
        self.hidden = int(hidden); self.seed = int(seed); self.source = str(source)
        self.model = None; self.mean = None; self.std = None; self.report = None

    @staticmethod
    def _kl_divergence(alpha, K, torch):
        ones = torch.ones((1, K), dtype=alpha.dtype, device=alpha.device)
        S = torch.sum(alpha, dim=1, keepdim=True)
        first = torch.lgamma(S) - torch.lgamma(alpha).sum(1, keepdim=True) + torch.lgamma(ones).sum(1, keepdim=True) - torch.lgamma(ones.sum(1, keepdim=True))
        second = ((alpha - ones) * (torch.digamma(alpha) - torch.digamma(S))).sum(1, keepdim=True)
        return first + second

    def fit(self, X, y, *, epochs: int = 120, lr: float = 1e-2, annealing_step: int = 40, weight_decay: float = 1e-4) -> EDLTrainingReport:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        X = np.asarray(X, float); y = np.asarray(y, int)
        if X.ndim != 2 or y.shape != (len(X),):
            raise ValueError('X must be 2D and y one integer per row')
        K = len(self.class_frame)
        if np.any((y < 0) | (y >= K)):
            raise ValueError('labels outside class frame')
        self.mean = X.mean(0); self.std = X.std(0); self.std = np.where(self.std > 1e-12, self.std, 1.0)
        Z = (X - self.mean) / self.std
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        model = nn.Sequential(nn.Linear(X.shape[1], self.hidden), nn.ReLU(), nn.Linear(self.hidden, K))
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        xt = torch.tensor(Z, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.long); one = F.one_hot(yt, K).float()
        last = float('nan')
        for ep in range(1, int(epochs) + 1):
            opt.zero_grad(); out = model(xt); evidence = F.softplus(out); alpha = evidence + 1.0; S = alpha.sum(1, keepdim=True)
            A = torch.sum(one * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
            anneal = min(1.0, ep / float(max(1, annealing_step)))
            kl_alpha = (alpha - 1.0) * (1.0 - one) + 1.0
            loss = torch.mean(A + anneal * self._kl_divergence(kl_alpha, K, torch))
            loss.backward(); opt.step(); last = float(loss.detach())
        with torch.no_grad():
            ev = F.softplus(model(xt)); pred = torch.argmax(ev + 1.0, dim=1).cpu().numpy()
        self.model = model.eval()
        self.report = EDLTrainingReport(int(epochs), float(np.mean(pred == y)), last, self.seed)
        return self.report

    def evidence(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError('provider is not fitted')
        import torch
        import torch.nn.functional as F
        X = np.asarray(X, float)
        if X.ndim == 1: X = X[None, :]
        Z = (X - self.mean) / self.std
        with torch.no_grad():
            e = F.softplus(self.model(torch.tensor(Z, dtype=torch.float32))).cpu().numpy()
        return e

    def batch(self, X, *, provenance: str) -> ClassEvidenceBatch:
        return ClassEvidenceBatch(self.evidence(X), self.class_frame, self.source, provenance)
