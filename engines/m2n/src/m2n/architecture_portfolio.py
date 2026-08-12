from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Mapping
import torch
from torch import nn


@dataclass(frozen=True)
class PortfolioContract:
    """Frozen finite M2N architecture set used when K2 cannot be uniquely resolved.

    The portfolio is not a hidden K3 mixture and does not claim that averaging
    produces a new physical mechanism.  Candidate identity remains observable;
    mean/spread are optional engineering summaries of model-form uncertainty.
    """
    candidate_ids: tuple[str, ...]
    unresolved_slots: tuple[str, ...]
    k3_status: str = 'OFF'
    interpretation: str = 'FINITE_ARCHITECTURE_SET'

    def validate(self) -> 'PortfolioContract':
        if len(self.candidate_ids) < 2:
            raise ValueError('portfolio requires at least two finite candidates')
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError('duplicate portfolio candidate ids')
        if not self.unresolved_slots:
            raise ValueError('portfolio requires at least one unresolved slot')
        if self.k3_status != 'OFF':
            raise ValueError('finite architecture portfolio must not silently enable K3')
        return self

    def to_dict(self):
        self.validate()
        return asdict(self)


class ArchitecturePortfolio(nn.Module):
    """Execute a frozen finite set of admissible architectures in parallel."""
    def __init__(self, models: Mapping[str, nn.Module], unresolved_slots: tuple[str, ...]):
        super().__init__()
        if len(models) < 2:
            raise ValueError('portfolio requires at least two models')
        self.models = nn.ModuleDict({str(k): v for k, v in models.items()})
        self.contract = PortfolioContract(tuple(self.models.keys()), tuple(unresolved_slots)).validate()

    def forward_all(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        return {k: m(*args, **kwargs) for k, m in self.models.items()}

    def forward(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        all_outputs = self.forward_all(*args, **kwargs)
        values = list(all_outputs.values())
        shape = values[0].shape
        if any(v.shape != shape for v in values[1:]):
            raise ValueError('portfolio candidates returned incompatible shapes')
        stack = torch.stack(values, dim=0)
        return {
            'candidate_stack': stack,
            'mean': torch.mean(stack, dim=0),
            'std': torch.std(stack, dim=0, unbiased=False),
            'minimum': torch.amin(stack, dim=0),
            'maximum': torch.amax(stack, dim=0),
        }
