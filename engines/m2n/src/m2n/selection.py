from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Mapping, Sequence
import numpy as np

@dataclass(frozen=True)
class ArchitectureSelectionPolicy:
    """Research policy for normal-data architecture evidence.

    `practical_ratio=0.8` means the winning architecture must show at least a
    20% validation-risk advantage. `restart_stability=0.75` requires that this
    practical advantage holds for at least 75% of pairwise optimizer restarts.
    These values are E27C research settings, not universal engineering constants.
    """
    practical_ratio: float = 0.80
    restart_stability: float = 0.75

@dataclass(frozen=True)
class ArchitectureSelectionPlan:
    status: str  # SELECTED | AMBIGUOUS
    selected: str | None
    runner_up: str | None
    median_risk_ratio: float
    pairwise_practical_win_fraction: float
    candidate_median_risks: tuple[tuple[str,float], ...]
    recommendation: str
    def as_dict(self): return asdict(self)


def select_architecture(
    validation_risks: Mapping[str, Sequence[float]],
    *,
    policy: ArchitectureSelectionPolicy = ArchitectureSelectionPolicy(),
) -> ArchitectureSelectionPlan:
    """Select among mechanism-admissible candidate architectures.

    Each candidate supplies one validation risk per independent optimizer
    restart. The function deliberately does not train models and does not use
    fault/OOD labels. It only qualifies *already measured normal validation
    evidence*.

    A candidate is selected only if its median risk is practically lower than
    every alternative and the pairwise restart evidence is stable. Otherwise
    the result is AMBIGUOUS.
    """
    if len(validation_risks) < 2:
        raise ValueError('need at least two candidate architectures')
    clean={}
    for k,v in validation_risks.items():
        a=np.asarray(tuple(v),dtype=float)
        if a.size < 2 or not np.all(np.isfinite(a)) or np.any(a < 0):
            raise ValueError(f'candidate {k!r} needs >=2 finite nonnegative restart risks')
        clean[str(k)]=a
    med={k:float(np.median(v)) for k,v in clean.items()}
    order=sorted(med,key=med.get)
    best=order[0]; runner=order[1]
    worst_ratio=0.0; min_stability=1.0
    for other in order[1:]:
        ratios=np.asarray([a/(b+1e-12) for a in clean[best] for b in clean[other]],dtype=float)
        rmed=float(np.median(ratios)); stability=float(np.mean(ratios < policy.practical_ratio))
        worst_ratio=max(worst_ratio,rmed); min_stability=min(min_stability,stability)
    selected=(worst_ratio < policy.practical_ratio and min_stability >= policy.restart_stability)
    if selected:
        return ArchitectureSelectionPlan(
            status='SELECTED', selected=best, runner_up=runner,
            median_risk_ratio=worst_ratio,
            pairwise_practical_win_fraction=min_stability,
            candidate_median_risks=tuple((k,med[k]) for k in order),
            recommendation='Freeze the selected candidate for downstream evaluation; keep the evidence/provenance record.'
        )
    return ArchitectureSelectionPlan(
        status='AMBIGUOUS', selected=None, runner_up=runner,
        median_risk_ratio=worst_ratio,
        pairwise_practical_win_fraction=min_stability,
        candidate_median_risks=tuple((k,med[k]) for k in order),
        recommendation='Do not auto-select a K3 placement. Collect broader normal contexts, improve optimization stability, or retain multiple candidates.'
    )
