from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Mapping, Sequence
import numpy as np


@dataclass(frozen=True)
class K3QualificationPolicy:
    enabled: bool = False
    finite_rejection_nrmse: float = 0.35
    minimum_relative_improvement: float = 0.10
    max_k3_nrmse: float | None = None
    minimal_augmentation_enabled: bool = False
    minimal_augmentation_epochs: int = 120
    fit_slack: float = 0.05
    constraint_weight: float = 100.0
    credit_assignment_cv_warn: float = 0.25

    @classmethod
    def from_mapping(cls, m: Mapping | None) -> 'K3QualificationPolicy':
        m=dict(m or {}); ma=dict(m.get('minimal_augmentation',{}) or {})
        return cls(
            bool(m.get('enabled',False)),
            float(m.get('finite_rejection_nrmse',0.35)),
            float(m.get('minimum_relative_improvement',0.10)),
            None if m.get('max_k3_nrmse') is None else float(m['max_k3_nrmse']),
            bool(ma.get('enabled',False)),
            int(ma.get('epochs',120)),
            float(ma.get('fit_slack',0.05)),
            float(ma.get('constraint_weight',100.0)),
            float(ma.get('credit_assignment_cv_warn',0.25)),
        )


@dataclass(frozen=True)
class K3GateDecision:
    status: str  # OFF | KEEP_FINITE | FINITE_REJECTED_TRY_K3 | K3_SUPPORTED | K3_NOT_SUPPORTED
    finite_best_nrmse: float | None
    k3_nrmse: float | None
    relative_improvement: float | None
    reason: str
    def to_dict(self): return asdict(self)


def finite_gate(policy: K3QualificationPolicy, candidate_risks: Mapping[str,Sequence[float]], target_scale: float) -> K3GateDecision:
    if not policy.enabled:
        return K3GateDecision('OFF',None,None,None,'K3 policy disabled')
    if target_scale <= 0: raise ValueError('target_scale must be positive')
    med={k:float(np.median(np.asarray(v,float))) for k,v in candidate_risks.items()}
    best=min(med.values())/target_scale
    if best <= policy.finite_rejection_nrmse:
        return K3GateDecision('KEEP_FINITE',best,None,None,'at least one finite mechanism candidate meets the frozen normalized-risk gate; K3 remains off')
    return K3GateDecision('FINITE_REJECTED_TRY_K3',best,None,None,'all frozen finite candidates miss the normalized-risk gate; a previously declared local K3 escape may now be evaluated')


def compare_k3(policy: K3QualificationPolicy, finite_best_nrmse: float, k3_risks: Sequence[float], target_scale: float) -> K3GateDecision:
    if not policy.enabled: return K3GateDecision('OFF',finite_best_nrmse,None,None,'K3 policy disabled')
    k3=float(np.median(np.asarray(k3_risks,float)))/target_scale
    imp=(finite_best_nrmse-k3)/max(finite_best_nrmse,1e-12)
    good=imp>=policy.minimum_relative_improvement and (policy.max_k3_nrmse is None or k3<=policy.max_k3_nrmse)
    if good:
        return K3GateDecision('K3_SUPPORTED',finite_best_nrmse,k3,imp,'qualified local K3 improves on the rejected finite family by the frozen practical margin')
    return K3GateDecision('K3_NOT_SUPPORTED',finite_best_nrmse,k3,imp,'finite family was inadequate, but the local K3 candidate did not satisfy the frozen promotion gate')
