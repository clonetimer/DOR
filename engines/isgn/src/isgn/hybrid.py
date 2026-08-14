"""Observed-regime support for hybrid and switched industrial mechanisms.

Toolkit 2.2 deliberately does *not* learn latent modes or replace the existing
mechanism solvers.  Instead it adds a small, auditable layer that lets callers
supply sample/interval-level operating regimes derived from observed controls,
observed process measurements, declared schedules, or external operating
context.  The layer excludes switching neighborhoods and projects each dwell
mode back into the ordinary ISGN 2.x data contracts so existing recovery and
certification paths can be reused.

Fault/health target labels are rejected as mechanism partitions by default.
This prevents a diagnostic label from silently becoming a causal mechanism
input.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from copy import deepcopy
from typing import Any, Dict, Mapping, Sequence
import numpy as np

from .contracts import AlgebraicDataset, PublicDataset
from .algebraic import (
    AlgebraicRelationResult,
    discover_algebraic_relation,
    discover_algebraic_relation_subset,
)

ELIGIBLE_REGIME_SOURCE_ROLES = frozenset({
    "observed_control", "observed_process", "declared_schedule", "external_context"
})
KNOWN_REGIME_SOURCE_ROLES = ELIGIBLE_REGIME_SOURCE_ROLES | frozenset({"fault_label", "unknown"})


def _as_label_array(a: Any) -> np.ndarray:
    x=np.asarray(a)
    if x.ndim not in (1,2):
        raise ValueError("regime labels must be a rank-1 sample array or rank-2 interval array")
    if x.size == 0:
        raise ValueError("regime labels must not be empty")
    return x


def _as_transition_mask(mask: Any | None, shape: tuple[int,...]) -> np.ndarray:
    if mask is None:
        return np.zeros(shape,dtype=bool)
    x=np.asarray(mask,dtype=bool)
    if x.shape != shape:
        raise ValueError(f"transition_mask shape {x.shape} does not match labels {shape}")
    return x


@dataclass
class RegimePartition:
    """Sample/interval-level observed operating-regime declaration.

    ``labels`` is rank-1 for :class:`AlgebraicDataset` samples and rank-2 with
    shape ``[trajectory, interval]`` for :class:`PublicDataset`.  Intervals
    marked by ``transition_mask`` are excluded from stationary-mode mechanism
    recovery by the projection helpers.
    """
    labels: np.ndarray
    source: str
    source_role: str = "unknown"
    transition_mask: np.ndarray | None = None
    metadata: Dict[str,Any] = field(default_factory=dict)

    def __post_init__(self):
        self.labels=_as_label_array(self.labels)
        self.transition_mask=_as_transition_mask(self.transition_mask,self.labels.shape)
        self.source=str(self.source)
        self.source_role=str(self.source_role)
        if self.source_role not in KNOWN_REGIME_SOURCE_ROLES:
            raise ValueError(f"unknown regime source_role {self.source_role!r}; expected one of {sorted(KNOWN_REGIME_SOURCE_ROLES)}")
        self.metadata=dict(self.metadata or {})

    @property
    def mechanism_eligible(self) -> bool:
        return self.source_role in ELIGIBLE_REGIME_SOURCE_ROLES

    @property
    def regimes(self) -> list[str]:
        vals=np.asarray(self.labels,dtype=object)[~np.asarray(self.transition_mask,dtype=bool)]
        return sorted({str(x) for x in vals.tolist()})

    @property
    def transition_fraction(self) -> float:
        return float(np.mean(np.asarray(self.transition_mask,dtype=bool)))

    def summary(self) -> Dict[str,Any]:
        return {
            "shape":list(self.labels.shape), "source":self.source,
            "source_role":self.source_role, "mechanism_eligible":self.mechanism_eligible,
            "regimes":self.regimes, "transition_fraction":self.transition_fraction,
            "metadata":dict(self.metadata),
        }

    def validate_algebraic(self, ds: AlgebraicDataset) -> None:
        ds.validate()
        if self.labels.ndim != 1 or self.labels.shape != (len(ds.values),):
            raise ValueError(f"algebraic partition labels must have shape {(len(ds.values),)}, got {self.labels.shape}")

    def validate_public(self, ds: PublicDataset) -> None:
        ds.validate()
        expected=(ds.X.shape[0],ds.X.shape[1]-1)
        if self.labels.ndim != 2 or self.labels.shape != expected:
            raise ValueError(f"public partition labels must have shape {expected}, got {self.labels.shape}")


def _threshold_labels_1d(values: np.ndarray, threshold: float, hysteresis: float,
                         low_label: str, high_label: str) -> np.ndarray:
    v=np.asarray(values,float)
    if not np.isfinite(v).all():
        raise ValueError("threshold regime source contains non-finite values")
    h=float(hysteresis)
    if h < 0: raise ValueError("hysteresis must be non-negative")
    if h == 0:
        return np.where(v>float(threshold),str(high_label),str(low_label)).astype(object)
    lo=float(threshold)-0.5*h; hi=float(threshold)+0.5*h
    state=bool(v[0]>float(threshold)); out=[]
    for z in v:
        if state and z < lo: state=False
        elif (not state) and z > hi: state=True
        out.append(str(high_label) if state else str(low_label))
    return np.asarray(out,dtype=object)


def _transition_neighborhood(labels: np.ndarray, margin: int) -> np.ndarray:
    m=int(margin)
    if m < 0: raise ValueError("transition_margin must be non-negative")
    x=np.asarray(labels,dtype=object)
    rows=x[None,:] if x.ndim==1 else x
    out=np.zeros(rows.shape,dtype=bool)
    for k,row in enumerate(rows):
        changes=np.flatnonzero(row[1:] != row[:-1])+1
        for j in changes:
            out[k,max(0,j-m):min(len(row),j+m+1)]=True
    return out[0] if x.ndim==1 else out


def threshold_regime_partition(values: Any, threshold: float, *,
                               source: str, source_role: str="observed_process",
                               low_label: str="low", high_label: str="high",
                               hysteresis: float=0.0, transition_margin: int=0,
                               metadata: Mapping[str,Any] | None=None) -> RegimePartition:
    """Build an explicit two-mode partition from one observed channel.

    For rank-2 arrays hysteresis is applied independently to each trajectory.
    ``transition_margin`` is measured in samples/intervals and masks both sides
    of every mode switch.
    """
    a=np.asarray(values,float)
    if a.ndim not in (1,2): raise ValueError("threshold values must be rank-1 or rank-2")
    if a.ndim==1:
        labels=_threshold_labels_1d(a,threshold,hysteresis,low_label,high_label)
    else:
        labels=np.stack([_threshold_labels_1d(row,threshold,hysteresis,low_label,high_label) for row in a],axis=0)
    trans=_transition_neighborhood(labels,int(transition_margin))
    meta=dict(metadata or {})
    meta.update({"partition_rule":"observed-threshold","threshold":float(threshold),"hysteresis":float(hysteresis),
                 "low_label":str(low_label),"high_label":str(high_label),"transition_margin":int(transition_margin)})
    return RegimePartition(labels=labels,transition_mask=trans,source=source,source_role=source_role,metadata=meta)


def _regime_mask(partition: RegimePartition, regime: str) -> np.ndarray:
    return (np.asarray(partition.labels,dtype=str)==str(regime)) & (~np.asarray(partition.transition_mask,dtype=bool))


def project_algebraic_regime(ds: AlgebraicDataset, partition: RegimePartition, *, regime: str,
                             allow_fault_label_partition: bool=False) -> AlgebraicDataset:
    """Project one stationary operating regime into the ordinary algebraic contract."""
    partition.validate_algebraic(ds)
    if not partition.mechanism_eligible and not (allow_fault_label_partition and partition.source_role=="fault_label"):
        raise ValueError(f"regime source_role {partition.source_role!r} is not eligible for mechanism discovery")
    mask=_regime_mask(partition,regime)
    if not np.any(mask): raise ValueError(f"regime {regime!r} has no non-transition samples")
    schema=deepcopy(ds.schema)
    schema["regime_projection"]={
        "parent_regime":str(ds.regime),"regime":str(regime),"partition":partition.summary(),
        "transition_samples_excluded":True,"fault_label_partition":partition.source_role=="fault_label",
    }
    meta=deepcopy(ds.metadata)
    meta["regime_projection"]={"regime":str(regime),"kept_rows":int(mask.sum()),"parent_rows":int(len(mask))}
    out=AlgebraicDataset(np.asarray(ds.values)[mask].copy(),np.asarray(ds.sample_id)[mask].copy(),schema,
                         regime=str(regime),variant=f"{ds.variant}-regime-{regime}",metadata=meta)
    out.validate(); return out


def project_public_regime_windows(ds: PublicDataset, partition: RegimePartition, *, regime: str,
                                  window_steps: int, stride_steps: int | None=None,
                                  min_windows: int=1, allow_fault_label_partition: bool=False) -> PublicDataset:
    """Extract fixed-length, transition-free dwell windows for an observed mode.

    This is the dynamic bridge: it does not introduce a new ODE solver.  It
    converts dwell intervals back to the normal :class:`PublicDataset` shape so
    ``recover_graph`` / missing-dynamics recovery can operate unchanged.
    """
    partition.validate_public(ds)
    if not partition.mechanism_eligible and not (allow_fault_label_partition and partition.source_role=="fault_label"):
        raise ValueError(f"regime source_role {partition.source_role!r} is not eligible for mechanism discovery")
    w=int(window_steps); stride=int(stride_steps if stride_steps is not None else window_steps)
    if w < 1 or stride < 1: raise ValueError("window_steps and stride_steps must be positive")
    valid=_regime_mask(partition,regime)
    Xs=[]; Us=[]; Ts=[]; parents=[]
    for k,row in enumerate(valid):
        start=0; T=len(row)
        while start+w <= T:
            if bool(np.all(row[start:start+w])):
                Xs.append(np.asarray(ds.X[k,start:start+w+1]).copy())
                Us.append(np.asarray(ds.U[k,start:start+w]).copy())
                if ds.sample_time_s is not None:
                    Ts.append(np.asarray(ds.sample_time_s[k,start:start+w+1],dtype=np.float64).copy())
                parents.append({"trajectory_id":str(np.asarray(ds.trajectory_id)[k]),"start_interval":int(start),"end_interval":int(start+w)})
                start += stride
            else:
                bad=np.flatnonzero(~row[start:start+w])
                start += int(bad[0])+1 if len(bad) else 1
    if len(Xs) < int(min_windows):
        raise ValueError(f"regime {regime!r} provides {len(Xs)} valid windows; need at least {int(min_windows)}")
    schema=deepcopy(ds.schema)
    schema["regime_projection"]={
        "parent_regime":str(ds.regime),"regime":str(regime),"partition":partition.summary(),
        "window_steps":w,"stride_steps":stride,"transition_intervals_excluded":True,"parent_windows":parents,
    }
    out=PublicDataset(np.stack(Xs),np.stack(Us),np.arange(len(Xs),dtype=np.int64),schema,float(ds.dt),str(regime),
                      variant=f"{ds.variant}-regime-{regime}",
                      sample_time_s=(np.stack(Ts) if ds.sample_time_s is not None else None))
    out.validate(); return out


@dataclass
class RegimeRelationEvidence:
    regime: str
    status: str
    train_rows: int
    validation_rows: int
    target_std: float
    relation: AlgebraicRelationResult | None = None
    reason: str | None = None
    metadata: Dict[str,Any] = field(default_factory=dict)
    def to_dict(self):
        d=asdict(self)
        if self.relation is not None: d["relation"]=self.relation.to_dict()
        return d


@dataclass
class HybridAlgebraicRecoveryResult:
    target: str
    inputs: list[str]
    status: str
    regimes: Dict[str,RegimeRelationEvidence]
    train_partition: Dict[str,Any]
    validation_partition: Dict[str,Any]
    global_baseline: AlgebraicRelationResult | None
    certification_allowed: bool
    metadata: Dict[str,Any] = field(default_factory=dict)
    backend: str = "observed-regime-router"
    def to_dict(self):
        return {
            "target":self.target,"inputs":list(self.inputs),"status":self.status,
            "regimes":{k:v.to_dict() for k,v in self.regimes.items()},
            "train_partition":dict(self.train_partition),"validation_partition":dict(self.validation_partition),
            "global_baseline":None if self.global_baseline is None else self.global_baseline.to_dict(),
            "certification_allowed":bool(self.certification_allowed),"metadata":dict(self.metadata),"backend":self.backend,
        }


def _project_union_nontransition(ds: AlgebraicDataset, partition: RegimePartition) -> AlgebraicDataset:
    partition.validate_algebraic(ds); mask=~np.asarray(partition.transition_mask,dtype=bool)
    schema=deepcopy(ds.schema); meta=deepcopy(ds.metadata)
    schema["transition_projection"]={"partition":partition.summary(),"transition_samples_excluded":True}
    out=AlgebraicDataset(np.asarray(ds.values)[mask].copy(),np.asarray(ds.sample_id)[mask].copy(),schema,
                         regime=f"{ds.regime}-nontransition",variant=f"{ds.variant}-nontransition",metadata=meta)
    out.validate(); return out


def discover_regime_algebraic_relations(train: AlgebraicDataset, validation: AlgebraicDataset, *,
                                         train_partition: RegimePartition, validation_partition: RegimePartition,
                                         inputs: Sequence[str], target: str, method: str="subset",
                                         degree: int=2, max_terms: int=3, complexity_weight: float=1e-3,
                                         alpha_grid: Sequence[float]=(1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2),
                                         support_threshold: float=2e-3,
                                         min_train_rows: int=20, min_validation_rows: int=10,
                                         min_target_std: float=0.0, include_global_baseline: bool=True,
                                         allow_fault_label_partition: bool=False) -> HybridAlgebraicRecoveryResult:
    """Run existing algebraic recovery independently inside observed dwell modes.

    This function is intentionally candidate-generation only.  It returns
    ``certification_allowed=False``: each recovered mode still needs the normal
    ISGN evidence/certification process and appropriate physical interpretation.
    """
    train.validate(); validation.validate(); train_partition.validate_algebraic(train); validation_partition.validate_algebraic(validation)
    if train.variable_names != validation.variable_names:
        raise ValueError("train/validation algebraic variable schemas must match")
    for p in (train_partition,validation_partition):
        if not p.mechanism_eligible and not (allow_fault_label_partition and p.source_role=="fault_label"):
            raise ValueError(f"regime source_role {p.source_role!r} is not eligible for mechanism discovery")
    if train_partition.source_role != validation_partition.source_role:
        raise ValueError("train and validation regime source_role must match")
    if target not in train.variable_names: raise KeyError(f"unknown target {target!r}")
    common=sorted(set(train_partition.regimes)&set(validation_partition.regimes))
    if not common: raise ValueError("no common non-transition regimes between train and validation")
    j=train.variable_names.index(target); evidence={}
    for regime in common:
        tr=project_algebraic_regime(train,train_partition,regime=regime,allow_fault_label_partition=allow_fault_label_partition)
        va=project_algebraic_regime(validation,validation_partition,regime=regime,allow_fault_label_partition=allow_fault_label_partition)
        std=float(np.std(np.asarray(tr.values)[:,j]))
        status="RECOVERED"; reason=None; relation=None
        if len(tr.values) < int(min_train_rows) or len(va.values) < int(min_validation_rows):
            status="INSUFFICIENT_MODE_SUPPORT"; reason=f"need train>={int(min_train_rows)} and validation>={int(min_validation_rows)} rows"
        elif std < float(min_target_std):
            status="DEGENERATE_TARGET_VARIATION"; reason=f"target std {std:.6g} < min_target_std {float(min_target_std):.6g}"
        else:
            if method=="subset":
                relation=discover_algebraic_relation_subset(tr,va,inputs=inputs,target=target,degree=int(degree),
                    max_terms=int(max_terms),complexity_weight=float(complexity_weight))
            elif method=="lasso":
                relation=discover_algebraic_relation(tr,va,inputs=inputs,target=target,degree=int(degree),
                    alpha_grid=alpha_grid,support_threshold=float(support_threshold),complexity_weight=float(complexity_weight))
            else: raise ValueError("method must be 'subset' or 'lasso'")
        evidence[regime]=RegimeRelationEvidence(regime,status,int(len(tr.values)),int(len(va.values)),std,relation,reason,
                                                {"transition_rows_excluded":True})
    global_result=None
    if include_global_baseline:
        tr=_project_union_nontransition(train,train_partition); va=_project_union_nontransition(validation,validation_partition)
        if method=="subset":
            global_result=discover_algebraic_relation_subset(tr,va,inputs=inputs,target=target,degree=int(degree),max_terms=int(max_terms),complexity_weight=float(complexity_weight))
        else:
            global_result=discover_algebraic_relation(tr,va,inputs=inputs,target=target,degree=int(degree),alpha_grid=alpha_grid,
                                                      support_threshold=float(support_threshold),complexity_weight=float(complexity_weight))
    recovered=sum(v.status=="RECOVERED" for v in evidence.values())
    blocked=len(evidence)-recovered
    status="HYBRID_RELATION_EVIDENCE" if recovered and not blocked else "PARTIAL_HYBRID_RELATION_EVIDENCE" if recovered else "NO_MODE_RECOVERABLE"
    return HybridAlgebraicRecoveryResult(str(target),list(inputs),status,evidence,train_partition.summary(),validation_partition.summary(),global_result,False,
        {"observed_partition_only":True,"latent_mode_learning":False,"fault_label_partition":train_partition.source_role=="fault_label",
         "transition_samples_excluded":True,"recovered_modes":int(recovered),"blocked_modes":int(blocked),
         "interpretation":"mode-conditioned candidate generation; downstream ISGN certification still required"})
