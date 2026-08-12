from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Mapping
import numpy as np
import torch


@dataclass(frozen=True)
class RobustSupportReference:
    """Normal-data support reference for M2N evidence qualification.

    The reference is intentionally observational: it says how far a state/input
    row lies from the frozen normal commissioning cloud.  It does not assign a
    physical fault label and it is not a learned K3 mechanism.
    """
    median: tuple[float, ...]
    scale: tuple[float, ...]
    distance_threshold: float

    @classmethod
    def fit(cls, x: np.ndarray, quantile: float = 0.99) -> "RobustSupportReference":
        a=np.asarray(x,float)
        if a.ndim!=2 or len(a)<8: raise ValueError('reference x must be [N,D] with N>=8')
        med=np.nanmedian(a,axis=0)
        mad=np.nanmedian(np.abs(a-med[None,:]),axis=0)*1.4826
        std=np.nanstd(a,axis=0)
        floor=np.maximum(np.abs(med)*1e-9,1e-12)
        scale=np.where(mad>floor,mad,np.where(std>floor,std,floor))
        d=np.sqrt(np.nanmean(np.square((a-med[None,:])/scale[None,:]),axis=1))
        threshold=float(np.nanquantile(d,quantile))
        return cls(tuple(map(float,med)),tuple(map(float,scale)),threshold)

    def score(self, x: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        a=np.asarray(x,float)
        med=np.asarray(self.median); scale=np.asarray(self.scale)
        if a.ndim!=2 or a.shape[1]!=len(med): raise ValueError('support x dimension mismatch')
        d=np.sqrt(np.nanmean(np.square((a-med[None,:])/scale[None,:]),axis=1))
        return d,d>self.distance_threshold

    def to_dict(self): return asdict(self)


def portfolio_derivative_evidence(portfolio, state: torch.Tensor, exogenous: torch.Tensor,
                                  observed_rate: torch.Tensor, rate_scale: torch.Tensor):
    """Return sample-level evidence without collapsing an unresolved K2 set.

    `best_finite_residual` asks whether *any* currently admissible finite model
    explains the observation.  `model_disagreement` asks how much the finite
    candidate predictions disagree.  Neither quantity selects or reweights the
    frozen portfolio.
    """
    if observed_rate.ndim!=2: raise ValueError('observed_rate must be [N,D]')
    if rate_scale.ndim!=1 or rate_scale.numel()!=observed_rate.shape[1]:
        raise ValueError('rate_scale dimension mismatch')
    pred_map=portfolio.forward_all(state,exogenous)
    ids=tuple(pred_map.keys())
    stack=torch.stack([pred_map[k] for k in ids],dim=0).to(dtype=torch.float64)
    obs=observed_rate.to(dtype=torch.float64)
    scale=torch.clamp(rate_scale.to(dtype=torch.float64),min=1e-12)
    residual=(stack-obs[None,:,:])/scale[None,None,:]
    candidate_sample_rms=torch.sqrt(torch.mean(residual**2,dim=2))
    best_values,best_indices=torch.min(candidate_sample_rms,dim=0)
    channel_best_abs=torch.min(torch.abs(residual),dim=0).values
    channel_median_abs=torch.median(torch.abs(residual),dim=0).values
    median_values=torch.median(candidate_sample_rms,dim=0).values
    pred_std=torch.std(stack,dim=0,unbiased=False)
    disagreement=torch.sqrt(torch.mean((pred_std/scale[None,:])**2,dim=1))
    consensus=torch.mean(stack,dim=0)
    consensus_residual=torch.sqrt(torch.mean(((consensus-obs)/scale[None,:])**2,dim=1))
    return {
        'candidate_ids':ids,
        'candidate_predictions':stack,
        'candidate_sample_rms':candidate_sample_rms,
        'best_finite_residual':best_values,
        'best_candidate_index':best_indices,
        'channel_best_abs':channel_best_abs,
        'channel_median_abs':channel_median_abs,
        'median_finite_residual':median_values,
        'model_disagreement':disagreement,
        'consensus_residual':consensus_residual,
    }


def summarize_portfolio_segment(evidence: Mapping[str, object], mask: np.ndarray) -> dict:
    m=torch.as_tensor(np.asarray(mask,bool))
    if m.ndim!=1 or not bool(torch.any(m)): raise ValueError('segment mask is empty')
    ids=tuple(evidence['candidate_ids'])
    best=evidence['best_finite_residual'][m]
    median=evidence['median_finite_residual'][m]
    disagree=evidence['model_disagreement'][m]
    consensus=evidence['consensus_residual'][m]
    candidate=evidence['candidate_sample_rms'][:,m]
    per_candidate=torch.sqrt(torch.mean(candidate**2,dim=1))
    cb=evidence['channel_best_abs'][m]
    cm=evidence['channel_median_abs'][m]
    order=torch.argsort(per_candidate)
    winners=evidence['best_candidate_index'][m]
    counts=torch.bincount(winners,minlength=len(ids)).to(dtype=torch.float64)
    fractions=counts/torch.sum(counts)
    return {
        'sample_count':int(torch.sum(m).item()),
        'best_finite_residual_rms':float(torch.sqrt(torch.mean(best**2)).item()),
        'median_finite_residual_rms':float(torch.sqrt(torch.mean(median**2)).item()),
        'consensus_residual_rms':float(torch.sqrt(torch.mean(consensus**2)).item()),
        'model_disagreement_rms':float(torch.sqrt(torch.mean(disagree**2)).item()),
        'best_candidate':ids[int(order[0])],
        'runner_up_candidate':ids[int(order[1])] if len(ids)>1 else None,
        'channel_best_rms':[float(x) for x in torch.sqrt(torch.mean(cb**2,dim=0)).tolist()],
        'channel_median_rms':[float(x) for x in torch.sqrt(torch.mean(cm**2,dim=0)).tolist()],
        'candidate_segment_rms':{ids[i]:float(per_candidate[i].item()) for i in range(len(ids))},
        'best_sample_fraction':{ids[i]:float(fractions[i].item()) for i in range(len(ids))},
    }
