from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Mapping
import torch
from .observation_ir import ObservationContract, ObservedStateModel

@dataclass(frozen=True)
class ObservationDesignPolicy:
    """Explicit engineering thresholds for observation-conditioned architecture design.

    Scores are computed on a dimensionless, scale-normalized local observation
    Jacobian. Defaults are research-prototype values used by E24 and MUST be
    reviewed for a new device family / unit scaling.
    """
    initialization_window: int = 10
    strong_min_sv: float = 5e-2
    strong_relative_sv: float = 1e-1
    weak_min_sv: float = 5e-3
    weak_relative_sv: float = 1e-2

@dataclass(frozen=True)
class ObservationPlan:
    mode: str
    observed_names: tuple[str, ...]
    hidden_names: tuple[str, ...]
    initialization_window: int
    min_sv: float
    max_sv: float
    relative_min_sv: float
    physical_hidden_state_supported: bool
    health_coordinate_permission: bool
    recommendation: str

    def as_dict(self):
        return asdict(self)

def _scale_vector(names, scales: Mapping[str, float] | None, *, dtype, device):
    if scales is None:
        vals=[1.0]*len(names)
    else:
        vals=[float(scales.get(n,1.0)) for n in names]
    if any(v <= 0 for v in vals):
        raise ValueError('state/observation scales must be positive')
    return torch.tensor(vals,dtype=dtype,device=device)

def normalized_observability_singular_values(
    observed_model: ObservedStateModel,
    observations: torch.Tensor,
    inputs_seq: Mapping[str, torch.Tensor],
    *,
    state_scales: Mapping[str, float] | None = None,
):
    """Dimensionless local observability singular values for hidden initial states.

    `observations` shape follows ObservedStateModel: (time, batch=1, n_observed).
    The Jacobian maps dimensionless hidden-state perturbations to dimensionless
    observation-trajectory perturbations using user-declared characteristic scales.
    """
    H=min(observed_model.contract.initialization_window, observations.shape[0])
    hidden_names=observed_model.contract.hidden_names
    if len(hidden_names)==0:
        return torch.tensor([float('inf')],dtype=observations.dtype,device=observations.device)
    if H < 1:
        raise ValueError('empty observation window')
    hscale=_scale_vector(hidden_names,state_scales,dtype=observations.dtype,device=observations.device)
    oscale=_scale_vector(observed_model.contract.observed_names,state_scales,dtype=observations.dtype,device=observations.device)
    nh=len(hidden_names)
    hn=torch.zeros((1,nh),dtype=observations.dtype,device=observations.device,requires_grad=True)
    obs0=observations[0]
    def f(hnorm):
        hphys=hnorm*hscale[None,:]
        z0=observed_model._state0(obs0,hphys)
        z=observed_model.rollout(z0,{k:v[:H-1] for k,v in inputs_seq.items()})
        y=observed_model.observe(z[:,0,:])
        return (y/oscale[None,:]).reshape(-1)
    J=torch.autograd.functional.jacobian(f,hn,vectorize=True).reshape(-1,nh)
    return torch.linalg.svdvals(J)

def plan_observation(
    state_model,
    *,
    state_names: tuple[str, ...],
    observed_names: tuple[str, ...],
    observation_window: torch.Tensor,
    inputs_seq: Mapping[str, torch.Tensor],
    policy: ObservationDesignPolicy = ObservationDesignPolicy(),
    state_scales: Mapping[str, float] | None = None,
) -> ObservationPlan:
    contract=ObservationContract(
        state_names=state_names,
        observed_names=observed_names,
        initialization_window=policy.initialization_window,
        min_observability_sv=policy.weak_min_sv,
    )
    om=ObservedStateModel(state_model,contract)
    hidden=contract.hidden_names
    if len(hidden)==0:
        return ObservationPlan(
            mode='DIRECT', observed_names=observed_names, hidden_names=hidden,
            initialization_window=0, min_sv=float('inf'), max_sv=float('inf'),
            relative_min_sv=1.0, physical_hidden_state_supported=True,
            health_coordinate_permission=True,
            recommendation='Use directly observed physical state; no hidden-state initializer is required.'
        )
    sv=normalized_observability_singular_values(om,observation_window,inputs_seq,state_scales=state_scales)
    mn=float(torch.min(sv));mx=float(torch.max(sv));rel=mn/max(mx,1e-12)
    if mn>=policy.strong_min_sv and rel>=policy.strong_relative_sv:
        mode='LATENT_STRONG';supported=True;health=True
        rec='Keep hidden physical states explicit and initialize them from the observation window.'
    elif mn>=policy.weak_min_sv and rel>=policy.weak_relative_sv:
        mode='LATENT_WEAK';supported=True;health=False
        rec='Keep latent physical states for prediction, but gate parameter/hidden-state PHM semantics; improve sensors, excitation, or initialization horizon.'
    else:
        mode='UNSUPPORTED';supported=False;health=False
        rec='Do not expose hidden physical coordinates as trusted states. Add sensing/history or fall back to a generic latent representation without physical-state claims.'
    return ObservationPlan(
        mode=mode, observed_names=observed_names, hidden_names=hidden,
        initialization_window=policy.initialization_window, min_sv=mn,max_sv=mx,
        relative_min_sv=rel,physical_hidden_state_supported=supported,
        health_coordinate_permission=health,recommendation=rec,
    )
