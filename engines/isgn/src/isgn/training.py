#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C1-A structure recovery trainer for the canonical HierarchicalISGNGraph."""
from __future__ import annotations
from typing import Dict, List, Tuple, Iterable, Optional
import copy
import numpy as np
import torch
import torch.nn.functional as F

from .grammar import HierarchicalISGNGraph
from .contracts import PublicDataset


def _tensor(a, device):
    return torch.as_tensor(a, dtype=torch.float32, device=device)


def _scaled_increment_loss(model, X, U):
    x = X[:, :-1]
    y = X[:, 1:]
    u = U
    # Vectorized one-step RK4 over BxT points.
    pred = model.one_step(x, u)
    target_inc = (y - x) / model.dt
    pred_inc = (pred - x) / model.dt
    scale = target_inc.detach().std(dim=(0, 1), keepdim=True).clamp_min(1e-3)
    return (((pred_inc - target_inc) / scale) ** 2).mean()


def rollout_nrmse(model, public: PublicDataset, device="cpu", max_steps=None) -> float:
    X = _tensor(public.X, device); U = _tensor(public.U, device)
    if max_steps is not None:
        U = U[:, :max_steps]; X = X[:, :max_steps+1]
    with torch.no_grad():
        pred = model.rollout(X[:, 0], U)
    rmse = torch.sqrt(torch.mean((pred - X) ** 2, dim=(0, 1)))
    sd = torch.std(X, dim=(0, 1)).clamp_min(1e-6)
    return float(torch.mean(rmse / sd))


def semantic_f1(pred: Dict[str, List[str]], truth: Dict[str, List[str]]) -> float:
    tp=fp=fn=0
    for target, gt_list in truth.items():
        gt=set(gt_list); pr=set(pred.get(target, []))
        tp += len(gt & pr); fp += len(pr - gt); fn += len(gt - pr)
    p = tp / max(tp+fp, 1); r = tp / max(tp+fn, 1)
    return 2*p*r/max(p+r, 1e-12)


def select_graph_by_effect(model, relative=0.10, absolute=0.04) -> Dict[str, List[str]]:
    result={}
    for eq in model.equations:
        vals=[]
        for term in eq.terms:
            c=abs(float(term.effective_coefficient().detach()))
            vals.append((term.spec.source_name,c,float(term.gate.value().detach())))
        mx=max([v[1] for v in vals] + [0.0])
        th=max(absolute, relative*mx)
        result[eq.target.name]=[name for name,c,g in vals if c>=th and g>=0.15]
    return result


def train_c1a(
    public_train: PublicDataset,
    public_val: PublicDataset,
    seed: int,
    device="cpu",
    epochs=1000,
    lr=0.025,
    l0=2e-3,
    refit_epochs=300,
    refit_rollout_every=20,
    allow_constant=False,
    known_present: Optional[Dict[str, Iterable[str]]] = None,
    known_absent: Optional[Dict[str, Iterable[str]]] = None,
):
    torch.manual_seed(seed); np.random.seed(seed)
    model=HierarchicalISGNGraph.from_schema(public_train.schema, dt=public_train.dt, seed=seed, allow_constant=allow_constant).to(device)
    model.apply_topology_prior(known_present=known_present, known_absent=known_absent)
    X=_tensor(public_train.X,device); U=_tensor(public_train.U,device)
    structure_params=[p for p in model.parameters() if p.requires_grad]
    best=None; best_val=float("inf")
    if structure_params:
        opt=torch.optim.Adam(structure_params, lr=lr)
        for ep in range(epochs):
            opt.zero_grad()
            pred_loss=_scaled_increment_loss(model,X,U)
            ramp=max(0.0,(ep/epochs-0.20)/0.55)
            loss=pred_loss + l0*ramp*model.l0_penalty()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(structure_params,10.0)
            opt.step()
            if ep % 50 == 0 or ep == epochs-1:
                v=rollout_nrmse(model,public_val,device=device,max_steps=min(120,public_val.U.shape[1]))
                if v < best_val:
                    best_val=v; best=copy.deepcopy(model.state_dict())
    if best is not None:
        model.load_state_dict(best)
    selected=select_graph_by_effect(model)
    # A declared known-present mechanism is preserved in the returned graph; if data later
    # contradict it, M2 PriorConsistency must report that conflict rather than silently deleting it.
    for target, names in (known_present or {}).items():
        selected.setdefault(target, [])
        selected[target] = sorted(set(selected[target]) | set(names))
    model.freeze_topology(selected)

    # Post-selection physical refit: topology fixed exactly, only physical coefficients train.
    params=model.trainable_physical_parameters()
    # A valid sparse-selection outcome can be an empty graph (for example when
    # strong known-absent priors and data eliminate all candidate paths).  Do
    # not construct an optimizer with an empty parameter list; return the
    # selected empty topology so certification can reject/interpret it.
    if params:
        opt2=torch.optim.Adam(params, lr=0.015)
        for ep in range(refit_epochs):
            opt2.zero_grad()
            lp=_scaled_increment_loss(model,X,U)
            # Periodic multi-step rollout constraint from full trajectories.
            if ep % refit_rollout_every == 0:
                r=model.rollout(X[:,0],U[:,:min(80,U.shape[1])])
                gt=X[:,:r.shape[1]]
                scale=gt.detach().std(dim=(0,1),keepdim=True).clamp_min(1e-3)
                lr_loss=(((r-gt)/scale)**2).mean()
                loss=lp+0.25*lr_loss
            else:
                loss=lp
            loss.backward(); torch.nn.utils.clip_grad_norm_(params,10.0); opt2.step()
    return model, selected, best_val
