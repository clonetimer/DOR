#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C1-B2a: random-init topology + constitutive operator recovery."""
from __future__ import annotations
from typing import Dict, List
import copy
import numpy as np
import torch
import torch.nn.functional as F

from .grammar import HierarchicalISGNGraph
from .contracts import PublicDataset
from .training import semantic_f1, select_graph_by_effect, rollout_nrmse


def _tensor(a, device):
    return torch.as_tensor(a, dtype=torch.float32, device=device)


def _scaled_increment_loss(model, X, U):
    x=X[:,:-1]; y=X[:,1:]
    pred=model.one_step(x,U)
    ti=(y-x)/model.dt; pi=(pred-x)/model.dt
    scale=ti.detach().std((0,1),keepdim=True).clamp_min(1e-3)
    return (((pi-ti)/scale)**2).mean()


def train_b2a(
    public_train: PublicDataset,
    public_val: PublicDataset,
    seed: int,
    device: str = "cpu",
    epochs: int = 900,
    lr: float = 0.02,
    l0: float = 1e-3,
    entropy_weight: float = 2e-3,
    constitutive_complexity_weight: float = 3e-3,
    refit_epochs: int = 900,
):
    torch.manual_seed(seed); np.random.seed(seed)
    model=HierarchicalISGNGraph.from_schema(
        public_train.schema, public_train.dt, seed=seed,
        allow_constant=False, constitutive_search=True,
    ).to(device)
    X=_tensor(public_train.X,device); U=_tensor(public_train.U,device)
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    best=None; bestv=float('inf')
    for ep in range(epochs):
        # Keep mixtures soft first, then let operator choice sharpen.
        frac=ep/max(epochs-1,1)
        temp=max(0.18,2.0*(1-frac)+0.12)
        model.set_operator_temperature(temp)
        opt.zero_grad()
        pred_loss=_scaled_increment_loss(model,X,U)
        ramp=max(0.0,(frac-0.18)/0.62)
        loss=(pred_loss + l0*ramp*model.l0_penalty()
              + entropy_weight*ramp*model.operator_entropy()
              + constitutive_complexity_weight*ramp*model.constitutive_complexity())
        # Periodic short rollout prevents a locally good constitutive mixture from drifting dynamically.
        if ep % 40 == 0 and ep > epochs//4:
            r=model.rollout(X[:,0],U[:,:min(60,U.shape[1])])
            gt=X[:,:r.shape[1]]
            sc=gt.detach().std((0,1),keepdim=True).clamp_min(1e-3)
            loss=loss+0.12*(((r-gt)/sc)**2).mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),10.0); opt.step()
        if ep%50==0 or ep==epochs-1:
            v=rollout_nrmse(model,public_val,device=device,max_steps=min(120,public_val.U.shape[1]))
            if np.isfinite(v) and v<bestv:
                bestv=v; best=copy.deepcopy(model.state_dict())
    if best is not None: model.load_state_dict(best)
    model.set_operator_temperature(0.08)
    selected_graph=select_graph_by_effect(model,relative=0.10,absolute=0.035)
    selected_ops=model.selected_transforms(selected_graph)
    model.freeze_topology(selected_graph)
    model.freeze_constitutive(selected_ops)

    # Physical refit with exact topology and exact discrete constitutive choice fixed.
    params=model.trainable_physical_parameters()
    # An empty selected graph is a valid abstention outcome on real/unsupported data.
    # Do not turn it into an optimizer exception; simply skip physical refit.
    if params:
        opt2=torch.optim.Adam(params,lr=0.012)
        for ep in range(refit_epochs):
            opt2.zero_grad(); lp=_scaled_increment_loss(model,X,U)
            loss=lp
            if ep%50==0:
                r=model.rollout(X[:,0],U[:,:min(100,U.shape[1])])
                gt=X[:,:r.shape[1]]; sc=gt.detach().std((0,1),keepdim=True).clamp_min(1e-3)
                loss=loss+0.25*(((r-gt)/sc)**2).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(params,10.0); opt2.step()
    return model, selected_graph, selected_ops, bestv


def expected_transform(truth: PrivateTruth, target: str, source: str) -> str:
    return str(truth.transforms.get(target,{}).get(source,{}).get("op","Identity"))


def transform_accuracy(selected_ops: Dict[str,Dict[str,str]], truth: PrivateTruth) -> float:
    correct=total=0
    # Score only ground-truth active edges; topology F1 handles false/missing edges separately.
    for target,sources in truth.graph.items():
        for source in sources:
            total+=1
            if selected_ops.get(target,{}).get(source) == expected_transform(truth,target,source):
                correct+=1
    return correct/max(total,1)


def constitutive_recovery(model, selected_graph, truth: PrivateTruth) -> Dict[str,float]:
    errs={}
    pars=model.constitutive_parameters(selected_graph)
    for target,row in truth.transforms.items():
        for source,info in row.items():
            true=float(info["param"])
            est=pars.get(target,{}).get(source,None)
            errs[f"{target}:{source}:{info['op']}"] = float('inf') if est is None else abs(float(est)-true)/abs(true)
    return errs
