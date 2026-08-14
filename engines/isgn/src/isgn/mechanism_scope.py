from __future__ import annotations
from copy import deepcopy
from typing import Iterable, Sequence
import numpy as np

from .contracts import PublicDataset


def project_mechanism_scope(public: PublicDataset, *, target_states: Sequence[str],
                            driver_states: Sequence[str] = (), driver_role: str = "context",
                            driver_sampling: str = "left", variant_suffix: str = "mechanism-scope") -> PublicDataset:
    """Project a public observation dataset into a local mechanism block.

    ``target_states`` remain integrated states. ``driver_states`` are still public
    measurements, but are re-role-labelled as observed exogenous drivers for this
    local mechanism block.  No truth channel, interpolation, imputation or state
    estimation is introduced by this function.
    """
    if driver_role not in {"input", "context"}:
        raise ValueError("driver_role must be 'input' or 'context'")
    if driver_sampling not in {"left", "midpoint"}:
        raise ValueError("driver_sampling must be 'left' or 'midpoint'")
    targets=list(target_states); drivers=list(driver_states)
    if not targets:
        raise ValueError("target_states must be non-empty")
    if len(set(targets)) != len(targets) or len(set(drivers)) != len(drivers):
        raise ValueError("target_states and driver_states must be unique")
    overlap=set(targets) & set(drivers)
    if overlap:
        raise ValueError(f"states cannot be both target and driver: {sorted(overlap)}")

    state_meta={v["name"]:v for v in public.schema.get("states",[])}
    missing=(set(targets)|set(drivers))-set(state_meta)
    if missing:
        raise KeyError(f"unknown observed state(s): {sorted(missing)}")
    old_input_names={v["name"] for v in public.schema.get("inputs",[])}
    conflict=set(drivers)&old_input_names
    if conflict:
        raise ValueError(f"driver name collides with existing input: {sorted(conflict)}")

    idx={v["name"]:i for i,v in enumerate(public.schema["states"])}
    X=public.X[:,:, [idx[n] for n in targets]].copy()
    promoted=[]
    for n in drivers:
        a=np.asarray(public.X[:,:,idx[n]],dtype=np.float32)
        if driver_sampling == "left":
            promoted.append(a[:,:-1])
        else:
            promoted.append(.5*(a[:,:-1]+a[:,1:]))
    U=np.asarray(public.U,dtype=np.float32).copy()
    if promoted:
        U=np.concatenate([U,np.stack(promoted,axis=-1)],axis=-1)

    schema=deepcopy(public.schema)
    schema["states"]=[deepcopy(state_meta[n]) for n in targets]
    new_inputs=deepcopy(public.schema.get("inputs",[]))
    for n in drivers:
        m=deepcopy(state_meta[n])
        m["role"]=driver_role
        m["observation_role"]="measurement_driver"
        m["promoted_from_state"]=True
        new_inputs.append(m)
    schema["inputs"]=new_inputs
    schema["mechanism_scope"]={
        "target_states":targets,
        "measurement_drivers":drivers,
        "driver_role":driver_role,
        "driver_sampling":driver_sampling,
        "truth_used":False,
        "state_estimation_applied":False,
    }
    return PublicDataset(X=X,U=U,trajectory_id=public.trajectory_id.copy(),schema=schema,
                         dt=float(public.dt),regime=public.regime,
                         variant=f"{public.variant}-{variant_suffix}",
                         sample_time_s=(None if public.sample_time_s is None else np.asarray(public.sample_time_s,dtype=np.float64).copy()))
