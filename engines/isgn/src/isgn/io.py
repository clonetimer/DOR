from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import json
import numpy as np
from .contracts import PublicDataset, AlgebraicDataset

def from_arrays(X, U, *, schema: Dict[str,Any], dt: float, trajectory_id=None,
                regime: str="user", variant: str="user") -> PublicDataset:
    X=np.asarray(X,dtype=np.float32); U=np.asarray(U,dtype=np.float32)
    if X.ndim != 3 or U.ndim != 3: raise ValueError("X and U must be rank-3 arrays")
    if X.shape[0] != U.shape[0] or X.shape[1] != U.shape[1]+1:
        raise ValueError("expected X[N,T,D], U[N,T-1,Ud]")
    if len(schema.get("states",[])) != X.shape[2] or len(schema.get("inputs",[])) != U.shape[2]:
        raise ValueError("schema dimensions do not match X/U")
    ids=np.arange(X.shape[0],dtype=np.int64) if trajectory_id is None else np.asarray(trajectory_id)
    return PublicDataset(X=X,U=U,trajectory_id=ids,schema=schema,dt=float(dt),regime=regime,variant=variant)

def save_npz(ds: PublicDataset, path: str|Path) -> None:
    np.savez_compressed(Path(path), X=ds.X, U=ds.U, trajectory_id=ds.trajectory_id,
                        schema_json=np.array(json.dumps(ds.schema)), dt=np.array(ds.dt),
                        regime=np.array(ds.regime), variant=np.array(ds.variant))

def load_npz(path: str|Path) -> PublicDataset:
    z=np.load(Path(path),allow_pickle=False)
    return PublicDataset(X=z["X"],U=z["U"],trajectory_id=z["trajectory_id"],
                         schema=json.loads(str(z["schema_json"])),dt=float(z["dt"]),
                         regime=str(z["regime"]),variant=str(z["variant"]))


def save_algebraic_npz(ds: AlgebraicDataset, path: str|Path) -> None:
    ds.validate()
    np.savez_compressed(Path(path), values=ds.values, sample_id=ds.sample_id,
                        schema_json=np.array(json.dumps(ds.schema)), regime=np.array(ds.regime),
                        variant=np.array(ds.variant), metadata_json=np.array(json.dumps(ds.metadata)))

def load_algebraic_npz(path: str|Path) -> AlgebraicDataset:
    z=np.load(Path(path),allow_pickle=False)
    ds=AlgebraicDataset(values=z["values"], sample_id=z["sample_id"],
                        schema=json.loads(str(z["schema_json"])), regime=str(z["regime"]),
                        variant=str(z["variant"]), metadata=json.loads(str(z["metadata_json"])))
    ds.validate(); return ds
