from __future__ import annotations
from pathlib import Path
from typing import Dict, Sequence
import numpy as np
from ..contracts import AlgebraicDataset

EUV_SCOPES={
    'plasma-energy': ('ce','E_in','E_IB'),
    'collector-transfer': ('E_IB','E_IF'),
    'metrology': ('E_IF','meas_E'),
}

def available_euv_scopes() -> tuple[str,...]: return tuple(EUV_SCOPES)

def load_euv_scope(source: str|Path, *, scope: str='plasma-energy', split: str='train',
                   normal_only: bool=True, fired_only: bool=True, max_samples: int=50000,
                   seed: int=0) -> AlgebraicDataset:
    root=Path(source)
    p=root/'core'/'shots.npz' if root.is_dir() else root
    if scope not in EUV_SCOPES: raise KeyError(f'unknown EUV scope {scope!r}; available={available_euv_scopes()}')
    with np.load(p,allow_pickle=True) as z:
        names=[str(x) for x in z['feature_names'].tolist()]
        want=list(EUV_SCOPES[scope]); idx=[names.index(n) for n in want]
        mask=np.asarray(z['split']).astype(str)==str(split)
        if normal_only: mask &= np.asarray(z['is_normal']).astype(int)==1
        Xall=np.asarray(z['X'])
        if fired_only: mask &= Xall[:,names.index('fired')]>0.5
        A=np.asarray(Xall[mask][:,idx],dtype=np.float64)
        sample=np.arange(len(Xall),dtype=np.int64)[mask]
    good=np.isfinite(A).all(axis=1); A=A[good]; sample=sample[good]
    if max_samples and len(A)>max_samples:
        rng=np.random.default_rng(int(seed)); take=np.sort(rng.choice(len(A),size=int(max_samples),replace=False)); A=A[take]; sample=sample[take]
    schema={'variables':[{'name':n,'dim':'dimensionless' if n=='ce' else 'energy'} for n in want],
            'source':'EUV-Source-Sim','scope':scope,'production_real_world':False}
    ds=AlgebraicDataset(A,sample,schema,regime=f'euv-{split}',variant=scope,
                        metadata={'split':split,'normal_only':bool(normal_only),'fired_only':bool(fired_only),'source_file':str(p)})
    ds.validate(); return ds
