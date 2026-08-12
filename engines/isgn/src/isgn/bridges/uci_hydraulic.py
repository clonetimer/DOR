from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, Sequence
import io, zipfile
import numpy as np
from ..contracts import AlgebraicDataset

_SCOPE_VARS={
    'cooling-power': ('TS3','TS4','FS2','CP'),
    'cooling-efficiency': ('TS3','TS4','CE'),
}
_RATES={'TS3':1,'TS4':1,'CE':1,'CP':1,'FS2':10,'FS1':10,'EPS1':100,'PS2':100}

def available_uci_hydraulic_scopes(): return tuple(_SCOPE_VARS)

def _read(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    with zf.open(name+'.txt') as f: return np.loadtxt(f)

def _to_1hz(a: np.ndarray, rate: int) -> np.ndarray:
    if rate==1: return np.asarray(a,float)
    if a.shape[1] != 60*rate: raise ValueError(f'expected 60 s cycle at {rate} Hz, got {a.shape}')
    return np.asarray(a,float).reshape(a.shape[0],60,rate).mean(axis=2)

def load_uci_hydraulic(source: str|Path, *, scope: str='cooling-power', split: str='train',
                       stable_only: bool=True, seed: int=2026) -> AlgebraicDataset:
    if scope not in _SCOPE_VARS: raise KeyError(f'unknown scope {scope!r}')
    with zipfile.ZipFile(Path(source)) as zf:
        profile=np.asarray(np.loadtxt(zf.open('profile.txt')),int)
        mats={n:_to_1hz(_read(zf,n),_RATES[n]) for n in _SCOPE_VARS[scope]}
    cycles=np.arange(len(profile),dtype=int)
    eligible=(profile[:,4]==0) if stable_only else np.ones(len(profile),dtype=bool)
    cycles=cycles[eligible]
    # cycle-level split: no points from one 60s cycle cross train/val/test
    rng=np.random.default_rng(int(seed)); perm=cycles.copy(); rng.shuffle(perm)
    n=len(perm); ntr=int(round(.70*n)); nv=int(round(.15*n))
    parts={'train':perm[:ntr],'val':perm[ntr:ntr+nv],'test':perm[ntr+nv:]}
    if split not in parts: raise KeyError('split must be train/val/test')
    c=np.sort(parts[split])
    vars=list(_SCOPE_VARS[scope])
    A=np.stack([mats[n][c] for n in vars],axis=-1).reshape(-1,len(vars))
    sid=(c[:,None]*60+np.arange(60)[None,:]).reshape(-1).astype(np.int64)
    cycle_condition={
        'cooler_condition_pct':profile[c,0].tolist(), 'valve_condition_pct':profile[c,1].tolist(),
        'pump_leakage_class':profile[c,2].tolist(), 'accumulator_bar':profile[c,3].tolist(),
    }
    schema={'variables':[{'name':n,'dim':('temperature' if n.startswith('TS') else 'volume_flow' if n.startswith('FS') else 'power')} for n in vars],
            'source':'UCI-447 Condition monitoring of hydraulic systems','scope':scope,'production_real_world':True,
            'sampling_alignment':'block mean to 1 Hz; split by whole 60 s cycles'}
    ds=AlgebraicDataset(A,sid,schema,regime=f'uci-hydraulic-{split}',variant=scope,
                        metadata={'stable_only':bool(stable_only),'cycle_ids':c.tolist(),'cycle_conditions':cycle_condition,'source_archive':str(source)})
    ds.validate(); return ds
