from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Sequence, Optional
import json, re
import numpy as np

from .contracts import PublicDataset

@dataclass
class RelationTask:
    mode: str
    inputs: List[str]
    target: str
    csv_path: str
    rows: int
    variables: List[str]
    def to_dict(self): return asdict(self)


def _names(ds:PublicDataset):
    return [x['name'] for x in ds.schema['states']],[x['name'] for x in ds.schema['inputs']]

def _aligned(ds:PublicDataset,name:str):
    states,inputs=_names(ds)
    if name in states:
        j=states.index(name); return .5*(ds.X[:,:-1,j].astype(float)+ds.X[:,1:,j].astype(float))
    if name in inputs:
        return ds.U[:,:,inputs.index(name)].astype(float)
    raise KeyError(f'{name!r} is not in state/input schema')

def export_explicit_relation(ds:PublicDataset,*,inputs:Sequence[str],target:str,out_csv:str|Path,
                             max_samples:int=10000,seed:int=0) -> RelationTask:
    """Export a SymMatika explicit mapping CSV: inputs first, target last."""
    names=list(inputs)+[target]
    cols=[_aligned(ds,n).reshape(-1) for n in names]
    A=np.stack(cols,axis=1)
    good=np.isfinite(A).all(axis=1); A=A[good]
    if len(A)>max_samples:
        rng=np.random.RandomState(seed); idx=np.sort(rng.choice(len(A),size=max_samples,replace=False)); A=A[idx]
    p=Path(out_csv); p.parent.mkdir(parents=True,exist_ok=True)
    np.savetxt(p,A,delimiter=',')
    task=RelationTask('explicit',list(inputs),target,str(p),int(len(A)),names)
    Path(str(p)+'.task.json').write_text(json.dumps(task.to_dict(),indent=2),encoding='utf-8')
    return task


def validate_expression(ds:PublicDataset,*,inputs:Sequence[str],target:str,expression:str) -> dict:
    """Validate a symbolic expression on ISGN data without merging it into the graph."""
    try: import sympy as sp
    except ModuleNotFoundError as e: raise RuntimeError('symbolic validation requires sympy; install isgn-toolkit[symbolic]') from e
    syms={n:sp.Symbol(n) for n in inputs}
    expr_text=expression.replace('^','**')
    expr=sp.sympify(expr_text,locals={**syms,'sqrt':sp.sqrt,'sin':sp.sin,'cos':sp.cos,'tan':sp.tan,'exp':sp.exp,'log':sp.log,'ln':sp.log,'tanh':sp.tanh})
    unknown={str(s) for s in expr.free_symbols}-set(inputs)
    if unknown: raise ValueError(f'expression contains undeclared symbols: {sorted(unknown)}')
    fn=sp.lambdify([syms[n] for n in inputs],expr,modules=['numpy'])
    xx=[_aligned(ds,n).reshape(-1) for n in inputs]; y=_aligned(ds,target).reshape(-1)
    pred=np.asarray(fn(*xx),dtype=float)
    if pred.ndim==0: pred=np.full_like(y,float(pred))
    pred=np.broadcast_to(pred,y.shape)
    mask=np.isfinite(pred)&np.isfinite(y); pred=pred[mask]; y=y[mask]
    rmse=float(np.sqrt(np.mean((pred-y)**2))); sd=float(np.std(y)+1e-12)
    return {'expression':expression,'target':target,'inputs':list(inputs),'rows':int(len(y)),'rmse':rmse,'nrmse':rmse/sd}
