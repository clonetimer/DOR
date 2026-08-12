from __future__ import annotations

"""Context-conditioned split-conformal bridge for genuine class probabilities.

The existing v2.3 class-score bridge conditions on modality-presence masks.
This v2.4 bridge adds *externally declared operating-context columns* and may
combine them with modality masks. It is deliberately exact-key/fail-closed:
unseen or sparse contexts ABSTAIN; no data-driven pooling is invented inside
FEC.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence
from math import ceil
import csv
import hashlib
import json
import numpy as np

from .class_score_bridge import _read_csv, _class_frame, _probabilities, _mask_columns, _masks


def _sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def _parse_context_columns(value: Sequence[str] | str | None) -> tuple[str,...]:
    if value is None: return ()
    if isinstance(value,str): cols=tuple(x.strip() for x in value.split(',') if x.strip())
    else: cols=tuple(str(x).strip() for x in value if str(x).strip())
    if len(set(cols))!=len(cols): raise ValueError('duplicate context columns')
    return cols


def _context(row: dict[str,str], cols: Sequence[str]) -> tuple[str,...]:
    if not cols: return ('__GLOBAL__',)
    missing=[c for c in cols if c not in row]
    if missing: raise ValueError(f'missing context columns: {missing}')
    return tuple(str(row[c]) for c in cols)


def _key(row: dict[str,str], ctx_cols: Sequence[str], mask: Sequence[int]) -> tuple[tuple[str,...],tuple[int,...]]:
    return _context(row,ctx_cols),tuple(int(x) for x in mask)


@dataclass(frozen=True)
class ContextClassConformalReference:
    contract: str
    class_frame: tuple[str,...]
    context_columns: tuple[str,...]
    mask_columns: tuple[str,...]
    alpha: float
    min_group_size: int
    groups: tuple[dict[str,Any],...]
    calibration_score_sha256: str
    source_score_contract_sha256: str|None
    provenance: str

    def to_dict(self):
        d=asdict(self); d['class_frame']=list(self.class_frame); d['context_columns']=list(self.context_columns); d['mask_columns']=list(self.mask_columns); d['groups']=[dict(x) for x in self.groups]; return d
    @staticmethod
    def from_dict(d):
        if d.get('contract')!='FEC-CONTEXT-CLASS-CONFORMAL-REFERENCE/1.0': raise ValueError('unsupported context class reference contract')
        return ContextClassConformalReference(d['contract'],tuple(map(str,d['class_frame'])),tuple(map(str,d.get('context_columns') or ())),tuple(map(str,d.get('mask_columns') or ())),float(d['alpha']),int(d['min_group_size']),tuple(dict(x) for x in d['groups']),str(d['calibration_score_sha256']),d.get('source_score_contract_sha256'),str(d.get('provenance','unspecified')))


def fit_context_class_score_reference(calibration_csv,output_reference,*,context_columns=None,alpha=.10,min_group_size=20,score_contract=None,provenance='external-selected-classifier'):
    if not 0<alpha<1: raise ValueError('alpha must be in (0,1)')
    if min_group_size<2: raise ValueError('min_group_size must be >=2')
    rows=_read_csv(calibration_csv); frame=_class_frame(rows)
    if 'true_label' not in rows[0]: raise ValueError('calibration score pack requires true_label')
    labels=np.asarray([str(r['true_label']) for r in rows],dtype=object); unknown=sorted(set(labels)-set(frame))
    if unknown: raise ValueError(f'calibration labels outside class_frame: {unknown}')
    label_index={c:i for i,c in enumerate(frame)}; y=np.asarray([label_index[str(x)] for x in labels],int)
    p=_probabilities(rows,frame); mask_cols=_mask_columns(rows); masks=_masks(rows,mask_cols); ctx_cols=_parse_context_columns(context_columns)
    true_scores=1.0-p[np.arange(len(p)),y]
    grouped: dict[tuple[tuple[str,...],tuple[int,...]],list[float]]={}
    for row,mask,score in zip(rows,masks,true_scores): grouped.setdefault(_key(row,ctx_cols,mask),[]).append(float(score))
    groups=[]
    for (ctx,mask),vals in sorted(grouped.items(),key=lambda kv:repr(kv[0])):
        arr=np.sort(np.asarray(vals,float)); n=len(arr)
        if n<min_group_size:
            groups.append({'context':list(ctx),'mask':list(mask),'n_reference':n,'supported':False})
            continue
        rank=min(n,ceil((n+1)*(1-alpha))); threshold=float(arr[rank-1])
        groups.append({'context':list(ctx),'mask':list(mask),'n_reference':n,'supported':True,'threshold':threshold,'alpha':float(alpha)})
    if not any(bool(g['supported']) for g in groups): raise ValueError('no context/mask group has enough labeled calibration samples')
    ref=ContextClassConformalReference('FEC-CONTEXT-CLASS-CONFORMAL-REFERENCE/1.0',tuple(frame),ctx_cols,tuple(mask_cols),float(alpha),int(min_group_size),tuple(groups),_sha256(Path(calibration_csv)),None if score_contract is None else _sha256(Path(score_contract)),str(provenance))
    out=Path(output_reference); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(ref.to_dict(),indent=2,ensure_ascii=False)+'\n',encoding='utf-8'); return out


def infer_context_class_prediction_sets(score_csv,reference_json,output_csv):
    rows=_read_csv(score_csv); ref=ContextClassConformalReference.from_dict(json.loads(Path(reference_json).read_text(encoding='utf-8')))
    frame=_class_frame(rows)
    if tuple(frame)!=ref.class_frame: raise ValueError(f'score class_frame {frame} does not match reference {ref.class_frame}')
    mask_cols=_mask_columns(rows)
    if tuple(mask_cols)!=ref.mask_columns: raise ValueError(f'score modality-mask columns {mask_cols} do not match reference {ref.mask_columns}')
    p=_probabilities(rows,frame); masks=_masks(rows,mask_cols)
    groups={(tuple(map(str,g['context'])),tuple(int(x) for x in g['mask'])):g for g in ref.groups}
    out_rows=[]
    for i,(row,mask) in enumerate(zip(rows,masks)):
        key=_key(row,ref.context_columns,mask); g=groups.get(key)
        base={k:v for k,v in row.items() if not k.startswith('score__') and k!='true_label'}
        top=int(np.argmax(p[i])); base.update({'top_class':frame[top],'top_score':float(p[i,top]),'calibration_context':json.dumps(list(key[0]),ensure_ascii=False),'modality_mask':json.dumps(list(key[1]))})
        if g is None or not bool(g.get('supported')):
            n=0 if g is None else int(g.get('n_reference',0)); base.update({'prediction_set':'[]','prediction_set_size':0,'calibration_supported':False,'calibration_n_reference':n,'calibration_reason':f'context/mask {key!r} lacks supported labeled calibration','prediction_set_status':'ABSTAIN'})
        else:
            score=1.0-p[i]; idx=tuple(int(k) for k in np.flatnonzero(score<=float(g['threshold']))); labs=[frame[j] for j in idx]
            base.update({'prediction_set':json.dumps(labs,ensure_ascii=False),'prediction_set_size':len(labs),'calibration_supported':True,'calibration_n_reference':int(g['n_reference']),'calibration_reason':'exact context-conditioned class conformal reference supported','prediction_set_status':'CLASS_SET_SUPPORTED' if labs else 'ABSTAIN'})
        if 'true_label' in row: base['true_label']=row['true_label']
        out_rows.append(base)
    out=Path(output_csv); out.parent.mkdir(parents=True,exist_ok=True); fields=list(out_rows[0].keys())
    with out.open('w',encoding='utf-8',newline='') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(out_rows)
    return out
