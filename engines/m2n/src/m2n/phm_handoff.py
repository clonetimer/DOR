from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Mapping, Any
import csv, json
import numpy as np
import torch

from .model_evidence import RobustSupportReference


@dataclass(frozen=True)
class EvidenceContract:
    contract: str
    output_names: tuple[str,...]
    support_input_columns: tuple[str,...]
    semantics: dict[str,str]
    prohibited_claims: tuple[str,...]
    def to_dict(self): return asdict(self)


def fit_support_reference(data: Mapping[str,np.ndarray], columns: tuple[str,...], idx: np.ndarray, quantile: float=0.99) -> RobustSupportReference:
    if not columns: raise ValueError('support reference needs at least one observed input column')
    X=np.column_stack([np.asarray(data[c],float)[idx] for c in columns])
    return RobustSupportReference.fit(X,quantile=quantile)


def save_support_reference(ref: RobustSupportReference, columns: tuple[str,...], target_scales: Mapping[str,float], path: str|Path):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    payload={'contract':'M2N-NORMAL-SUPPORT/2.0','columns':list(columns),'reference':ref.to_dict(),'target_scales':{str(k):float(v) for k,v in target_scales.items()}}
    p.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    return p


def load_support_reference(path: str|Path):
    d=json.loads(Path(path).read_text(encoding='utf-8'))
    r=d['reference']; ref=RobustSupportReference(tuple(r['median']),tuple(r['scale']),float(r['distance_threshold']))
    return ref,tuple(d['columns']),{str(k):float(v) for k,v in d.get('target_scales',{}).items()}


def write_evidence_csv(path: str|Path, *, time: np.ndarray, output_names: tuple[str,...], target: np.ndarray, predictions: Mapping[str,np.ndarray], support_distance: np.ndarray, support_ood: np.ndarray, target_scales: Mapping[str,float], independent_context: np.ndarray | None=None):
    ids=tuple(predictions)
    stack=np.stack([np.asarray(predictions[k],float) for k in ids],axis=0) # [M,N,D]
    target=np.asarray(target,float)
    if target.ndim==1: target=target[:,None]
    scales=np.asarray([max(float(target_scales[n]),1e-12) for n in output_names],float)
    resid=np.abs(stack-target[None,:,:])/scales[None,None,:]
    best_channel=np.min(resid,axis=0)
    candidate_rms=np.sqrt(np.mean(resid**2,axis=2))
    best_rms=np.min(candidate_rms,axis=0)
    med_rms=np.median(candidate_rms,axis=0)
    disagreement=np.sqrt(np.mean((np.std(stack,axis=0)/scales[None,:])**2,axis=1)) if len(ids)>1 else np.zeros(target.shape[0])
    context=None if independent_context is None else np.asarray(independent_context)
    if context is not None and len(context)!=len(time):
        raise ValueError('independent_context length must match evidence time length')
    fields=['time_s']
    if context is not None: fields.append('m2n_independent_context')
    fields += ['m2n_best_finite_residual','m2n_median_finite_residual','m2n_model_disagreement','m2n_normal_support_distance','m2n_normal_support_ood','m2n_mechanism_evidence_in_support']
    fields += [f'm2n_best_finite_residual__{n}' for n in output_names]
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for i in range(len(time)):
            row={'time_s':float(time[i]),'m2n_best_finite_residual':float(best_rms[i]),'m2n_median_finite_residual':float(med_rms[i]),'m2n_model_disagreement':float(disagreement[i]),'m2n_normal_support_distance':float(support_distance[i]),'m2n_normal_support_ood':int(bool(support_ood[i])),'m2n_mechanism_evidence_in_support':int(not bool(support_ood[i]))}
            if context is not None:
                v=context[i]
                row['m2n_independent_context']=v.item() if isinstance(v,np.generic) else v
            for j,n in enumerate(output_names): row[f'm2n_best_finite_residual__{n}']=float(best_channel[i,j])
            w.writerow(row)
    return p


def export_fec_contact_pack(evidence_csv: str|Path, output_csv: str|Path, contract_json: str|Path, *, window_s: float=0.05, min_samples: int=4):
    """Convert sample-level M2N evidence into a generic FEC-compatible Contact Pack.

    The function intentionally stops before FEC inference.  M2N OOD/support is
    carried as reference-compatibility context rather than a duplicated anomaly
    Contact, matching the validated M2N->FEC bridge semantics.
    """
    import pandas as pd
    df=pd.read_csv(evidence_csv)
    if 'time_s' not in df: raise KeyError('time_s missing from M2N evidence')
    contacts=[c for c in df.columns if c.startswith('m2n_best_finite_residual__')]
    if not contacts: raise ValueError('no mechanism residual contacts found')
    t=np.asarray(df['time_s'],float)
    context_col='m2n_independent_context' if 'm2n_independent_context' in df else None
    groups=[(None,np.arange(len(df)))] if context_col is None else [(cid,np.flatnonzero(np.asarray(df[context_col],dtype=object)==cid)) for cid in pd.unique(df[context_col])]
    rows=[]
    for cid,ii in groups:
        tg=t[ii]
        if len(tg)>1 and np.any(np.diff(tg)<=0):
            raise ValueError('time_s must be strictly increasing within each independent context before FEC windowing')
        start=float(tg[0]); bins=np.floor((tg-start)/float(window_s)+1e-12).astype(int)
        for b in np.unique(bins):
            local=bins==b
            if int(np.sum(local))<min_samples: continue
            m=ii[local]
            r={'time_s':float(np.mean(t[m]))}
            if context_col is not None: r[context_col]=cid
            for c in contacts+['m2n_model_disagreement']:
                x=np.asarray(df.loc[m,c],float); r[c]=float(np.sqrt(np.mean(x*x)))
            r['m2n_normal_support_distance']=float(np.median(np.asarray(df.loc[m,'m2n_normal_support_distance'],float)))
            r['m2n_normal_support_ood_fraction']=float(np.mean(np.asarray(df.loc[m,'m2n_normal_support_ood'],int)>0))
            r['m2n_mechanism_evidence_in_support_fraction']=float(np.mean(np.asarray(df.loc[m,'m2n_mechanism_evidence_in_support'],int)>0)) if 'm2n_mechanism_evidence_in_support' in df else 1.0-r['m2n_normal_support_ood_fraction']
            rows.append(r)
    if len(rows)<2: raise ValueError('FEC contact windowing produced fewer than two windows')
    out=pd.DataFrame(rows); Path(output_csv).parent.mkdir(parents=True,exist_ok=True); out.to_csv(output_csv,index=False)
    contract=EvidenceContract(
        'M2N-FEC-CONTACT-PACK/2.1',tuple(c.split('__',1)[1] for c in contacts),(),
        semantics={**{c:'window RMS of the best admissible finite/local M2N mechanism residual; anomaly evidence candidate, not a fault probability' for c in contacts},'m2n_model_disagreement':'window RMS of admissible architecture prediction spread','m2n_normal_support_distance':'reference compatibility input; do not fuse as anomaly evidence','m2n_normal_support_ood_fraction':'reference compatibility/OOD context; do not fuse as anomaly evidence','m2n_mechanism_evidence_in_support_fraction':'fraction of samples in this window that remain inside the frozen normal-support domain; mechanism residual/disagreement interpretation must be qualified by this value',**({'m2n_independent_context':'independent run/cycle identifier preserved so evidence windows never cross context boundaries'} if context_col is not None else {})},
        prohibited_claims=('fault probability','fault class','RUL','physical parameter drift unless separately qualified'),
    )
    Path(contract_json).write_text(json.dumps(contract.to_dict(),indent=2,ensure_ascii=False),encoding='utf-8')
    return Path(output_csv),Path(contract_json)
