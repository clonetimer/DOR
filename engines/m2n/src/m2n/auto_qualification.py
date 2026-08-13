from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Mapping, Sequence
import math
import numpy as np


@dataclass(frozen=True)
class AutoQualificationResult:
    status: str  # READY | AMBIGUOUS | UNSUPPORTED
    observation_status: str  # SUPPORTED | PARTIAL | UNSUPPORTED
    reasons: tuple[str,...]
    diagnostics: dict
    def to_dict(self): return asdict(self)


def _numeric(a) -> bool:
    return np.asarray(a).dtype.kind in 'fiu'


def qualify_tabular_design(
    data: Mapping[str,np.ndarray],
    *,
    required_inputs: Sequence[str],
    target_columns: Sequence[str],
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    independent_context: np.ndarray | None=None,
    independent_context_evidence_scope: str='PRODUCTION',
    min_train_independent_contexts: int | None=None,
    min_validation_independent_contexts: int | None=None,
    min_train_rows: int=50,
    min_validation_rows: int=20,
    min_target_std: float=1e-10,
    max_validation_outside_train_fraction: float=0.25,
    max_standardized_condition: float=1e8,
) -> AutoQualificationResult:
    """Automatic observation/data qualification for declarative M2N design.

    This is deliberately conservative and transparent.  It checks whether the
    columns needed by the declared mechanism are actually observed, whether the
    supervised target varies, whether the normal validation set remains mostly
    inside the commissioning support, and whether the observed input design is
    exactly/near rank deficient.  It does not claim global structural
    identifiability.
    """
    req=tuple(dict.fromkeys(map(str,required_inputs)))
    tgt=tuple(dict.fromkeys(map(str,target_columns)))
    allcols=req+tgt
    missing=[c for c in allcols if c not in data]
    diag={'n_train':int(len(train_idx)),'n_validation':int(len(validation_idx)),'required_inputs':list(req),'target_columns':list(tgt)}
    if missing:
        diag['missing_columns']=missing
        return AutoQualificationResult('UNSUPPORTED','UNSUPPORTED',(f'missing required observed columns: {missing}',),diag)
    for c in allcols:
        if not _numeric(data[c]):
            return AutoQualificationResult('UNSUPPORTED','UNSUPPORTED',(f'required column {c!r} is not numeric',),diag)
        a=np.asarray(data[c],float)
        use=np.r_[train_idx,validation_idx]
        if not np.all(np.isfinite(a[use])):
            return AutoQualificationResult('UNSUPPORTED','PARTIAL',(f'non-finite values in required column {c!r}',),diag)
    reasons=[]
    scope=str(independent_context_evidence_scope or 'PRODUCTION').upper()
    if scope not in ('PRODUCTION','BLINDED_FOR_VALIDATION'):
        return AutoQualificationResult(
            'UNSUPPORTED','PARTIAL',
            (f'unsupported independent-context evidence scope: {scope!r}',),diag,
        )
    diag['independent_context_evidence_scope']=scope
    if scope=='BLINDED_FOR_VALIDATION' and independent_context is None:
        return AutoQualificationResult(
            'UNSUPPORTED','PARTIAL',
            ('BLINDED_FOR_VALIDATION requires an explicit independent-context column; row-level samples cannot establish controlled context-to-context falsification',),diag,
        )
    if len(train_idx)<min_train_rows or len(validation_idx)<min_validation_rows:
        reasons.append(f'insufficient rows: train={len(train_idx)}, validation={len(validation_idx)}')
    if independent_context is not None:
        context=np.asarray(independent_context)
        n=len(next(iter(data.values()))) if data else 0
        if len(context)!=n:
            return AutoQualificationResult(
                'UNSUPPORTED','PARTIAL',
                (f'independent context vector length {len(context)} does not match data rows {n}',),diag,
            )
        use=np.r_[train_idx,validation_idx]
        if context.dtype.kind in 'fiu':
            bad=~np.isfinite(np.asarray(context,float)[use])
        else:
            selected=np.asarray(context,dtype=object)[use]
            bad=np.asarray([x is None or not str(x).strip() for x in selected],dtype=bool)
        if np.any(bad):
            return AutoQualificationResult(
                'UNSUPPORTED','PARTIAL',
                ('independent context column contains missing/non-finite values in train or validation rows',),diag,
            )
        train_context=np.unique(context[train_idx])
        validation_context=np.unique(context[validation_idx])
        diag['n_train_independent_contexts']=int(len(train_context))
        diag['n_validation_independent_contexts']=int(len(validation_context))
        diag['train_rows_per_independent_context']={
            'min':int(min(np.sum(context[train_idx]==x) for x in train_context)),
            'max':int(max(np.sum(context[train_idx]==x) for x in train_context)),
        } if len(train_context) else {'min':0,'max':0}
        diag['validation_rows_per_independent_context']={
            'min':int(min(np.sum(context[validation_idx]==x) for x in validation_context)),
            'max':int(max(np.sum(context[validation_idx]==x) for x in validation_context)),
        } if len(validation_context) else {'min':0,'max':0}
        overlap=np.intersect1d(train_context,validation_context)
        diag['independent_context_split_overlap_count']=int(len(overlap))
        if len(overlap):
            return AutoQualificationResult(
                'UNSUPPORTED','PARTIAL',
                (f'independent contexts overlap train and validation splits ({len(overlap)} overlapping contexts); validation is not independent',),diag,
            )
        missing_min=(min_train_independent_contexts is None, min_validation_independent_contexts is None)
        if missing_min[0] != missing_min[1]:
            return AutoQualificationResult(
                'UNSUPPORTED','PARTIAL',
                ('independent-context minimums must be supplied together or both omitted',),diag,
            )
        if not any(missing_min):
            mt=int(min_train_independent_contexts); mv=int(min_validation_independent_contexts)
            if mt < 1 or mv < 1:
                return AutoQualificationResult(
                    'UNSUPPORTED','PARTIAL',
                    ('independent-context minimums must both be positive integers',),diag,
                )
            diag['min_train_independent_contexts']=mt
            diag['min_validation_independent_contexts']=mv
            context_minimums_met=len(train_context)>=mt and len(validation_context)>=mv
            diag['production_independent_context_qualified']=bool(scope=='PRODUCTION' and context_minimums_met)
            if not context_minimums_met:
                reasons.append(
                    'insufficient independent contexts: '
                    f'train={len(train_context)} (<{mt}), validation={len(validation_context)} (<{mv})'
                )
        elif scope=='PRODUCTION':
            diag['production_independent_context_qualified']=False
            reasons.append(
                'production independent-context qualification requires evidence-justified '
                'min_train_independent_contexts and min_validation_independent_contexts; qualification fails closed'
            )
        else:
            diag['production_independent_context_qualified']=False
            diag['qualification_limitation']=(
                'BLINDED_FOR_VALIDATION permits controlled architecture falsification without claiming production sample adequacy'
            )
            if len(train_context)<2 or len(validation_context)<2:
                reasons.append(
                    'BLINDED_FOR_VALIDATION needs at least two disjoint train and two validation independent contexts '
                    'to audit context-to-context stability'
                )
    target_std={c:float(np.std(np.asarray(data[c],float)[train_idx])) for c in tgt}
    diag['target_train_std']=target_std
    for c,s in target_std.items():
        if not math.isfinite(s) or s<=min_target_std:
            reasons.append(f'target {c!r} has insufficient training variation')
    if req:
        X=np.column_stack([np.asarray(data[c],float)[train_idx] for c in req])
        ranges={c:[float(np.min(np.asarray(data[c],float)[train_idx])),float(np.max(np.asarray(data[c],float)[train_idx]))] for c in req}
        diag['input_train_ranges']=ranges
        for c,(lo,hi) in ranges.items():
            if hi-lo<=1e-12: reasons.append(f'input {c!r} has insufficient training variation')
        center=np.mean(X,axis=0); scale=np.std(X,axis=0); scale=np.where(scale>1e-12,scale,1.0)
        Z=(X-center)/scale
        svals=np.linalg.svd(Z,full_matrices=False,compute_uv=False) if Z.size else np.array([])
        rank=int(np.linalg.matrix_rank(Z)) if Z.size else 0
        cond=float(svals[0]/max(svals[-1],1e-15)) if len(svals)>1 else 1.0
        diag['standardized_input_rank']=rank; diag['standardized_input_dimension']=len(req); diag['standardized_input_condition']=cond
        if len(req)>1 and rank<len(req): reasons.append('observed input design is rank deficient; mechanism attribution is not identifiable from this commissioning sample')
        elif len(req)>1 and cond>max_standardized_condition: reasons.append(f'observed input design is ill-conditioned (condition={cond:.3g})')
        V=np.column_stack([np.asarray(data[c],float)[validation_idx] for c in req])
        lo=np.min(X,axis=0); hi=np.max(X,axis=0)
        outside=np.any((V<lo[None,:])|(V>hi[None,:]),axis=1)
        frac=float(np.mean(outside)) if len(V) else 1.0
        diag['validation_outside_train_box_fraction']=frac
        if frac>max_validation_outside_train_fraction:
            reasons.append(f'validation support exceeds commissioning box for {frac:.1%} of rows')
    return AutoQualificationResult('AMBIGUOUS' if reasons else 'READY','SUPPORTED',tuple(reasons),diag)
