from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Any, Mapping
import copy
import csv
import hashlib
import json
import math

import numpy as np
import torch

from .canonical_ir import Dim, CanonicalIR
from .knowledge_contract import (
    KnowledgeLevel, KnowledgeSource, MechanismSlot, SlotEvidence,
    NeuralArchitecturePlan, plan_neural_architecture,
)
from .mechanics_ir import (
    CoordinateSpec, InputSpec, DriveSpec, ConstantLoadSpec, ResistanceSpec,
    SecondOrderMechanicsIR,
)
from .peaked_response_ir import PeakAxisSpec, SeparablePeakIR
from .generic_graph import build_generic_problem
from .auto_qualification import qualify_tabular_design
from .backend_router import run_backend_requests, assess_model_set_routing
from .qualified_k3 import K3QualificationPolicy, finite_gate, compare_k3
from .bias_routing import build_bias_route_plan
from .representation_policy import build_representation_plan, k3_representation_for_slot
from .exact_physics import audit_exact_physics
from .phm_handoff import fit_support_reference, save_support_reference, load_support_reference, write_evidence_csv
from .architecture_lowering import FieldAssignment, FamilyPatch, SlotBinding, lower_architecture_plan, compile_executable_ir
from .selection import ArchitectureSelectionPolicy
from .design_session import run_design_session


SUPPORTED_TEMPLATES = {'second_order_mechanics', 'separable_peak_response', 'generic_mechanism_graph'}
SUPPORTED_SCHEMA_VERSIONS = ('1.0','2.0','2.1','2.2','2.3','2.4','2.5','2.6','2.7','2.8','2.9','2.10','2.11','2.12')


def _sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''):
            h.update(b)
    return h.hexdigest()


def _dim(obj: Mapping[str, Any] | None) -> Dim:
    obj=obj or {}
    return Dim.of(**obj)


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        r=csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError('data CSV has no header')
        rows=list(r)
    out={k:[] for k in r.fieldnames}
    for row in rows:
        for k in r.fieldnames:
            out[k].append(row[k])
    arrays={}
    for k,v in out.items():
        # Keep split/context labels as strings; numeric columns become float64.
        try:
            arrays[k]=np.asarray([float(x) for x in v],dtype=np.float64)
        except Exception:
            arrays[k]=np.asarray(v,dtype=object)
    return arrays


def _evidence_coordinates(spec: Mapping[str,Any], data: Mapping[str,np.ndarray]) -> tuple[np.ndarray, np.ndarray | None, str | None]:
    """Resolve the sample time and independent-context coordinates for outputs.

    ``data.time_column`` is authoritative when declared.  Otherwise common time
    column names and the second-order derivative time column are used.  Local
    time is allowed to reset between declared independent contexts; callers that
    aggregate over time must preserve the returned context vector.
    """
    n=len(next(iter(data.values()))) if data else 0
    dcfg=spec.get('data',{}) or {}
    qcfg=spec.get('qualification',{}) or {}
    context_col=qcfg.get('independent_context_column')
    context=None
    if context_col:
        context_col=str(context_col)
        if context_col not in data:
            raise ValueError(f'missing declared independent context column {context_col!r}')
        context=np.asarray(data[context_col])
        if len(context)!=n:
            raise ValueError('independent context length does not match data rows')
        if context.dtype.kind in 'fiu':
            if not np.all(np.isfinite(np.asarray(context,float))):
                raise ValueError(f'independent context column {context_col!r} contains non-finite values')
        elif any(x is None or not str(x).strip() for x in np.asarray(context,dtype=object)):
            raise ValueError(f'independent context column {context_col!r} contains missing values')

    declared_time=dcfg.get('time_column')
    candidates=[]
    if declared_time:
        candidates.append(str(declared_time))
    candidates.extend(['time_s','t_s','time','timestamp'])
    if spec.get('template')=='second_order_mechanics':
        der=(spec.get('supervision',{}) or {}).get('derive_acceleration') or {}
        if der.get('time_column'):
            candidates.append(str(der['time_column']))
    time_col=None
    for c in dict.fromkeys(candidates):
        if c in data:
            time_col=c
            break
    if declared_time and str(declared_time) not in data:
        raise ValueError(f'missing declared time column {str(declared_time)!r}')
    if time_col is None:
        return np.arange(n,dtype=float),context,None
    try:
        t=np.asarray(data[time_col],float)
    except Exception as e:
        raise ValueError(f'time column {time_col!r} is not numeric') from e
    if len(t)!=n or not np.all(np.isfinite(t)):
        raise ValueError(f'time column {time_col!r} contains invalid values')
    if context is not None:
        for cid in np.unique(context):
            ii=np.flatnonzero(context==cid)
            if len(ii)>1 and np.any(np.diff(t[ii])<=0):
                raise ValueError(f'time column {time_col!r} must be strictly increasing within independent context {cid!r}')
    return t,context,time_col


def _split_indices(data: Mapping[str,np.ndarray], cfg: Mapping[str,Any]) -> tuple[np.ndarray,np.ndarray]:
    if 'split_column' in cfg:
        col=str(cfg['split_column'])
        if col not in data: raise ValueError(f'missing split column {col!r}')
        tr=np.isin(data[col],np.asarray(cfg.get('train_values',['train']),dtype=object))
        va=np.isin(data[col],np.asarray(cfg.get('validation_values',['val','validation']),dtype=object))
        tri=np.flatnonzero(tr); vai=np.flatnonzero(va)
    else:
        n=len(next(iter(data.values())))
        seed=int(cfg.get('split_seed',20260811)); frac=float(cfg.get('train_fraction',0.8))
        if not 0.1 <= frac <= 0.95: raise ValueError('train_fraction must lie in [0.1,0.95]')
        idx=np.arange(n); np.random.default_rng(seed).shuffle(idx); cut=max(1,min(n-1,int(round(frac*n))))
        tri=np.sort(idx[:cut]); vai=np.sort(idx[cut:])
    if len(tri)<2 or len(vai)<2: raise ValueError('train and validation splits each need at least two rows')
    return tri,vai


def _prepare_derived_supervision(spec: dict[str,Any], data: dict[str,np.ndarray]) -> tuple[dict[str,Any], tuple[str,...]]:
    """Materialize optional derived supervision without changing the input CSV.

    This is an Estimation Contract convenience, not a physical truth channel.
    """
    if spec['template']!='second_order_mechanics': return spec,()
    sup=spec['supervision']
    if sup.get('target_acceleration_column'):
        return spec,()
    cfg=sup.get('derive_acceleration')
    if not cfg:
        raise ValueError('second_order_mechanics supervision needs target_acceleration_column or derive_acceleration')
    vcol=str(cfg.get('velocity_column',sup['velocity_column'])); tcol=str(cfg['time_column'])
    if vcol not in data or tcol not in data: raise ValueError(f'derived acceleration needs columns {vcol!r} and {tcol!r}')
    v=np.asarray(data[vcol],dtype=float); t=np.asarray(data[tcol],dtype=float)
    if len(v)<3 or len(t)!=len(v) or not np.all(np.isfinite(v)) or not np.all(np.isfinite(t)):
        raise ValueError('derived acceleration requires >=3 finite aligned time/velocity samples')
    if np.any(np.diff(t)<=0): raise ValueError('time column must be strictly increasing to derive acceleration')
    method=str(cfg.get('method','gradient'))
    if method=='gradient':
        a=np.gradient(v,t,edge_order=2)
    elif method=='savgol':
        from scipy.signal import savgol_filter
        window=int(cfg.get('window_length',11)); poly=int(cfg.get('polyorder',3))
        if window%2==0: window+=1
        if window<=poly or window>len(v): raise ValueError('invalid Savitzky-Golay window/polyorder for acceleration derivation')
        dt=float(np.median(np.diff(t)))
        a=savgol_filter(v,window_length=window,polyorder=poly,deriv=1,delta=dt,mode='interp')
    else:
        raise ValueError("derive_acceleration.method must be 'gradient' or 'savgol'")
    key='__m2n_derived_acceleration__'; data[key]=np.asarray(a,dtype=np.float64)
    out=copy.deepcopy(spec); out['supervision']['target_acceleration_column']=key
    return out,(f'acceleration supervision derived from {vcol} versus {tcol} using {method}',)


def _required_columns(spec: Mapping[str,Any]) -> tuple[str,...]:
    sup=spec['supervision']; kind=spec['template']
    cols=[]
    if kind=='second_order_mechanics':
        cols += [sup['position_column'],sup['velocity_column']]
        if sup.get('target_acceleration_column'): cols.append(sup['target_acceleration_column'])
        elif sup.get('derive_acceleration'):
            cols += [sup['derive_acceleration']['time_column'],sup['derive_acceleration'].get('velocity_column',sup['velocity_column'])]
        cols += list(sup.get('input_columns',{}).values())
    elif kind=='separable_peak_response':
        cols += list(sup['input_columns'].values())+[sup['target_column']]
    elif kind=='generic_mechanism_graph':
        cols += list(sup['input_columns'].values())+list(sup['output_columns'].values())
    return tuple(dict.fromkeys(map(str,cols)))


def _basic_data_qualification(spec: Mapping[str,Any], data: Mapping[str,np.ndarray], tri:np.ndarray, vai:np.ndarray) -> tuple[str,str,tuple[str,...],dict[str,Any]]:
    required=_required_columns(spec)
    kind=spec['template']; sup=spec['supervision']
    qcfg=spec.get('qualification',{})
    context_col=qcfg.get('independent_context_column')
    if context_col is not None and str(context_col) not in data:
        name=str(context_col)
        return 'UNSUPPORTED','PARTIAL',(f'missing declared independent context column: {name!r}',),{
            'missing_independent_context_column':name,'n_train':int(len(tri)),'n_validation':int(len(vai)),
        }
    missing=[c for c in required if c not in data]
    if missing:
        return 'UNSUPPORTED','UNSUPPORTED',(f'missing required observed columns: {missing}',),{'missing_columns':missing,'n_train':int(len(tri)),'n_validation':int(len(vai))}
    for c in required:
        a=np.asarray(data[c])
        if a.dtype.kind not in 'fiu' or not np.all(np.isfinite(np.asarray(a,float)[np.r_[tri,vai]])):
            return 'UNSUPPORTED','PARTIAL',(f'required observed column {c!r} is non-numeric or non-finite',),{'n_train':int(len(tri)),'n_validation':int(len(vai))}
    if kind=='second_order_mechanics':
        # q is required to execute the state wrapper but does not enter the
        # current acceleration constitutive attribution; do not create a false
        # rank deficiency when q is intentionally held fixed in commissioning.
        inputs=[sup['velocity_column'],*sup.get('input_columns',{}).values()]
        targets=[sup['target_acceleration_column']]
        has_structure_search=str(spec.get('knowledge',{}).get('resistance',{}).get('level','K1')) in ('K2','K3')
    elif kind=='separable_peak_response':
        inputs=list(sup['input_columns'].values()); targets=[sup['target_column']]; has_structure_search=True
    else:
        inputs=list(sup['input_columns'].values()); targets=list(sup['output_columns'].values()); has_structure_search=any(str(x.get('level')) in ('K2','K3') for x in spec.get('knowledge',{}).get('slots',[]))
    max_out=float(qcfg.get('max_validation_outside_train_fraction',0.25)) if has_structure_search else 1.0
    min_train_contexts=qcfg.get('min_train_independent_contexts')
    min_validation_contexts=qcfg.get('min_validation_independent_contexts')
    q=qualify_tabular_design(
        data,required_inputs=inputs,target_columns=targets,train_idx=tri,validation_idx=vai,
        independent_context=None if context_col is None else np.asarray(data[str(context_col)]),
        independent_context_evidence_scope=str(qcfg.get('evidence_scope','PRODUCTION')),
        min_train_independent_contexts=None if min_train_contexts is None else int(min_train_contexts),
        min_validation_independent_contexts=None if min_validation_contexts is None else int(min_validation_contexts),
        min_train_rows=int(qcfg.get('min_train_rows',50)),min_validation_rows=int(qcfg.get('min_validation_rows',20)),
        min_target_std=float(qcfg.get('min_target_std',1e-10)),
        max_validation_outside_train_fraction=max_out,
        max_standardized_condition=float(qcfg.get('max_standardized_condition',1e8)),
    )
    reasons=list(q.reasons); diagnostics=dict(q.diagnostics)
    if q.status=='UNSUPPORTED':
        return q.status,q.observation_status,tuple(reasons),diagnostics

    # Template-specific excitation rules.  They complement rather than replace
    # the generic observed-column/rank/support audit above.
    if kind=='second_order_mechanics':
        vcol=sup['velocity_column']; v=np.asarray(data[vcol],float)[tri]
        diagnostics['velocity_min']=float(np.min(v)); diagnostics['velocity_max']=float(np.max(v))
        k2=spec.get('knowledge',{}).get('resistance',{})
        fam=set(k2.get('finite_candidates',[]))
        if fam & {'viscous_coulomb','asymmetric_viscous_coulomb','stribeck'}:
            pos=int(np.sum(v>0)); neg=int(np.sum(v<0)); diagnostics['velocity_positive_count']=pos; diagnostics['velocity_negative_count']=neg
            if pos < 5 or neg < 5:
                reasons.append('friction-family attribution needs both velocity directions under the default qualification audit')
    elif kind=='separable_peak_response':
        axes=spec['mechanism']['axes']; inputs_map=sup['input_columns']; ranges={}
        for a in axes:
            name=a['name']; col=inputs_map[name]; x=np.asarray(data[col],float)[tri]
            ranges[name]=[float(np.min(x)),float(np.max(x))]
            if np.ptp(x)<=1e-12: reasons.append(f'axis {name!r} has insufficient training variation')
        diagnostics['axis_train_ranges']=ranges
    return ('AMBIGUOUS' if reasons else 'READY'),q.observation_status,tuple(dict.fromkeys(reasons)),diagnostics

def _mechanics_problem(spec:Mapping[str,Any], data_q:str, obs_q:str, reasons:tuple[str,...], *, finite_status_override:str|None=None):
    m=spec['mechanism']; know=spec.get('knowledge',{})
    qdim=_dim(m['coordinate'].get('position_dim',{'length':1})); vdim=qdim/Dim.of(time=1); adim=qdim/(Dim.of(time=1)**2)
    inputs=tuple(InputSpec(x['name'],_dim(x.get('dim'))) for x in m['inputs'])
    drive_cfg=m['drive']
    drive=DriveSpec(drive_cfg.get('id','drive'),drive_cfg['input'],_dim(drive_cfg.get('gain_dim')),float(drive_cfg.get('init',1.0)),drive_cfg.get('law','gain'),int(drive_cfg.get('hidden',1)))
    rcfg=m['resistance']; resistance=ResistanceSpec(
        rcfg.get('id','resistance'),rcfg.get('default_law','viscous'),
        viscous_gain_init=float(rcfg.get('viscous_gain_init',0.1)),
        coulomb_gain_init=float(rcfg.get('coulomb_gain_init',0.1)),
        viscous_pos_init=float(rcfg.get('viscous_pos_init',0.1)), viscous_neg_init=float(rcfg.get('viscous_neg_init',0.1)),
        coulomb_pos_init=float(rcfg.get('coulomb_pos_init',0.1)), coulomb_neg_init=float(rcfg.get('coulomb_neg_init',0.1)),
        breakaway_delta_init=float(rcfg.get('breakaway_delta_init',0.5)),
        stribeck_velocity_init=float(rcfg.get('stribeck_velocity_init',0.03)),
        smoothing_velocity_init=float(rcfg.get('smoothing_velocity_init',0.001)),
        hidden=int(rcfg.get('hidden',1)),
    )
    cl=m.get('constant_load'); constant_load=None if cl is None else ConstantLoadSpec(cl.get('id','load'),float(cl.get('init',0.0)))
    skeleton=SecondOrderMechanicsIR(CoordinateSpec(m['coordinate']['position'],m['coordinate']['velocity'],qdim),inputs,drive,resistance,float(m['dt']),constant_load)

    slots=[]; ev={}; bindings=[]
    slots.append(MechanismSlot('kinematics',KnowledgeLevel.K0,'kinematic state relation',(vdim,),vdim,exact_family='q_dot_equals_v',knowledge_source=KnowledgeSource.DECLARED_ENGINEERING))
    if drive_cfg.get('law','gain')=='gain':
        slots.append(MechanismSlot('drive',KnowledgeLevel.K1,'input-to-acceleration drive',(_dim(next(x.get('dim') for x in m['inputs'] if x['name']==drive_cfg['input'])),),adim,exact_family='positive_gain',learnable_parameters=('gain',),knowledge_source=KnowledgeSource(drive_cfg.get('knowledge_source','DECLARED_ENGINEERING'))))
        ev['drive']=SlotEvidence(data_qualification=data_q,observation_qualification=obs_q,hard_mechanism_status='COMPLETE',reasons=reasons)
    k=know.get('resistance',{'level':'K1','exact_family':rcfg.get('default_law','viscous')})
    level=KnowledgeLevel(k.get('level','K1')); src=KnowledgeSource(k.get('knowledge_source','DECLARED_ENGINEERING'))
    if level is KnowledgeLevel.K0:
        slot=MechanismSlot('resistance',level,'local mechanical resistance',(vdim,),adim,exact_family=k.get('exact_family',rcfg.get('default_law')),knowledge_source=src)
    elif level is KnowledgeLevel.K1:
        slot=MechanismSlot('resistance',level,'local mechanical resistance',(vdim,),adim,exact_family=k.get('exact_family',rcfg.get('default_law')),learnable_parameters=tuple(k.get('learnable_parameters',['parameters'])),knowledge_source=src)
    elif level is KnowledgeLevel.K2:
        fam=tuple(k['finite_candidates'])
        local_family=k.get('local_neural_family','local_neural') if bool(k.get('allow_k3_escape',False)) else None
        slot=MechanismSlot('resistance',level,'local mechanical resistance',(vdim,),adim,finite_candidates=fam,allow_k3_escape=bool(k.get('allow_k3_escape',False)),local_neural_family=local_family,placement_candidates=tuple(k.get('placement_candidates',())),knowledge_source=src)
        patches=[FamilyPatch(f,(FieldAssignment('resistance.law',f),)) for f in fam]
        if local_family is not None:
            patches.append(FamilyPatch(local_family,(FieldAssignment('resistance.law','local_neural'),)))
        bindings=[SlotBinding('resistance',tuple(patches))]
    else:
        slot=MechanismSlot('resistance',level,'local mechanical resistance',(vdim,),adim,local_neural_family=k.get('local_neural_family','local_neural'),placement_candidates=tuple(k.get('placement_candidates',())),knowledge_source=src)
        bindings=[SlotBinding('resistance',(FamilyPatch(k.get('local_neural_family','local_neural'),(FieldAssignment('resistance.law','local_neural'),)),))]
    if level in (KnowledgeLevel.K0,KnowledgeLevel.K1):
        exact=k.get('exact_family',rcfg.get('default_law','viscous'))
        skeleton=replace(skeleton,resistance=replace(skeleton.resistance,law=exact))
    slots.append(slot)
    finite_status=(finite_status_override or 'PENDING') if level is KnowledgeLevel.K2 else 'NOT_APPLICABLE'
    ev['kinematics']=SlotEvidence(data_qualification=data_q,observation_qualification=obs_q,hard_mechanism_status='COMPLETE',reasons=reasons)
    ev['resistance']=SlotEvidence(data_qualification=data_q,observation_qualification=obs_q,hard_mechanism_status='COMPLETE',finite_status=finite_status,placement_control_status=('DECLARED' if len(slot.placement_candidates)>1 and k.get('placement_control_declared',False) else 'NOT_REQUIRED'),reasons=reasons)
    return plan_neural_architecture(tuple(slots),ev),skeleton,tuple(bindings)


def _peak_problem(spec:Mapping[str,Any], data_q:str, obs_q:str, reasons:tuple[str,...]):
    m=spec['mechanism']; k=spec.get('knowledge',{}).get('response_surface',{})
    axes=tuple(PeakAxisSpec(a['name'],_dim(a.get('dim')),a.get('transform','identity'),float(a.get('scale',1.0)),float(a.get('center_init',0.0)),float(a.get('width_init',1.0)),bool(a.get('learn_center',True))) for a in m['axes'])
    families=m['families']; first=families[0]
    skeleton=SeparablePeakIR(axes,tuple(first['active_axes']),m['output_name'],_dim(m.get('output_dim')),float(m.get('amplitude_init',1.0)))
    level=KnowledgeLevel(k.get('level','K2')); src=KnowledgeSource(k.get('knowledge_source','FINITE_ENGINEERING_HYPOTHESIS'))
    if level is not KnowledgeLevel.K2:
        raise ValueError('goal-driven separable_peak_response currently expects a K2 response_surface slot')
    fam_names=tuple(str(x['name']) for x in families)
    slot=MechanismSlot('response_surface',level,'multivariate local constitutive response',tuple(a.dim for a in axes),_dim(m.get('output_dim')),finite_candidates=fam_names,knowledge_source=src)
    ev=SlotEvidence(data_qualification=data_q,observation_qualification=obs_q,hard_mechanism_status='COMPLETE',finite_status='PENDING',reasons=reasons)
    bindings=(SlotBinding('response_surface',tuple(FamilyPatch(str(f['name']),(FieldAssignment('active_axes',tuple(f['active_axes'])),)) for f in families)),)
    return plan_neural_architecture((slot,),{'response_surface':ev}),skeleton,bindings


def _predict(spec:Mapping[str,Any], model:torch.nn.Module, data:Mapping[str,np.ndarray], idx:np.ndarray) -> torch.Tensor:
    sup=spec['supervision']; kind=spec['template']
    if kind=='second_order_mechanics':
        q=torch.as_tensor(np.asarray(data[sup['position_column']],float)[idx],dtype=torch.float64)
        v=torch.as_tensor(np.asarray(data[sup['velocity_column']],float)[idx],dtype=torch.float64)
        state=torch.stack([q,v],dim=1)
        kwargs={name:torch.as_tensor(np.asarray(data[col],float)[idx],dtype=torch.float64) for name,col in sup.get('input_columns',{}).items()}
        return model.rate(state,**kwargs)[:,1:2]
    kwargs={name:torch.as_tensor(np.asarray(data[col],float)[idx],dtype=torch.float64) for name,col in sup['input_columns'].items()}
    out=model(**kwargs)
    if kind=='separable_peak_response':
        return out[spec['mechanism']['output_name']][:,None]
    names=tuple(spec['supervision']['output_columns'].keys())
    return torch.stack([out[n] for n in names],dim=1)

def _target(spec:Mapping[str,Any], data:Mapping[str,np.ndarray], idx:np.ndarray) -> torch.Tensor:
    sup=spec['supervision']; kind=spec['template']
    if kind=='second_order_mechanics': cols=[sup['target_acceleration_column']]
    elif kind=='separable_peak_response': cols=[sup['target_column']]
    else: cols=[sup['output_columns'][k] for k in sup['output_columns']]
    return torch.stack([torch.as_tensor(np.asarray(data[c],float)[idx],dtype=torch.float64) for c in cols],dim=1)

def _canonical_graph_and_inputs(spec:Mapping[str,Any], model:torch.nn.Module, data:Mapping[str,np.ndarray], idx:np.ndarray):
    """Return CanonicalGraph + input tensors for local-augmentation audits.

    This never changes the forward semantics.  It only exposes already-declared
    local neural nodes so K3 can be penalized/audited as a complement rather than
    an unconstrained competing full model.
    """
    kind=spec['template']; sup=spec['supervision']
    if kind=='second_order_mechanics' and hasattr(model,'graph'):
        q=torch.as_tensor(np.asarray(data[sup['position_column']],float)[idx],dtype=torch.float64)
        v=torch.as_tensor(np.asarray(data[sup['velocity_column']],float)[idx],dtype=torch.float64)
        kw={str(spec['mechanism']['coordinate']['position']):q,str(spec['mechanism']['coordinate']['velocity']):v}
        kw.update({name:torch.as_tensor(np.asarray(data[col],float)[idx],dtype=torch.float64) for name,col in sup.get('input_columns',{}).items()})
        return model.graph,kw
    if kind=='generic_mechanism_graph' and hasattr(model,'forward_env') and hasattr(model,'ir'):
        kw={name:torch.as_tensor(np.asarray(data[col],float)[idx],dtype=torch.float64) for name,col in sup['input_columns'].items()}
        return model,kw
    return None,None


def _augmentation_tensor(spec:Mapping[str,Any], model:torch.nn.Module, data:Mapping[str,np.ndarray], idx:np.ndarray) -> torch.Tensor | None:
    graph,kw=_canonical_graph_and_inputs(spec,model,data,idx)
    if graph is None: return None
    node_ids=[n.id for n in graph.ir.nodes if 'neural' in n.op]
    if not node_ids: return None
    env=graph.forward_env(**kw)
    vals=[env[n].reshape(-1) for n in node_ids]
    return torch.cat(vals) if vals else None


def _physical_parameter_snapshot(model:torch.nn.Module) -> dict[str,float]:
    """Snapshot trainable non-neural parameters for K3 credit-assignment diagnostics."""
    out={}
    for name,p in model.named_parameters():
        if not p.requires_grad or 'local_modules' in name: continue
        a=p.detach().cpu().reshape(-1)
        for i,v in enumerate(a.tolist()): out[f'{name}[{i}]']=float(v)
    return out


def _fit_one_detailed(spec:Mapping[str,Any], base_model:torch.nn.Module, data:Mapping[str,np.ndarray], train_idx:np.ndarray, val_idx:np.ndarray, seed:int, *, use_val_in_fit:bool=False, k3_policy:K3QualificationPolicy|None=None):
    torch.manual_seed(int(seed)); model=copy.deepcopy(base_model).double()
    cfg=spec.get('training',{}); jitter=float(cfg.get('init_jitter',0.02))
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad and jitter: p.add_(jitter*torch.randn_like(p))
    params=[p for p in model.parameters() if p.requires_grad]
    fit_idx=np.unique(np.r_[train_idx,val_idx]) if use_val_in_fit else train_idx
    if params:
        opt=torch.optim.Adam(params,lr=float(cfg.get('lr',0.05)))
        y=_target(spec,data,fit_idx)
        for _ in range(int(cfg.get('epochs',350))):
            opt.zero_grad(); pred=_predict(spec,model,data,fit_idx); loss=torch.mean((pred-y)**2); loss.backward(); opt.step()

        # APHYNITY-inspired but not APHYNITY-reimplementation: after a feasible
        # predictive K3 is obtained, minimize the *declared local neural complement*
        # while keeping fit close to that feasible solution.  This attacks hybrid
        # credit assignment without changing the frozen K2->K3 promotion gate.
        if k3_policy is not None and k3_policy.minimal_augmentation_enabled:
            aug=_augmentation_tensor(spec,model,data,fit_idx)
            if aug is not None and aug.numel():
                with torch.no_grad():
                    pred0=_predict(spec,model,data,fit_idx); base_rmse=torch.sqrt(torch.mean((pred0-y)**2))
                    scale=torch.std(y).clamp_min(1e-12)
                    fit_tol=(base_rmse/scale)*(1.0+float(k3_policy.fit_slack))
                opt2=torch.optim.Adam(params,lr=float(cfg.get('lr',0.05))*0.25)
                for _ in range(int(k3_policy.minimal_augmentation_epochs)):
                    opt2.zero_grad(); pred=_predict(spec,model,data,fit_idx); fit_nrmse=torch.sqrt(torch.mean((pred-y)**2))/scale
                    aug2=_augmentation_tensor(spec,model,data,fit_idx)
                    aug_norm=torch.sqrt(torch.mean(aug2**2))/scale
                    violation=torch.relu(fit_nrmse-fit_tol)
                    loss=aug_norm**2 + float(k3_policy.constraint_weight)*violation**2
                    loss.backward(); opt2.step()
    with torch.no_grad():
        pred=_predict(spec,model,data,val_idx); yv=_target(spec,data,val_idx); rmse=float(torch.sqrt(torch.mean((pred-yv)**2)))
        scale=max(float(torch.std(yv)),1e-12)
        aug=_augmentation_tensor(spec,model,data,val_idx)
        aug_nrmse=None if aug is None else float(torch.sqrt(torch.mean(aug**2))/scale)
    diag={'augmentation_nrmse':aug_nrmse,'physical_parameters':_physical_parameter_snapshot(model)}
    return model,rmse,diag

def _fit_one(spec:Mapping[str,Any], base_model:torch.nn.Module, data:Mapping[str,np.ndarray], train_idx:np.ndarray, val_idx:np.ndarray, seed:int, *, use_val_in_fit:bool=False, k3_policy:K3QualificationPolicy|None=None):
    model,rmse,_=_fit_one_detailed(spec,base_model,data,train_idx,val_idx,seed,use_val_in_fit=use_val_in_fit,k3_policy=k3_policy)
    return model,rmse


def _credit_assignment_summary(diags:list[dict[str,Any]], warn_threshold:float) -> dict[str,Any]:
    keys=sorted({k for d in diags for k in d.get('physical_parameters',{})})
    rows={}; max_cv=0.0
    for k in keys:
        vals=np.asarray([d.get('physical_parameters',{}).get(k,np.nan) for d in diags],float)
        vals=vals[np.isfinite(vals)]
        if not len(vals): continue
        mean=float(np.mean(vals)); std=float(np.std(vals)); cv=float(std/max(abs(mean),1e-12)); max_cv=max(max_cv,cv)
        rows[k]={'mean':mean,'std':std,'cv':cv,'values':vals.tolist()}
    aug=[d.get('augmentation_nrmse') for d in diags if d.get('augmentation_nrmse') is not None]
    return {
        'physical_parameter_restart_stability':rows,
        'max_physical_parameter_cv':max_cv,
        'credit_assignment_warning': bool(max_cv>warn_threshold),
        'warning_threshold':float(warn_threshold),
        'augmentation_nrmse_median':None if not aug else float(np.median(np.asarray(aug,float))),
        'semantics':'diagnostic only: parameter restart stability and complement magnitude do not prove physical identifiability',
    }


def _network_blueprints(space) -> list[dict[str,Any]]:
    out=[]
    for c in space.candidates:
        m=compile_executable_ir(c.ir)
        total=sum(int(p.numel()) for p in m.parameters())
        train=sum(int(p.numel()) for p in m.parameters() if p.requires_grad)
        fam=dict(c.family_choices)
        is_neural=any(('neural' in str(v).lower()) or ('local_nn' in str(v).lower()) for v in fam.values())
        if isinstance(c.ir, CanonicalIR):
            is_neural=is_neural or any('neural' in n.op for n in c.ir.nodes)
        out.append({'candidate_id':c.id,'family_choices':fam,'module_class':type(m).__name__,'total_parameters':total,'trainable_parameters':train,'contains_local_neural_family':is_neural})
    return out


def _blueprint_markdown(spec:Mapping[str,Any], plan:NeuralArchitecturePlan, result:Mapping[str,Any], diagnostics:Mapping[str,Any]) -> str:
    lines=['# M2N Goal-Driven Design Report','',f"- Goal: `{spec.get('goal',{}).get('name','unnamed')}`",f"- Template: `{spec['template']}`",f"- Planning status: **{plan.planning_status}**",f"- Design status: **{result['status']}**",'','## Knowledge-to-architecture decisions','']
    lines += ['| Slot | K-level | Action | Family |','|---|---|---|---|']
    for p in plan.slot_plans:
        lines.append(f"| {p.slot_id} | {p.level} | {p.action} | {', '.join(p.architecture_family) or '-'} |")
    lines += ['','## Basic data qualification','', '```json', json.dumps(diagnostics,indent=2,ensure_ascii=False),'```','']
    if result.get('network_blueprints'):
        lines += ['## Executable network blueprints','', '| Candidate | Module | Trainable params | Local neural |','|---|---|---:|---|']
        for b in result['network_blueprints']:
            lines.append(f"| `{b['candidate_id']}` | `{b['module_class']}` | {b['trainable_parameters']} | {b['contains_local_neural_family']} |")
        lines.append('')
    if result.get('candidate_validation_risks'):
        lines += ['## Candidate validation risk','', '| Candidate | RMSE restarts |','|---|---|']
        for k,v in result['candidate_validation_risks'].items(): lines.append(f"| `{k}` | {', '.join(f'{x:.6g}' for x in v)} |")
    lines += ['','## Final architecture decision','']
    if result['status'] in ('SELECTED','FIXED'):
        lines.append(f"Selected/fixed candidate: `{result.get('selected_candidate_id')}`")
    else:
        lines.append('No unique architecture was committed. Retain the admissible finite candidate set or improve the observation/data contract.')
    lines += ['','> This report is an architecture-design artifact. Low validation error does not by itself establish physical truth or parameter semantics.']
    return '\n'.join(lines)+'\n'



def _target_output_names(spec:Mapping[str,Any]) -> tuple[str,...]:
    if spec['template']=='second_order_mechanics': return ('acceleration',)
    if spec['template']=='separable_peak_response': return (str(spec['mechanism']['output_name']),)
    return tuple(map(str,spec['supervision']['output_columns'].keys()))

def _target_columns(spec:Mapping[str,Any]) -> tuple[str,...]:
    sup=spec['supervision']
    if spec['template']=='second_order_mechanics': return (str(sup['target_acceleration_column']),)
    if spec['template']=='separable_peak_response': return (str(sup['target_column']),)
    return tuple(str(sup['output_columns'][k]) for k in sup['output_columns'])

def _support_input_columns(spec:Mapping[str,Any]) -> tuple[str,...]:
    sup=spec['supervision']
    if spec['template']=='second_order_mechanics':
        return tuple(dict.fromkeys([str(sup['position_column']),str(sup['velocity_column']),*map(str,sup.get('input_columns',{}).values())]))
    return tuple(dict.fromkeys(map(str,sup['input_columns'].values())))

def _problem(spec:Mapping[str,Any], data_q:str, obs_q:str, reasons:tuple[str,...], *, finite_status_override:Mapping[str,str]|None=None):
    finite_status_override=dict(finite_status_override or {})
    if spec['template']=='second_order_mechanics':
        return _mechanics_problem(spec,data_q,obs_q,reasons,finite_status_override=finite_status_override.get('resistance'))
    if spec['template']=='separable_peak_response':
        return _peak_problem(spec,data_q,obs_q,reasons)
    return build_generic_problem(spec,data_qualification=data_q,observation_qualification=obs_q,reasons=reasons,finite_status_by_slot=finite_status_override)

def _slot_families(plan:NeuralArchitecturePlan) -> dict[str,tuple[str,...]]:
    return {p.slot_id:tuple(p.architecture_family) for p in plan.slot_plans if p.action=='RESTRICTED_K2_SEARCH' and len(p.architecture_family)>1}

@dataclass(frozen=True)
class GoalDrivenDesignSummary:
    status:str
    planning_status:str
    selected_candidate_id:str|None
    output_dir:str
    artifact_paths:dict[str,str]
    diagnostics:dict[str,Any]
    def to_dict(self): return asdict(self)


def run_goal_driven_design(spec_path:str|Path, data_path:str|Path, output_dir:str|Path) -> GoalDrivenDesignSummary:
    spec_path=Path(spec_path); data_path=Path(data_path); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    raw_spec=json.loads(spec_path.read_text(encoding='utf-8'))
    if str(raw_spec.get('schema_version')) not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f'device card schema_version must be one of {SUPPORTED_SCHEMA_VERSIONS}')
    if raw_spec.get('template') not in SUPPORTED_TEMPLATES:
        raise ValueError(f"unsupported template {raw_spec.get('template')!r}; supported={sorted(SUPPORTED_TEMPLATES)}")
    data=_read_csv(data_path); spec,derived_notes=_prepare_derived_supervision(copy.deepcopy(raw_spec),data); tri,vai=_split_indices(data,spec.get('data',{}))
    data_q,obs_q,reasons,diagnostics=_basic_data_qualification(spec,data,tri,vai)
    if derived_notes: diagnostics['derived_supervision']=list(derived_notes)
    diagnostics.update({'data_sha256':_sha256(data_path),'spec_sha256':_sha256(spec_path),'data_qualification':data_q,'observation_qualification':obs_q,'qualification_reasons':list(reasons)})
    plan,skeleton,bindings=_problem(spec,data_q,obs_q,reasons)
    bias_plan=build_bias_route_plan(spec)
    representation_plans=build_representation_plan(spec)
    exact_contract=audit_exact_physics(skeleton)

    (out/'device_card.snapshot.json').write_text(json.dumps(raw_spec,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'architecture_plan.json').write_text(json.dumps(plan.to_dict(),indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    (out/'qualification.json').write_text(json.dumps(diagnostics,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'bias_route_plan.json').write_text(json.dumps(bias_plan.to_dict(),indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'representation_plan.json').write_text(json.dumps([x.to_dict() for x in representation_plans],indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'exact_physics_contract.json').write_text(json.dumps(exact_contract.to_dict(),indent=2,ensure_ascii=False),encoding='utf-8')
    artifacts={'architecture_plan':'architecture_plan.json','qualification':'qualification.json','bias_route_plan':'bias_route_plan.json','representation_plan':'representation_plan.json','exact_physics_contract':'exact_physics_contract.json','spec_snapshot':'device_card.snapshot.json'}

    # P3: optional backends are called only after qualification and never mutate
    # the admitted architecture family automatically.
    backend_actions=run_backend_requests(spec,data,tri,out/'backend_actions',qualification_status=data_q)
    if backend_actions:
        ba=[x.to_dict() for x in backend_actions]
        (out/'backend_actions.json').write_text(json.dumps(ba,indent=2,ensure_ascii=False),encoding='utf-8')
        artifacts['backend_actions']='backend_actions.json'

    if plan.planning_status!='READY':
        result={'status':plan.planning_status,'planning_status':plan.planning_status,'selected_candidate_id':None,'candidate_validation_risks':{},'reasons':list(plan.reasons),'backend_actions':[x.to_dict() for x in backend_actions],'qualification_evidence_scope':str(spec.get('qualification',{}).get('evidence_scope','PRODUCTION'))}
        (out/'design_result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
        (out/'DESIGN_REPORT.md').write_text(_blueprint_markdown(spec,plan,result,diagnostics),encoding='utf-8')
        artifacts.update({'result':'design_result.json','report':'DESIGN_REPORT.md'})
        return GoalDrivenDesignSummary(result['status'],plan.planning_status,None,str(out),artifacts,diagnostics)

    # P3: if the exact finite cross-product becomes too large, route to a model-set
    # inference contract rather than silently truncating candidates.
    exact_limit=int(spec.get('backends',{}).get('exact_enumeration_limit',32))
    route=assess_model_set_routing(_slot_families(plan),out,exact_enumeration_limit=exact_limit)
    if route is not None:
        (out/'model_set_routing.json').write_text(json.dumps(route.to_dict(),indent=2,ensure_ascii=False),encoding='utf-8')
        artifacts['model_set_routing']='model_set_routing.json'
        if route.status=='MODEL_SET_BACKEND_RECOMMENDED' and not bool(spec.get('backends',{}).get('force_exact_enumeration',False)):
            result={'status':'MODEL_SET_BACKEND_RECOMMENDED','planning_status':'READY','selected_candidate_id':None,'candidate_validation_risks':{},'reasons':[route.reason],'backend_actions':[x.to_dict() for x in backend_actions],'qualification_evidence_scope':str(spec.get('qualification',{}).get('evidence_scope','PRODUCTION'))}
            (out/'design_result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
            (out/'DESIGN_REPORT.md').write_text(_blueprint_markdown(spec,plan,result,diagnostics),encoding='utf-8')
            artifacts.update({'result':'design_result.json','report':'DESIGN_REPORT.md'})
            return GoalDrivenDesignSummary(result['status'],'READY',None,str(out),artifacts,diagnostics)

    restarts=tuple(int(x) for x in spec.get('training',{}).get('restarts',[101,102,103]))
    if len(restarts)<2: raise ValueError('training.restarts needs at least two seeds for architecture selection')
    k3_policy=K3QualificationPolicy.from_mapping(spec.get('k3_policy'))
    trained_by_candidate={}; training_diag_by_candidate={}
    def evaluator(candidate,model):
        vals=[]; models=[]; diags=[]
        is_local_neural=any('neural' in str(v).lower() for _,v in candidate.family_choices)
        if isinstance(candidate.ir,CanonicalIR): is_local_neural=is_local_neural or any('neural' in n.op for n in candidate.ir.nodes)
        kp=k3_policy if is_local_neural else None
        for seed in restarts:
            mm,r,diag=_fit_one_detailed(spec,model,data,tri,vai,seed,k3_policy=kp); vals.append(r); models.append((r,mm)); diags.append(diag)
        trained_by_candidate[candidate.id]=models; training_diag_by_candidate[candidate.id]=diags
        return vals
    policy_cfg=spec.get('selection',{})
    dr=run_design_session(plan,skeleton,bindings,evaluator,selection_policy=ArchitectureSelectionPolicy(float(policy_cfg.get('practical_ratio',0.8)),float(policy_cfg.get('restart_stability',0.75))),max_candidates=int(spec.get('max_candidates',64)))
    candidate_exact_contracts={c.id:audit_exact_physics(c.ir).to_dict() for c in dr.candidate_space.candidates}
    d=dr.to_dict(); d['network_blueprints']=_network_blueprints(dr.candidate_space); d['backend_actions']=[x.to_dict() for x in backend_actions]
    d['qualification_evidence_scope']=str(spec.get('qualification',{}).get('evidence_scope','PRODUCTION'))

    # P4: a finite family must first fail an absolute normalized fit gate. Only
    # then may a previously declared *local* K3 escape be evaluated.
    k3_decision=None; deploy_space=dr.candidate_space; deploy_ids=None; artifact_stage='finite'
    if dr.candidate_validation_risks:
        scales=[float(x) for x in diagnostics.get('target_train_std',{}).values()]
        target_scale=max(float(np.mean(scales)) if scales else 1.0,1e-12)
        risk_map={k:v for k,v in dr.candidate_validation_risks}
        k3_decision=finite_gate(k3_policy,risk_map,target_scale)
        d['k3_gate']=k3_decision.to_dict()
        if k3_decision.status=='FINITE_REJECTED_TRY_K3':
            eligible=[]
            if spec['template']=='second_order_mechanics':
                k=spec.get('knowledge',{}).get('resistance',{})
                if k.get('level')=='K2' and k.get('allow_k3_escape',False): eligible=['resistance']
            elif spec['template']=='generic_mechanism_graph':
                eligible=[str(x['id']) for x in spec.get('knowledge',{}).get('slots',[]) if x.get('level')=='K2' and x.get('allow_k3_escape',False)]
            if len(eligible)==1:
                rep_plan=k3_representation_for_slot(spec,eligible[0])
                d['k3_representation']=rep_plan.to_dict()
                if rep_plan.requested_kind!='LOCAL_FUNCTION':
                    d['status']='AMBIGUOUS'; d['selected_candidate_id']=None; deploy_ids=[]
                    d['k3_gate']={'status':'SPECIALIZED_REPRESENTATION_REQUIRED','reason':f"K3 slot {eligible[0]!r} is {rep_plan.requested_kind}; native local MLP neuralization is intentionally blocked. Recommended pattern: {rep_plan.eligibility.architecture_pattern}. Eligibility: {rep_plan.eligibility.status}."}
                    k3_plan=None
                else:
                    k3_plan,k3_skel,k3_bind=_problem(spec,data_q,obs_q,reasons,finite_status_override={eligible[0]:'REJECTED'})
                if k3_plan is not None and k3_plan.planning_status=='READY':
                    k3_space=lower_architecture_plan(k3_plan,k3_skel,k3_bind,max_candidates=int(spec.get('max_candidates',64)))
                    if len(k3_space.candidates)!=1:
                        raise ValueError('qualified K3 frontend currently expects exactly one declared local K3 placement')
                    kc=k3_space.candidates[0]; candidate_exact_contracts[kc.id]=audit_exact_physics(kc.ir).to_dict(); km=compile_executable_ir(kc.ir); kvals=evaluator(kc,km)
                    k3_decision=compare_k3(k3_policy,float(d['k3_gate']['finite_best_nrmse']),kvals,target_scale)
                    d['k3_gate']=k3_decision.to_dict(); d['k3_candidate_validation_risks']={kc.id:list(map(float,kvals))}; d['k3_network_blueprint']=_network_blueprints(k3_space)[0]
                    d['k3_training_policy']={'minimal_augmentation_enabled':k3_policy.minimal_augmentation_enabled,'method':'MINIMAL_COMPLEMENT_CONSTRAINED_REFINEMENT' if k3_policy.minimal_augmentation_enabled else 'STANDARD_PREDICTION_FIT','note':'inspired by minimal-augmentation cooperation; this is not a reimplementation of APHYNITY'}
                    d['k3_credit_assignment_audit']=_credit_assignment_summary(training_diag_by_candidate.get(kc.id,[]),k3_policy.credit_assignment_cv_warn)
                    if k3_decision.status=='K3_SUPPORTED':
                        d['status']='K3_SELECTED'; d['selected_candidate_id']=kc.id; d['selected_family_choices']=dict(kc.family_choices)
                        deploy_space=k3_space; deploy_ids=[kc.id]; artifact_stage='k3'
                    else:
                        d['status']='AMBIGUOUS'; d['selected_candidate_id']=None; deploy_ids=[]
                elif k3_plan is not None:
                    d['status']=k3_plan.planning_status; d['selected_candidate_id']=None; d['k3_gate']={'status':'BLOCKED','reason':'; '.join(k3_plan.reasons)}; deploy_ids=[]
            else:
                d['status']='AMBIGUOUS'; d['selected_candidate_id']=None
                d['k3_gate']={'status':'BLOCKED','reason':'qualified K3 requires exactly one declared local escape slot/placement in the goal-driven frontend'}; deploy_ids=[]

    (out/'candidate_exact_physics_contracts.json').write_text(json.dumps(candidate_exact_contracts,indent=2,ensure_ascii=False),encoding='utf-8')
    artifacts['candidate_exact_physics_contracts']='candidate_exact_physics_contracts.json'
    (out/'design_result.json').write_text(json.dumps(d,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    (out/'candidate_space.json').write_text(json.dumps(dr.candidate_space.to_dict(),indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'network_blueprints.json').write_text(json.dumps(d['network_blueprints'],indent=2,ensure_ascii=False),encoding='utf-8')
    artifacts.update({'result':'design_result.json','candidate_space':'candidate_space.json','network_blueprints':'network_blueprints.json'})

    # Final deployable artifacts. Ordinary finite AMBIGUOUS retains the finite set;
    # a rejected finite family with unsupported K3 emits no misleading deployable model.
    if deploy_ids is None:
        if d['status'] in ('SELECTED','FIXED'):
            deploy_ids=[d.get('selected_candidate_id')]
        elif d['status']=='AMBIGUOUS' and (k3_decision is None or k3_decision.status in ('OFF','KEEP_FINITE')):
            deploy_ids=[c.id for c in dr.candidate_space.candidates]
        else: deploy_ids=[]
    model_dir=out/'models'; model_dir.mkdir(exist_ok=True)
    for cid in deploy_ids:
        c=next(x for x in deploy_space.candidates if x.id==cid)
        base=compile_executable_ir(c.ir)
        final_model,_=_fit_one(spec,base,data,tri,vai,int(spec.get('training',{}).get('final_seed',9001)),use_val_in_fit=True,k3_policy=(k3_policy if artifact_stage=='k3' else None))
        payload={'toolkit_format':'M2N_GOAL_MODEL_V2','candidate_id':cid,'family_choices':dict(c.family_choices),'state_dict':final_model.state_dict(),'device_card':raw_spec,'artifact_stage':artifact_stage}
        torch.save(payload,model_dir/f'{_safe_name(cid)}.pt')
    if deploy_ids: artifacts['models']='models/'

    # P5: freeze normal support and evidence semantics alongside every design.
    support_cols=_support_input_columns(spec)
    ref=fit_support_reference(data,support_cols,np.unique(np.r_[tri,vai]),quantile=float(spec.get('qualification',{}).get('support_quantile',0.99)))
    target_names=_target_output_names(spec); target_cols=_target_columns(spec)
    target_scales={n:max(float(np.std(np.asarray(data[c],float)[tri])),1e-12) for n,c in zip(target_names,target_cols)}
    save_support_reference(ref,support_cols,target_scales,out/'normal_support.json'); artifacts['normal_support']='normal_support.json'
    evidence_contract={'contract':'M2N-MODEL-EVIDENCE/2.2','output_names':list(target_names),'support_input_columns':list(support_cols),'model_count':len(deploy_ids),'qualification_evidence_scope':str(spec.get('qualification',{}).get('evidence_scope','PRODUCTION')),'semantics':{'mechanism_residual':'normalized residual of an admissible frozen M2N model family','model_disagreement':'prediction spread across a retained finite architecture set','support_ood':'normal-commissioning reference qualification, not a fault probability','mechanism_evidence_in_support':'1 only when the sample remains inside the frozen normal-support domain; out-of-support residual/disagreement are retained for audit but are not normal-domain mechanism evidence'},'prohibited_claims':['fault probability','fault class','RUL','physical parameter drift unless separately qualified']}
    (out/'evidence_contract.json').write_text(json.dumps(evidence_contract,indent=2,ensure_ascii=False),encoding='utf-8'); artifacts['evidence_contract']='evidence_contract.json'

    (out/'DESIGN_REPORT.md').write_text(_blueprint_markdown(spec,plan,d,diagnostics),encoding='utf-8'); artifacts['report']='DESIGN_REPORT.md'
    (out/'reproduce.txt').write_text(f'm2n design --spec "{spec_path}" --data "{data_path}" --out "{out}"\n',encoding='utf-8'); artifacts['reproduce']='reproduce.txt'
    return GoalDrivenDesignSummary(str(d['status']),str(d.get('planning_status',dr.planning_status)),d.get('selected_candidate_id'),str(out),artifacts,diagnostics)

def _safe_name(s:str)->str:
    return ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in s)[:180] or 'model'


def _candidate_from_payload(payload:Mapping[str,Any]):
    spec=payload['device_card']; dummy_q='READY'; dummy_o='SUPPORTED'; reasons=()
    stage=str(payload.get('artifact_stage','finite'))
    finite_override={}
    if stage=='k3':
        if spec['template']=='second_order_mechanics': finite_override={'resistance':'REJECTED'}
        elif spec['template']=='generic_mechanism_graph':
            eligible=[str(x['id']) for x in spec.get('knowledge',{}).get('slots',[]) if x.get('level')=='K2' and x.get('allow_k3_escape',False)]
            if len(eligible)==1: finite_override={eligible[0]:'REJECTED'}
    plan,skeleton,bindings=_problem(spec,dummy_q,dummy_o,reasons,finite_status_override=finite_override)
    space=lower_architecture_plan(plan,skeleton,bindings,max_candidates=int(spec.get('max_candidates',64)))
    cid=payload['candidate_id']; c=next((x for x in space.candidates if x.id==cid),None)
    if c is None: raise ValueError(f'artifact candidate {cid!r} is not reproducible from embedded device card')
    model=compile_executable_ir(c.ir).double(); model.load_state_dict(payload['state_dict']); model.eval(); return spec,model

def _inference_required_columns(spec:Mapping[str,Any]) -> tuple[str,...]:
    sup=spec['supervision']
    if spec['template']=='second_order_mechanics':
        return tuple(dict.fromkeys([sup['position_column'],sup['velocity_column'],*sup.get('input_columns',{}).values()]))
    return tuple(dict.fromkeys(sup['input_columns'].values()))


def predict_model_artifact(artifact_path:str|Path, data_path:str|Path, output_path:str|Path) -> Path:
    payload=torch.load(Path(artifact_path),map_location='cpu',weights_only=True)
    if payload.get('toolkit_format') not in ('M2N_GOAL_MODEL_V1','M2N_GOAL_MODEL_V2'):
        raise ValueError('unsupported model artifact format')
    spec,model=_candidate_from_payload(payload); data=_read_csv(Path(data_path)); n=len(next(iter(data.values()))); idx=np.arange(n)
    missing=[c for c in _inference_required_columns(spec) if c not in data]
    if missing: raise ValueError(f'missing inference columns: {missing}')
    with torch.no_grad(): pred=np.asarray(_predict(spec,model,data,idx).detach().cpu().numpy(),float)
    if pred.ndim==1: pred=pred[:,None]
    names=_target_output_names(spec)
    output_path=Path(output_path); output_path.parent.mkdir(parents=True,exist_ok=True)
    t,context,time_col=_evidence_coordinates(spec,data)
    headers=[]; arrays=[]
    if context is not None:
        headers.append('m2n_independent_context'); arrays.append(np.asarray(context))
    if time_col is not None:
        headers.append('time_s'); arrays.append(np.asarray(t,float))
    for j,name in enumerate(names):
        headers.append(f'm2n_prediction__{name}'); arrays.append(pred[:,j])
    mat=np.column_stack(arrays)
    with output_path.open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f); w.writerow(headers); w.writerows(mat.tolist())
    return output_path



def generate_model_evidence(design_dir:str|Path, data_path:str|Path, output_path:str|Path) -> Path:
    """Run a frozen selected/model-set design and emit M2N PHM evidence.

    The function uses only the model artifact(s), the frozen normal-support
    reference, and the observation/target columns declared in the device card.
    It does not consume fault labels or simulator-truth fields.
    """
    root=Path(design_dir); spec=json.loads((root/'device_card.snapshot.json').read_text(encoding='utf-8'))
    data=_read_csv(Path(data_path)); spec2,_=_prepare_derived_supervision(copy.deepcopy(spec),data)
    n=len(next(iter(data.values()))); idx=np.arange(n)
    model_paths=sorted((root/'models').glob('*.pt'))
    if not model_paths: raise ValueError('design contains no deployable model artifacts; evidence cannot be generated')
    preds={}
    for mp in model_paths:
        payload=torch.load(mp,map_location='cpu',weights_only=True); _,m=_candidate_from_payload(payload)
        with torch.no_grad(): preds[str(payload['candidate_id'])]=np.asarray(_predict(spec2,m,data,idx).detach().cpu().numpy(),float)
    target=np.asarray(_target(spec2,data,idx).detach().cpu().numpy(),float)
    ref,cols,scales=load_support_reference(root/'normal_support.json')
    X=np.column_stack([np.asarray(data[c],float) for c in cols])
    dist,ood=ref.score(X)
    t,context,_=_evidence_coordinates(spec2,data)
    return write_evidence_csv(
        output_path,time=t,output_names=_target_output_names(spec2),target=target,predictions=preds,
        support_distance=dist,support_ood=ood,target_scales=scales,independent_context=context,
    )
