from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Any
import numpy as np
import torch

from .contracts import PublicDataset
from .training import train_c1a, rollout_nrmse
from .constitutive_training import train_b2a
from .completion import (equation_residual_nrmse, target_residual_nrmse, calibrate_replicate_coverage,
                         CoverageCalibration, source_identifiability, decide)
from .phm import fit_nominal, mechanism_features, NominalMechanism
from .grammar import HierarchicalISGNGraph
from .backends.base import CanonicalMechanismModel
from .backends.registry import get_backend, backend_status, recommend_backend


@dataclass
class RecoverySummary:
    graph: Dict[str, List[str]]
    operators: Dict[str, Dict[str, str]]
    coefficients: Dict[str, Dict[str, float]]
    val_nrmse: float
    seed: int
    constitutive_search: bool
    backend: str = "native"
    operator_parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    backend_metadata: Dict[str, Any] = field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass
class RecoveryResult:
    model: Any
    summary: RecoverySummary
    schema: Dict[str, Any]
    dt: float


def _recover_native(train: PublicDataset, val: PublicDataset, *, seed: int,
                    device: str, epochs: int, refit_epochs: int,
                    constitutive_search: bool, known_present=None, known_absent=None) -> RecoveryResult:
    if constitutive_search:
        if known_present or known_absent:
            raise ValueError("native constitutive_search trainer in v1.1 does not accept topology priors; use two-stage workflow")
        model, graph, ops, _ = train_b2a(train, val, seed=seed, device=device,
                                         epochs=epochs, refit_epochs=refit_epochs)
    else:
        model, graph, _ = train_c1a(train, val, seed=seed, device=device,
                                    epochs=epochs, refit_epochs=refit_epochs,
                                    known_present=known_present, known_absent=known_absent)
        ops = {t: {s: "Identity" for s in srcs} for t, srcs in graph.items()}
    val_err = rollout_nrmse(model, val, device=device, max_steps=min(200, val.U.shape[1]))
    op_params = model.constitutive_parameters(graph) if hasattr(model,'constitutive_parameters') else {t:{s:None for s in srcs} for t,srcs in graph.items()}
    summary = RecoverySummary(graph=graph, operators=ops,
                              coefficients=model.effective_coefficients(),
                              val_nrmse=float(val_err), seed=int(seed),
                              constitutive_search=bool(constitutive_search),backend='native',
                              operator_parameters=op_params,
                              backend_metadata={'solver':'HierarchicalISGNGraph'})
    return RecoveryResult(model=model, summary=summary, schema=train.schema, dt=float(train.dt))


def recover_graph(train: PublicDataset, val: PublicDataset, *, seed: int = 101,
                  device: str = "cpu", epochs: int = 1000, refit_epochs: int = 300,
                  constitutive_search: bool = False,
                  known_present: Optional[Dict[str, Iterable[str]]] = None,
                  known_absent: Optional[Dict[str, Iterable[str]]] = None,
                  backend: str = "native", backend_options: Optional[Dict[str, Any]] = None) -> RecoveryResult:
    """Recover an operational Mechanism Graph from normal trajectories.

    Backends:
      - ``native``: frozen hierarchical ISGN-v1 solver.
      - ``pysindy``: official optional PySINDy/SINDYc adapter.
      - ``reference-sindy``: small offline STLSQ reference implementation; not PySINDy.
      - ``sklearn-lasso``: scale-normalized independent sparse linear optimizer.
      - ``auto``: prefer official PySINDy when installed, otherwise native.

    Certification and PHM sit above this solver layer and consume the same
    ``RecoveryResult`` regardless of backend.
    """
    if backend == 'auto':
        backend = recommend_backend(constitutive_search=constitutive_search, has_partial_prior=bool(known_present or known_absent))['backend']
    if backend == 'native':
        return _recover_native(train,val,seed=seed,device=device,epochs=epochs,refit_epochs=refit_epochs,
                               constitutive_search=constitutive_search,known_present=known_present,known_absent=known_absent)
    opts=dict(backend_options or {})
    opts.setdefault('constitutive_search',constitutive_search)
    b=get_backend(backend)
    if known_present or known_absent:
        d=b.complete(train,val,known_present=known_present,known_absent=known_absent,**opts)
    else:
        d=b.recover(train,val,**opts)
    summary=RecoverySummary(graph=d.graph,operators=d.operators,coefficients=d.coefficients,
                            val_nrmse=float(d.validation_error),seed=int(seed),
                            constitutive_search=constitutive_search,backend=d.backend,
                            operator_parameters=d.operator_parameters,backend_metadata=d.metadata)
    return RecoveryResult(model=d.model,summary=summary,schema=train.schema,dt=float(train.dt))


def complete_graph(train: PublicDataset, val: PublicDataset, *, known_present: Dict[str, Iterable[str]],
                   known_absent: Optional[Dict[str, Iterable[str]]] = None, **kwargs) -> RecoveryResult:
    return recover_graph(train, val, known_present=known_present,
                         known_absent=known_absent, **kwargs)


def _source_columns_from_graph(ds: PublicDataset, graph: Dict[str, List[str]]):
    state_names=[v["name"] for v in ds.schema["states"]]
    input_names=[v["name"] for v in ds.schema["inputs"]]
    used=[]
    for names in graph.values():
        for n in names:
            if n == "__constant__" or n in [x[1] for x in used]: continue
            if n in state_names: used.append(("state",n,state_names.index(n)))
            elif n in input_names: used.append(("input",n,input_names.index(n)))
    return used


def _prior_conflicts_from_summary(summary: RecoverySummary, declared_present: Dict[str,List[str]], relative_floor:float=.03):
    conflicts=[]
    for target,names in declared_present.items():
        row=summary.coefficients.get(target,{})
        mx=max([abs(float(v)) for v in row.values()]+[1e-12])
        for name in names:
            if abs(float(row.get(name,0.0))) < relative_floor*mx:
                conflicts.append(f'{name}->{target}')
    return conflicts


def _equation_residual_result(result:RecoveryResult,data:PublicDataset,device='cpu') -> float:
    if hasattr(result.model,'rhs_numpy'):
        xm=.5*(data.X[:,:-1].astype(float)+data.X[:,1:].astype(float))
        yp=result.model.rhs_numpy(xm,data.U.astype(float))
        yt=(data.X[:,1:].astype(float)-data.X[:,:-1].astype(float))/data.dt
        e=yt-yp
        rms=np.sqrt(np.mean(e*e,axis=(0,1))); scale=np.maximum(np.std(yt,axis=(0,1)),1e-6)
        return float(np.mean(rms/scale))
    return equation_residual_nrmse(result.model,data,device=device)


def calibrate_completion_coverage(result: RecoveryResult, replicate_groups, *, quantile: float=0.95,
                                  min_floor: float=1e-6, device: str='cpu') -> CoverageCalibration:
    """Calibrate per-target measurement/numerical residual floors from repeat acquisitions.

    The replicate groups must represent repeated measurements of the same physical
    condition within each group. Pairwise residual differences cancel deterministic
    model bias, preventing the candidate model from defining its own Grammar gap as
    the normal coverage floor.
    """
    return calibrate_replicate_coverage(result.model,list(replicate_groups),quantile=quantile,min_floor=min_floor,device=device)


def certify_completion(result: RecoveryResult, data: PublicDataset, *,
                       declared_present: Optional[Dict[str, List[str]]] = None,
                       coverage_floor: Optional[float] = None,
                       coverage_calibration: Optional[CoverageCalibration | Dict[str, Any]] = None,
                       coverage_threshold: float = 8.0,
                       device: str = 'cpu') -> Dict[str, Any]:
    """Backend-agnostic completion certificate.

    Preferred P5+ path: pass a per-target ``coverage_calibration`` produced from
    repeated measurement realizations.  Legacy scalar ``coverage_floor`` remains
    supported for backward compatibility, but can dilute a localized Grammar gap
    when multiple target equations are averaged together.
    """
    if coverage_floor is not None and coverage_calibration is not None:
        raise ValueError("provide either coverage_floor or coverage_calibration, not both")
    graph=result.summary.graph
    prior_conflicts=_prior_conflicts_from_summary(result.summary,declared_present or {})
    cols=_source_columns_from_graph(data, graph)
    ident=source_identifiability(data, [(k,i) for k,_,i in cols]) if cols else {
        'identifiable': False, 'rank':0, 'n_columns':0, 'condition_number':float('inf')}
    residual=_equation_residual_result(result,data,device=device)

    if coverage_calibration is not None:
        cal=coverage_calibration if isinstance(coverage_calibration,CoverageCalibration) else CoverageCalibration.from_dict(coverage_calibration)
        names=[v['name'] for v in data.schema['states']]
        missing=[n for n in names if n not in cal.target_floors or n not in cal.target_scales]
        if missing:
            return {"status":"UNCERTIFIED", "reason":"per_target_coverage_calibration_incomplete",
                    "missing_targets":missing,"prior_conflicts":prior_conflicts,
                    "equation_residual_nrmse":float(residual),"identifiability":ident,
                    "graph":graph,"backend":result.summary.backend,
                    "coverage_calibration":cal.to_dict()}
        by_target=target_residual_nrmse(result.model,data,device=device,scales=cal.target_scales)
        per_target={}
        for n in names:
            floor=max(float(cal.target_floors[n]),1e-12)
            val=float(by_target[n]); per_target[n]={"residual_nrmse":val,"coverage_floor":floor,"coverage_ratio":val/floor}
        coverage_target=max(names,key=lambda n:per_target[n]['coverage_ratio']) if names else None
        ratio=float(per_target[coverage_target]['coverage_ratio']) if coverage_target is not None else float('inf')
        cert=decide(prior_conflicts=prior_conflicts,coverage_ratio=ratio,
                    identifiable=bool(ident['identifiable']),rank=int(ident['rank']),n_columns=int(ident['n_columns']),
                    condition_number=float(ident['condition_number']),coverage_threshold=coverage_threshold,
                    evidence={"equation_residual_nrmse":float(residual),"coverage_method":cal.method,
                              "coverage_target":coverage_target,"per_target":per_target,
                              "calibration_replicate_groups":cal.replicate_groups,
                              "calibration_replicate_trajectories":cal.replicate_trajectories,
                              "calibration_pair_count":cal.pair_count,"calibration_quantile":cal.quantile})
        d=cert.to_dict(); d.update({"graph":graph,"backend":result.summary.backend,"coverage_target":coverage_target,
                                   "per_target_coverage":per_target,"coverage_calibration":cal.to_dict()})
        return d

    if coverage_floor is None or not np.isfinite(coverage_floor) or coverage_floor <= 0:
        return {"status":"UNCERTIFIED", "reason":"coverage_floor_required",
                "prior_conflicts":prior_conflicts, "equation_residual_nrmse":float(residual),
                "identifiability":ident, "graph":graph, "backend":result.summary.backend}
    ratio=float(residual/coverage_floor)
    cert=decide(prior_conflicts=prior_conflicts, coverage_ratio=ratio,
                identifiable=bool(ident['identifiable']), rank=int(ident['rank']),
                n_columns=int(ident['n_columns']), condition_number=float(ident['condition_number']),
                coverage_threshold=coverage_threshold,
                evidence={"equation_residual_nrmse":float(residual),"coverage_floor":float(coverage_floor),
                          "coverage_method":"legacy_scalar"})
    d=cert.to_dict(); d["graph"]=graph; d['backend']=result.summary.backend; return d

def fit_normal_phm(data: PublicDataset, graph: Dict[str, List[str]], *, operators=None, operator_parameters=None) -> NominalMechanism:
    return fit_nominal(data, graph, operators=operators, operator_parameters=operator_parameters)

def extract_mechanism_features(data: PublicDataset, nominal: NominalMechanism) -> np.ndarray:
    return mechanism_features(data, nominal)


def save_checkpoint(result: RecoveryResult, path: str | Path, *, metadata: Optional[Dict[str,Any]]=None) -> None:
    base={"format":"isgn-fusion-v1.1-checkpoint","schema":result.schema,"dt":result.dt,
          "summary":result.summary.to_dict(),"metadata":metadata or {}}
    if isinstance(result.model,HierarchicalISGNGraph):
        base.update({"model_kind":"native","state_dict":result.model.state_dict(),
                     "allow_constant":result.model.allow_constant,
                     "constitutive_search":result.model.constitutive_search})
    elif isinstance(result.model,CanonicalMechanismModel):
        base.update({"model_kind":"canonical","canonical_model":result.model.to_dict()})
    else:
        raise TypeError(f'unsupported checkpoint model type {type(result.model)!r}')
    torch.save(base,Path(path))


def load_checkpoint(path: str | Path, *, device: str="cpu") -> RecoveryResult:
    p=torch.load(Path(path),map_location=device,weights_only=False)
    # Backwards compatibility with Toolkit v1.0 checkpoints.
    kind=p.get('model_kind','native')
    sdict=p['summary']
    # Older summaries do not include fusion fields.
    s=RecoverySummary(**sdict)
    if kind=='canonical':
        model=CanonicalMechanismModel.from_dict(p['canonical_model'])
    else:
        model=HierarchicalISGNGraph.from_schema(p["schema"],p["dt"],seed=sdict.get("seed",101),
                                                allow_constant=p.get("allow_constant",False),
                                                constitutive_search=p.get("constitutive_search",False)).to(device)
        model.load_state_dict(p["state_dict"])
    return RecoveryResult(model=model,summary=s,schema=p['schema'],dt=float(p['dt']))
