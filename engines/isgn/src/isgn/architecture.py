from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from torch import nn

from .atlas import MechanismAtlas


@dataclass(frozen=True)
class TargetArchitectureSpec:
    target: str
    sources: Sequence[str]
    policy: str  # hard_masked_mlp / analytic_fixed
    coefficients: Mapping[str, float] = field(default_factory=dict)
    intercept: float = 0.0
    operators: Mapping[str, str] = field(default_factory=dict)
    operator_parameters: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScopeArchitectureSpec:
    scope_id: str
    scope_kind: str
    completion_status: str
    targets: Sequence[TargetArchitectureSpec]
    parameter_policy: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ArchitectureBlueprint:
    system_id: str
    input_variables: List[str]
    output_variables: List[str]
    scopes: List[ScopeArchitectureSpec]
    excluded_scopes: List[Dict[str, Any]] = field(default_factory=list)
    uncovered_blocks: List[Dict[str, Any]] = field(default_factory=list)
    interfaces: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": "isgn-mechanism-architecture-v1",
            "system_id": self.system_id,
            "input_variables": list(self.input_variables),
            "output_variables": list(self.output_variables),
            "scopes": [asdict(s) for s in self.scopes],
            "excluded_scopes": list(self.excluded_scopes),
            "uncovered_blocks": list(self.uncovered_blocks),
            "interfaces": list(self.interfaces),
            "metadata": dict(self.metadata),
        }


def _analytic_spec(scope, target: str, sources: Sequence[str]) -> Optional[TargetArchitectureSpec]:
    meta = scope.metadata or {}
    # Dynamic/canonical mechanism payload: produced by an already-certified
    # Mechanism Model checkpoint and embedded into the Atlas by the caller.
    mech = meta.get("mechanism")
    if isinstance(mech, Mapping):
        coeff_row=(mech.get("coefficients") or {}).get(target,{})
        ops_row=(mech.get("operators") or {}).get(target,{})
        par_row=(mech.get("operator_parameters") or {}).get(target,{})
        if isinstance(coeff_row,Mapping) and coeff_row and all(str(k) in sources for k in coeff_row):
            supported={"Identity","Saturation","DeadZone","SmoothStribeck"}
            if all(str(ops_row.get(k,"Identity")) in supported for k in coeff_row):
                return TargetArchitectureSpec(target=target,sources=list(sources),policy="analytic_fixed",
                    coefficients={str(k):float(v) for k,v in coeff_row.items()},intercept=0.0,
                    operators={str(k):str(ops_row.get(k,"Identity")) for k in coeff_row},
                    operator_parameters={str(k):par_row.get(k) for k in coeff_row},
                    metadata={"certified_mechanism":True,"mechanism_payload":"canonical"})
    # Algebraic relation payload.
    coeff = meta.get("coefficients")
    if scope.scope_kind != "algebraic" or not isinstance(coeff, Mapping) or not coeff:
        return None
    valid = True
    for term in coeff:
        parts = str(term).split("*")
        if not parts or any(p not in sources for p in parts):
            valid = False; break
    if not valid:
        return None
    return TargetArchitectureSpec(target=target,sources=list(sources),policy="analytic_fixed",
                                  coefficients={str(k):float(v) for k,v in coeff.items()},
                                  intercept=float(meta.get("intercept",0.0)),
                                  metadata={"certified_mechanism":True,"mechanism_payload":"algebraic-polynomial"})

def _execution_plan(scope_specs: Sequence[ScopeArchitectureSpec]) -> tuple[list[str], set[str], Dict[str, str]]:
    """Return a safe target execution order and internally produced algebraic values.

    Only algebraic targets are treated as instantaneous values that may feed a
    downstream scope. Dynamic targets represent derivatives and therefore never
    replace the current-state value of the same name inside one forward call.
    """
    specs: Dict[str, TargetArchitectureSpec] = {}
    kinds: Dict[str, str] = {}
    for scope in scope_specs:
        for target in scope.targets:
            if target.target in specs:
                raise ValueError(f"target {target.target!r} is produced by multiple compiled scopes; explicit arbitration is required")
            specs[target.target] = target
            kinds[target.target] = scope.scope_kind
    algebraic = {name for name, kind in kinds.items() if kind == "algebraic"}
    deps: Dict[str, set[str]] = {name: {str(s) for s in spec.sources if str(s) in algebraic} for name, spec in specs.items()}
    order: list[str] = []
    remaining = {k: set(v) for k, v in deps.items()}
    while remaining:
        ready = sorted(k for k, v in remaining.items() if not v)
        if not ready:
            cyclic = sorted(remaining)
            raise ValueError(f"cyclic algebraic scope dependency cannot be compiled safely: {cyclic}")
        for name in ready:
            order.append(name)
            remaining.pop(name)
            for v in remaining.values():
                v.discard(name)
    return order, algebraic, kinds


def compile_mechanism_atlas(atlas: MechanismAtlas, *, include_uncertified: bool=False,
                            use_analytic_when_available: bool=True,
                            activate_uncovered_residuals: bool=False) -> ArchitectureBlueprint:
    """Compile a MechanismAtlas into a safe neural-architecture blueprint.

    COMPLETE scopes become hard receptive-field modules: a target subnet can only
    read sources present in its certified graph. Certified algebraic scopes with
    explicit polynomial coefficients can become fixed analytic layers. Certified
    algebraic outputs are topologically chained into downstream scopes and are not
    redundantly required as external inputs. Dynamic targets remain derivative
    outputs and never overwrite current-state inputs. Uncertified scopes are
    excluded by default. Uncovered regions are declarations, not hidden residual
    networks, unless explicitly activated by the caller.
    """
    scope_specs: List[ScopeArchitectureSpec] = []
    excluded: List[Dict[str,Any]]=[]
    outputs: set[str]=set()
    for sid, scope in atlas.scopes.items():
        if not scope.certified and not include_uncertified:
            excluded.append({"scope_id":sid,"reason":"scope is not COMPLETE","completion_status":scope.completion_status})
            continue
        targets=[]
        for target in scope.target_states:
            sources=list(dict.fromkeys(scope.graph.get(target,())))
            if not sources:
                # A scope target without an explicit graph is not silently given a
                # dense network; that would erase the mechanism prior.
                continue
            spec=_analytic_spec(scope,target,sources) if use_analytic_when_available and scope.certified else None
            if spec is None:
                spec=TargetArchitectureSpec(target=target,sources=sources,policy="hard_masked_mlp",
                                            metadata={"certified_mechanism":bool(scope.certified)})
            targets.append(spec); outputs.add(target)
        if targets:
            scope_specs.append(ScopeArchitectureSpec(sid,scope.scope_kind,scope.completion_status,targets,
                                                      parameter_policy=dict(scope.parameter_policy or {}),
                                                      metadata={"phm_coverage":dict(scope.phm_coverage or {}),**dict(scope.metadata or {})}))

    execution_order, algebraic_outputs, kinds = _execution_plan(scope_specs)
    inputs: set[str]=set()
    for scope in scope_specs:
        for target in scope.targets:
            for source in map(str, target.sources):
                # A certified algebraic target is an instantaneous internally
                # produced boundary value. Dynamic outputs are derivatives and
                # therefore remain external current-state inputs when referenced.
                if source not in algebraic_outputs:
                    inputs.add(source)

    uncovered=[]
    for r in atlas.uncovered_regions:
        uncovered.append({"region_id":r.region_id,"variables":list(r.variables),"route":r.route,"reason":r.reason,
                          "active_residual_block":bool(activate_uncovered_residuals),"metadata":dict(r.metadata or {})})
        if activate_uncovered_residuals:
            inputs.update(r.variables)
    interfaces=[asdict(i) for i in atlas.interfaces]
    return ArchitectureBlueprint(
        system_id=atlas.system_id,input_variables=sorted(inputs),output_variables=sorted(outputs),scopes=scope_specs,
        excluded_scopes=excluded,uncovered_blocks=uncovered,interfaces=interfaces,
        metadata={"safe_default_excludes_uncertified":not include_uncertified,
                  "uncovered_residuals_active":bool(activate_uncovered_residuals),
                  "execution_order":execution_order,
                  "internal_algebraic_variables":sorted(algebraic_outputs),
                  "target_scope_kinds":kinds,
                  "compiler_semantics":"certified graph defines target receptive fields; certified algebraic outputs may feed downstream scopes; certification is not inferred by the compiler"},
    )


class _MaskedMLPTarget(nn.Module):
    def __init__(self, n_sources: int, hidden_dim: int=24, depth: int=2):
        super().__init__(); n=int(n_sources)
        layers: List[nn.Module]=[]
        if depth <= 0:
            layers=[nn.Linear(n,1)]
        else:
            layers.extend([nn.Linear(n,hidden_dim),nn.SiLU()])
            for _ in range(depth-1): layers.extend([nn.Linear(hidden_dim,hidden_dim),nn.SiLU()])
            layers.append(nn.Linear(hidden_dim,1))
        self.net=nn.Sequential(*layers)
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class _AnalyticMechanismTarget(nn.Module):
    def __init__(self, spec: TargetArchitectureSpec):
        super().__init__(); self.spec=spec; self.variable_index={str(n):i for i,n in enumerate(spec.sources)}
    @staticmethod
    def _transform(z: torch.Tensor, op: str, param: Any) -> torch.Tensor:
        if op=="Identity": return z
        if op=="Saturation": return torch.clamp(z,-float(param),float(param))
        if op=="DeadZone":
            c=float(param); return torch.sign(z)*torch.clamp(torch.abs(z)-c,min=0.0)
        if op=="SmoothStribeck":
            p=dict(param or {}); eps=float(p.get("smoothing_velocity",p.get("eps",0.0))); vs=float(p.get("breakaway_velocity",p.get("vs",0.0)))
            rv=float(p.get("viscous_ratio",0.0)); rs=float(p.get("stribeck_ratio",0.0))
            if eps<=0 or vs<=0: raise ValueError("invalid SmoothStribeck parameters in architecture blueprint")
            th=torch.tanh(z/eps); return rv*z+th*(1.0+rs*torch.exp(-(torch.abs(z)/vs)**2))
        raise ValueError(f"unsupported analytic operator {op!r}")
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        y=torch.full(x.shape[:-1],float(self.spec.intercept),dtype=x.dtype,device=x.device)
        if self.spec.operators:
            for source,c in self.spec.coefficients.items():
                z=x[...,self.variable_index[source]]; op=self.spec.operators.get(source,"Identity"); par=self.spec.operator_parameters.get(source)
                y=y+float(c)*self._transform(z,op,par)
            return y
        for term,c in self.spec.coefficients.items():
            z=torch.ones(x.shape[:-1],dtype=x.dtype,device=x.device)
            for name in term.split("*"): z=z*x[...,self.variable_index[name]]
            y=y+float(c)*z
        return y


class CompiledAtlasNetwork(nn.Module):
    """Executable target-wise network with Atlas-defined receptive fields.

    Algebraic targets can be evaluated internally and fed to later scopes. Dynamic
    targets are derivative outputs and do not overwrite same-name current states.
    """
    def __init__(self, blueprint: ArchitectureBlueprint, *, hidden_dim: int=24, depth: int=2):
        super().__init__(); self.blueprint=blueprint
        self.input_index={n:i for i,n in enumerate(blueprint.input_variables)}
        self.output_index={n:i for i,n in enumerate(blueprint.output_variables)}
        specs: Dict[str,TargetArchitectureSpec]={}; kinds: Dict[str,str]={}; modules={}; policies={}
        for scope in blueprint.scopes:
            for t in scope.targets:
                if t.target in modules:
                    raise ValueError(f"target {t.target!r} is produced by multiple compiled scopes; explicit arbitration is required")
                if t.policy=="analytic_fixed":
                    m=_AnalyticMechanismTarget(t)
                elif t.policy=="hard_masked_mlp":
                    m=_MaskedMLPTarget(len(t.sources),hidden_dim=hidden_dim,depth=depth)
                else: raise KeyError(f"unknown target policy {t.policy!r}")
                modules[t.target]=m; policies[t.target]=t.policy; specs[t.target]=t; kinds[t.target]=scope.scope_kind
        self.targets=nn.ModuleDict(modules); self.policies=policies; self.specs=specs; self.kinds=kinds
        order=list(blueprint.metadata.get("execution_order",[]))
        self.execution_order=order if order else list(blueprint.output_variables)
        self.algebraic_outputs=set(blueprint.metadata.get("internal_algebraic_variables",[]))
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != len(self.blueprint.input_variables):
            raise ValueError(f"expected last dimension {len(self.blueprint.input_variables)}, got {x.shape[-1]}")
        values={name:x[...,idx] for name,idx in self.input_index.items()}
        outputs: Dict[str,torch.Tensor]={}
        for target in self.execution_order:
            spec=self.specs[target]
            missing=[s for s in spec.sources if s not in values]
            if missing:
                raise RuntimeError(f"target {target!r} is missing compiled source values {missing!r}")
            local=torch.stack([values[s] for s in spec.sources],dim=-1)
            y=self.targets[target](local); outputs[target]=y
            if self.kinds[target]=="algebraic":
                values[target]=y
        return torch.stack([outputs[target] for target in self.blueprint.output_variables],dim=-1)


def build_compiled_network(blueprint: ArchitectureBlueprint, *, hidden_dim: int=24, depth: int=2) -> CompiledAtlasNetwork:
    return CompiledAtlasNetwork(blueprint,hidden_dim=hidden_dim,depth=depth)
