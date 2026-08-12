from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
import json
from pathlib import Path

CERTIFIED_STATUSES={"COMPLETE"}

@dataclass
class MechanismScopeRecord:
    scope_id: str
    target_states: List[str]
    scope_kind: str = "dynamic"
    observed_drivers: List[str] = field(default_factory=list)
    latent_boundaries: List[str] = field(default_factory=list)
    graph: Dict[str,List[str]] = field(default_factory=dict)
    observation_status: str = "UNASSESSED"
    completion_status: str = "UNCERTIFIED"
    parameter_policy: Dict[str,Any] = field(default_factory=dict)
    phm_coverage: Dict[str,Any] = field(default_factory=dict)
    metadata: Dict[str,Any] = field(default_factory=dict)

    @property
    def certified(self) -> bool: return self.completion_status in CERTIFIED_STATUSES

    @property
    def variables(self) -> set[str]:
        out=set(self.target_states)|set(self.observed_drivers)|set(self.latent_boundaries)
        for t,srcs in self.graph.items(): out.add(t); out.update(srcs)
        return out

@dataclass
class ScopeInterface:
    source_scope: str
    target_scope: str
    variables: List[str]
    kind: str = "measured_boundary"
    metadata: Dict[str,Any] = field(default_factory=dict)

@dataclass
class UncoveredRegion:
    region_id: str
    variables: List[str]
    route: str
    reason: str
    metadata: Dict[str,Any] = field(default_factory=dict)

@dataclass
class MechanismAtlas:
    system_id: str
    scopes: Dict[str,MechanismScopeRecord] = field(default_factory=dict)
    interfaces: List[ScopeInterface] = field(default_factory=list)
    uncovered_regions: List[UncoveredRegion] = field(default_factory=list)
    metadata: Dict[str,Any] = field(default_factory=dict)

    def add_scope(self, scope: MechanismScopeRecord) -> None:
        if scope.scope_id in self.scopes: raise ValueError(f"duplicate scope_id {scope.scope_id!r}")
        self.scopes[scope.scope_id]=scope

    def add_interface(self, interface: ScopeInterface) -> None:
        if interface.source_scope not in self.scopes or interface.target_scope not in self.scopes:
            raise KeyError("scope interface references an unknown scope")
        self.interfaces.append(interface)

    def add_uncovered(self, region: UncoveredRegion) -> None: self.uncovered_regions.append(region)

    def route_variables(self, variables: Sequence[str], *, require_certified: bool=True) -> Dict[str,Any]:
        requested=set(variables); matches=[]
        for sid,s in self.scopes.items():
            overlap=sorted(requested & s.variables)
            if overlap and (s.certified or not require_certified):
                matches.append({"scope_id":sid,"overlap":overlap,"completion_status":s.completion_status,
                                "phm_coverage":s.phm_coverage})
        covered=set(v for m in matches for v in m["overlap"])
        uncovered=sorted(requested-covered)
        hints=[]
        for r in self.uncovered_regions:
            ov=sorted(requested & set(r.variables))
            if ov: hints.append({"region_id":r.region_id,"variables":ov,"route":r.route,"reason":r.reason})
        return {"matches":matches,"uncovered":uncovered,"route_hints":hints}

    def to_dict(self) -> Dict[str,Any]:
        return {"format":"isgn-mechanism-atlas-v1","system_id":self.system_id,
                "scopes":{k:asdict(v) for k,v in self.scopes.items()},
                "interfaces":[asdict(x) for x in self.interfaces],
                "uncovered_regions":[asdict(x) for x in self.uncovered_regions],
                "metadata":self.metadata}

    @classmethod
    def from_dict(cls,d:Mapping[str,Any]) -> "MechanismAtlas":
        a=cls(system_id=str(d["system_id"]),metadata=dict(d.get("metadata",{})))
        for k,v in d.get("scopes",{}).items(): a.scopes[k]=MechanismScopeRecord(**v)
        a.interfaces=[ScopeInterface(**x) for x in d.get("interfaces",[])]
        a.uncovered_regions=[UncoveredRegion(**x) for x in d.get("uncovered_regions",[])]
        return a

def save_atlas(atlas: MechanismAtlas, path: str|Path) -> None:
    Path(path).write_text(json.dumps(atlas.to_dict(),indent=2,ensure_ascii=False),encoding="utf-8")

def load_atlas(path: str|Path) -> MechanismAtlas:
    return MechanismAtlas.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
