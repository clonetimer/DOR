from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .atlas import MechanismAtlas, MechanismScopeRecord, ScopeInterface, UncoveredRegion

_READY = {"READY", "PASS", "QUALIFIED", "CONDITIONALLY_QUALIFIED", "USABLE"}
_CONDITIONAL = {"CONDITIONAL", "WARN", "MARGINAL"}
_BLOCKED = {"BLOCKED", "FAIL", "OBSERVATION_INSUFFICIENT", "UNUSABLE"}


def _norm_status(x: str) -> str:
    s = str(x or "UNKNOWN").upper()
    if s in _READY:
        return "READY"
    if s in _CONDITIONAL:
        return "CONDITIONAL"
    if s in _BLOCKED:
        return "BLOCKED"
    return "UNKNOWN"


@dataclass(frozen=True)
class VariableEvidence:
    name: str
    role: str = "state"  # state / driver / input / context / measurement / latent / algebraic
    observation_status: str = "UNKNOWN"
    dimension: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def observation_class(self) -> str:
        return _norm_status(self.observation_status)

    @property
    def observable(self) -> bool:
        return self.observation_class in {"READY", "CONDITIONAL"}


@dataclass(frozen=True)
class RelationEvidence:
    relation_id: str
    target: str
    sources: Sequence[str]
    kind: str = "dynamic"  # dynamic / algebraic
    origin: str = "discovered"  # known / consensus / discovered / external
    confidence: float = 1.0
    stable: bool = True
    bind_states: Optional[bool] = None
    operator: Optional[str] = None
    terms: Sequence[str] = ()
    coefficients: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def trusted_for_synthesis(self) -> bool:
        return bool(self.stable) and float(self.confidence) > 0.0


@dataclass
class ScopeProposal:
    proposal_id: str
    scope_kind: str
    targets: List[str]
    observed_drivers: List[str]
    latent_boundaries: List[str]
    graph: Dict[str, List[str]]
    relation_ids: List[str]
    score: float
    score_components: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def viable(self) -> bool:
        return not self.latent_boundaries and self.score_components.get("observation", 0.0) > 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["viable"] = self.viable
        return d


@dataclass
class AtlasSynthesisResult:
    status: str
    atlas: MechanismAtlas
    proposals: List[ScopeProposal]
    selected_proposals: List[str]
    uncovered_targets: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": "isgn-atlas-synthesis-v1",
            "status": self.status,
            "atlas": self.atlas.to_dict(),
            "proposals": [p.to_dict() for p in self.proposals],
            "selected_proposals": list(self.selected_proposals),
            "uncovered_targets": list(self.uncovered_targets),
            "metadata": dict(self.metadata),
        }


def _variable_map(variables: Sequence[VariableEvidence]) -> Dict[str, VariableEvidence]:
    out: Dict[str, VariableEvidence] = {}
    for v in variables:
        if v.name in out:
            raise ValueError(f"duplicate variable evidence {v.name!r}")
        out[v.name] = v
    return out


def _relation_map(relations: Sequence[RelationEvidence], kind: str) -> Dict[str, RelationEvidence]:
    """Choose one stable relation per target using origin/confidence priority.

    Consensus/known structure is deliberately preferred over a lower-level backend
    relation. Backend disagreement should be represented by stable=False or by
    omitting a consensus relation, in which case no scope is fabricated.
    """
    origin_rank = {"known": 4, "consensus": 3, "discovered": 2, "external": 1}
    out: Dict[str, RelationEvidence] = {}
    for r in relations:
        if r.kind != kind or not r.trusted_for_synthesis:
            continue
        cur = out.get(r.target)
        key = (origin_rank.get(r.origin, 0), float(r.confidence), len(r.sources))
        if cur is None:
            out[r.target] = r
            continue
        ckey = (origin_rank.get(cur.origin, 0), float(cur.confidence), len(cur.sources))
        if key > ckey:
            out[r.target] = r
    return out


def _bind_edge(r: RelationEvidence, source: str, vmap: Mapping[str, VariableEvidence], target_relations: Mapping[str, RelationEvidence]) -> bool:
    if r.bind_states is not None:
        return bool(r.bind_states)
    # Known state-to-state architecture is treated as a cohesive chain. A stable
    # discovered dependency may still be externalized when the source is an
    # observable boundary; this prevents a strongly coupled industrial plant from
    # collapsing automatically into one giant monolithic scope.
    sv = vmap.get(source)
    if r.origin == "known" and sv is not None and sv.role == "state" and source in target_relations:
        return True
    return False


def _classify_source(source: str, integrated: set[str], vmap: Mapping[str, VariableEvidence]) -> Tuple[str, str]:
    if source in integrated:
        return "internal", source
    v = vmap.get(source)
    if v is None:
        return "latent", source
    if v.role == "latent" or v.observation_class == "BLOCKED":
        return "latent", source
    if v.observable or v.role in {"driver", "input", "context", "measurement", "algebraic"}:
        return "driver", source
    return "latent", source


def _score_dynamic(states: set[str], drivers: set[str], latent: set[str], rels: Sequence[RelationEvidence], vmap: Mapping[str, VariableEvidence]) -> Tuple[float, Dict[str, float]]:
    if not states:
        return -1e9, {"closure": 0.0, "observation": 0.0, "stability": 0.0, "prior_cohesion": 0.0, "complexity_penalty": 0.0}
    total_deps = sum(len(r.sources) for r in rels) or 1
    closure = max(0.0, 1.0 - len(latent) / total_deps)
    oq = []
    for s in states:
        c = vmap.get(s).observation_class if s in vmap else "UNKNOWN"
        oq.append(1.0 if c == "READY" else 0.65 if c == "CONDITIONAL" else 0.0)
    observation = sum(oq) / len(oq)
    stability = sum(float(r.confidence) for r in rels) / max(1, len(rels))
    known_total = 0
    known_internal = 0
    known_externalized = 0
    for r in rels:
        if r.origin != "known":
            continue
        for s in r.sources:
            sv = vmap.get(s)
            if sv is not None and sv.role == "state":
                known_total += 1
                if s in states:
                    known_internal += 1
                elif s in drivers:
                    known_externalized += 1
    prior_cohesion = (known_internal / known_total) if known_total else 0.5
    complexity_penalty = 0.055 * max(0, len(states) - 1) + 0.012 * len(drivers) + 0.16 * len(latent) + 0.10 * known_externalized
    score = 0.32 * closure + 0.23 * observation + 0.20 * min(1.0, stability) + 0.25 * prior_cohesion - complexity_penalty
    return float(score), {
        "closure": float(closure), "observation": float(observation), "stability": float(stability),
        "prior_cohesion": float(prior_cohesion), "complexity_penalty": float(complexity_penalty),
    }


def _dynamic_candidates(vmap: Mapping[str, VariableEvidence], relations: Sequence[RelationEvidence]) -> List[ScopeProposal]:
    rmap = _relation_map(relations, "dynamic")
    relation_targets = {t for t in rmap if vmap.get(t) is not None and vmap[t].role == "state" and vmap[t].observable}

    # Automatic Atlas synthesis must be useful *before* discovery is complete.
    # A trusted known state chain may therefore pull an observable source state
    # into a discovery-planning scope even when that source does not yet have a
    # recovered derivative relation. Such targets are carried explicitly as
    # ``discovery_required_targets``; no relation is fabricated for them.
    planning_states = set(relation_targets)
    changed = True
    while changed:
        changed = False
        for t, r in rmap.items():
            if t not in planning_states or r.origin != "known":
                continue
            for source in map(str, r.sources):
                sv = vmap.get(source)
                if sv is not None and sv.role == "state" and sv.observable and _bind_edge(r, source, vmap, rmap):
                    if source not in planning_states:
                        planning_states.add(source); changed = True

    candidates: Dict[Tuple[str, ...], ScopeProposal] = {}

    def make(seed_states: Iterable[str], label: str) -> None:
        states = set(seed_states) & planning_states
        changed = True
        while changed:
            changed = False
            for t in list(states):
                r = rmap.get(t)
                if r is None:
                    continue
                for source in map(str, r.sources):
                    sv = vmap.get(source)
                    if (source in planning_states and sv is not None and sv.role == "state" and sv.observable
                            and _bind_edge(r, source, vmap, rmap) and source not in states):
                        states.add(source); changed = True

        rels = [rmap[t] for t in sorted(states) if t in rmap]
        missing_targets = sorted(states - set(rmap))
        drivers: set[str] = set(); latent: set[str] = set(); graph: Dict[str, List[str]] = {}
        for r in rels:
            graph[r.target] = list(dict.fromkeys(map(str, r.sources)))
            for source in r.sources:
                cls, name = _classify_source(str(source), states, vmap)
                if cls == "driver": drivers.add(name)
                elif cls == "latent": latent.add(name)

        # Candidate driver pool is a search-space declaration for derivative rows
        # that still require discovery. It is not promoted to observed_drivers
        # until a stable relation selects it. This prevents pre-discovery scope
        # planning from pretending that correlation is already mechanism.
        candidate_driver_pool = sorted(
            name for name, v in vmap.items()
            if name not in states and v.observable and v.role in {"state", "driver", "input", "context", "measurement", "algebraic"}
        ) if missing_targets else []

        score, comps = _score_dynamic(states, drivers, latent, rels, vmap)
        discovery_readiness = len(rels) / max(1, len(states))
        comps["discovery_readiness"] = float(discovery_readiness)
        score -= 0.06 * len(missing_targets)
        key = tuple(sorted(states))
        proposal = ScopeProposal(
            proposal_id="dynamic:" + "+".join(sorted(states)), scope_kind="dynamic",
            targets=sorted(states), observed_drivers=sorted(drivers), latent_boundaries=sorted(latent), graph=graph,
            relation_ids=[r.relation_id for r in rels], score=score, score_components=comps,
            metadata={
                "candidate_origin": label,
                "scope_phase": "DISCOVERY_PLANNING" if missing_targets else "RELATION_EVIDENCE_AVAILABLE",
                "discovery_required_targets": missing_targets,
                "candidate_driver_pool": candidate_driver_pool,
            },
        )
        cur = candidates.get(key)
        if cur is None or proposal.score > cur.score:
            candidates[key] = proposal

    for t in sorted(relation_targets):
        make([t], "atomic")

    # Components induced only by binding edges. Known state-chain components may
    # include targets whose derivative relation is still waiting for discovery.
    adjacency = {t: set() for t in planning_states}
    for t, r in rmap.items():
        if t not in planning_states:
            continue
        for source in map(str, r.sources):
            sv = vmap.get(source)
            if source in planning_states and sv is not None and sv.role == "state" and sv.observable and _bind_edge(r, source, vmap, rmap):
                adjacency[t].add(source); adjacency[source].add(t)
    seen: set[str] = set()
    for t in sorted(planning_states):
        if t in seen: continue
        stack=[t]; comp=set()
        while stack:
            q=stack.pop()
            if q in comp: continue
            comp.add(q); seen.add(q); stack.extend(adjacency.get(q,()))
        if len(comp)>1 and comp & relation_targets:
            make(comp, "binding-component")
    return sorted(candidates.values(), key=lambda p: (-p.score, p.proposal_id))


def _algebraic_candidates(vmap: Mapping[str, VariableEvidence], relations: Sequence[RelationEvidence]) -> List[ScopeProposal]:
    out: List[ScopeProposal] = []
    for target, r in sorted(_relation_map(relations, "algebraic").items()):
        tv = vmap.get(target)
        if tv is not None and tv.observation_class == "BLOCKED":
            continue
        latent=[]; drivers=[]
        for s in r.sources:
            v=vmap.get(str(s))
            if v is None or v.observation_class == "BLOCKED" or v.role == "latent": latent.append(str(s))
            else: drivers.append(str(s))
        observation = 1.0 if tv is None or tv.observation_class in {"READY","UNKNOWN"} else 0.65
        closure = 1.0 if not latent else max(0.0,1-len(latent)/max(1,len(r.sources)))
        complexity_penalty=.012*len(drivers)+.16*len(latent)
        score=.38*closure+.27*observation+.25*min(1.0,float(r.confidence))+.10-complexity_penalty
        out.append(ScopeProposal(
            proposal_id=f"algebraic:{target}", scope_kind="algebraic", targets=[target],
            observed_drivers=sorted(set(drivers)), latent_boundaries=sorted(set(latent)),
            graph={target:list(map(str,r.sources))}, relation_ids=[r.relation_id], score=float(score),
            score_components={"closure":closure,"observation":observation,"stability":min(1.0,float(r.confidence)),"prior_cohesion":0.5,"complexity_penalty":complexity_penalty},
            metadata={"terms":list(r.terms),"coefficients":dict(r.coefficients),"operator":r.operator,"origin":r.origin},
        ))
    return sorted(out,key=lambda p:(-p.score,p.proposal_id))


def _select_dynamic(proposals: Sequence[ScopeProposal], targets: set[str]) -> List[ScopeProposal]:
    selected: List[ScopeProposal] = []
    uncovered=set(targets)
    viable=[p for p in proposals if p.viable]
    while uncovered:
        ranked=[]
        for p in viable:
            new=uncovered & set(p.targets)
            if not new: continue
            # Coverage benefit is explicit; score still encodes closure/observation
            # and the strong reward for keeping known state chains internal.
            utility=p.score + .18*len(new) - .025*max(0,len(p.targets)-len(new))
            ranked.append((utility,len(new),-len(p.targets),p.proposal_id,p))
        if not ranked: break
        p=max(ranked,key=lambda x:(x[0],x[1],x[2],x[3]))[-1]
        selected.append(p); uncovered-=set(p.targets)
        viable=[q for q in viable if q.proposal_id!=p.proposal_id]
    return selected


def _scope_id(p: ScopeProposal, index: int) -> str:
    if p.scope_kind == "algebraic":
        return f"alg-{p.targets[0]}"
    base="-".join(p.targets)
    return f"dyn-{base}" if base else f"dynamic-{index}"


def synthesize_mechanism_atlas(*, system_id: str, variables: Sequence[VariableEvidence], relations: Sequence[RelationEvidence],
                               metadata: Optional[Mapping[str,Any]]=None) -> AtlasSynthesisResult:
    """Propose and safely select a multi-scope MechanismAtlas.

    The synthesis stage does *not* certify mechanisms. It only turns stable
    relation evidence + observation qualification into modular scope proposals.
    Completion remains a separate gate and selected scopes default to
    ``UNCERTIFIED``.
    """
    vmap=_variable_map(variables)
    dynamic=_dynamic_candidates(vmap,relations)
    algebraic=_algebraic_candidates(vmap,relations)
    rmap=_relation_map(relations,"dynamic")
    eligible_targets={t for t in rmap if t in vmap and vmap[t].role=="state" and vmap[t].observable}
    chosen_dyn=_select_dynamic(dynamic,eligible_targets)
    chosen_alg=[p for p in algebraic if p.viable]
    selected=chosen_dyn+chosen_alg

    atlas=MechanismAtlas(str(system_id),metadata={"synthesis":"automatic-proposal-v1",**dict(metadata or {})})
    proposal_to_scope: Dict[str,str]={}
    for i,p in enumerate(selected):
        sid=_scope_id(p,i)
        # avoid rare collisions
        base=sid; k=2
        while sid in atlas.scopes: sid=f"{base}-{k}"; k+=1
        proposal_to_scope[p.proposal_id]=sid
        planning = bool(p.metadata.get("discovery_required_targets"))
        atlas.add_scope(MechanismScopeRecord(
            scope_id=sid,target_states=list(p.targets),scope_kind=p.scope_kind,observed_drivers=list(p.observed_drivers),
            latent_boundaries=list(p.latent_boundaries),graph=dict(p.graph),
            observation_status="SYNTHESIS_PLANNED" if planning else "SYNTHESIS_QUALIFIED",
            completion_status="UNCERTIFIED",metadata={"synthesis_score":p.score,"synthesis_components":p.score_components,
            "relation_ids":p.relation_ids,**p.metadata},
        ))

    # Interfaces are inferred whenever a produced target is consumed by another
    # selected scope. The interface carries only the boundary variable; it does
    # not invent propagation semantics.
    producers: Dict[str,List[str]]={}
    for sid,s in atlas.scopes.items():
        for t in s.target_states: producers.setdefault(t,[]).append(sid)
    iface_seen=set()
    for target_sid,s in atlas.scopes.items():
        consumed=set(s.observed_drivers)|{x for srcs in s.graph.values() for x in srcs if x not in s.target_states}
        for var in sorted(consumed):
            for source_sid in producers.get(var,[]):
                if source_sid==target_sid: continue
                key=(source_sid,target_sid,var)
                if key in iface_seen: continue
                iface_seen.add(key)
                atlas.add_interface(ScopeInterface(source_sid,target_sid,[var],kind="inferred_measured_boundary",
                                                   metadata={"source":"automatic-synthesis"}))

    selected_targets={t for p in selected for t in p.targets}
    uncovered=sorted(eligible_targets-selected_targets)
    # Blocked states and stable-relation targets that could not be selected are
    # carried explicitly into the Atlas rather than silently discarded.
    for name,v in sorted(vmap.items()):
        if v.role=="state" and v.observation_class=="BLOCKED":
            atlas.add_uncovered(UncoveredRegion(f"observation-{name}",[name],"OBSERVATION_REMEDIATION",
                                                "state is blocked by observation qualification",
                                                metadata={"observation_status":v.observation_status}))
    for t in uncovered:
        atlas.add_uncovered(UncoveredRegion(f"scope-{t}",[t],"MECHANISM_SCOPE_REQUIRED",
                                            "stable relation exists but no viable closed/observed scope was selected"))
    # State variables without a stable relation are important missing evidence,
    # especially under backend disagreement.
    for name,v in sorted(vmap.items()):
        if v.role=="state" and v.observable and name not in rmap and name not in selected_targets:
            atlas.add_uncovered(UncoveredRegion(f"relation-{name}",[name],"STABLE_RELATION_REQUIRED",
                                                "no stable dynamic relation is available for automatic scope synthesis"))

    status="ATLAS_PROPOSED" if atlas.scopes else "NO_VIABLE_SCOPE"
    if atlas.scopes and atlas.uncovered_regions: status="PARTIAL_ATLAS_PROPOSED"
    return AtlasSynthesisResult(status,atlas,sorted(dynamic+algebraic,key=lambda p:(p.scope_kind,-p.score,p.proposal_id)),
                                [p.proposal_id for p in selected],uncovered,
                                metadata={"n_variables":len(vmap),"n_relations":len(relations),"n_selected_scopes":len(atlas.scopes),
                                          "certification_performed":False})


def synthesis_input_from_dict(d: Mapping[str,Any]) -> Tuple[str,List[VariableEvidence],List[RelationEvidence],Dict[str,Any]]:
    system_id=str(d["system_id"])
    variables=[VariableEvidence(**x) for x in d.get("variables",[])]
    relations=[RelationEvidence(**x) for x in d.get("relations",[])]
    return system_id,variables,relations,dict(d.get("metadata",{}))


def variable_evidence_from_public_dataset(dataset, gate_report=None) -> List[VariableEvidence]:
    """Convert the common PublicDataset/Gate contracts into synthesis evidence."""
    state_status: Dict[str,str]={}
    if gate_report is not None:
        for s in getattr(gate_report,'states',[]) or []:
            st=getattr(s,'status','UNKNOWN')
            state_status[str(getattr(s,'state'))] = 'BLOCKED' if str(st).upper()=='FAIL' else 'CONDITIONAL' if str(st).upper()=='WARN' else 'READY'
    out=[]
    for v in dataset.schema.get('states',[]):
        name=str(v['name']); out.append(VariableEvidence(name,'state',state_status.get(name,'UNKNOWN'),str(v.get('dim','')),metadata=dict(v)))
    for v in dataset.schema.get('inputs',[]):
        role=str(v.get('role','input')); name=str(v['name'])
        out.append(VariableEvidence(name,'context' if role=='context' else 'input','READY',str(v.get('dim','')),metadata=dict(v)))
    return out


def relation_evidence_from_recovery(summary, *, origin: str='consensus', confidence: Optional[float]=None,
                                    stable: bool=True, known_present: Optional[Mapping[str,Sequence[str]]]=None) -> List[RelationEvidence]:
    """Convert a backend-agnostic RecoverySummary into synthesis relations."""
    if confidence is None:
        err=float(getattr(summary,'val_nrmse',1.0)); confidence=max(0.0,1.0-min(1.0,err))
    known={str(t):set(map(str,s)) for t,s in (known_present or {}).items()}
    out=[]
    for target,sources in getattr(summary,'graph',{}).items():
        target=str(target); srcs=list(map(str,sources))
        # Preserve a known state chain as an explicit binding relation only when
        # the full recovered target row is exactly that known chain. Mixed rows
        # remain consensus relations so observed boundaries may be externalized.
        is_known=target in known and set(srcs)==known[target]
        out.append(RelationEvidence(f'{getattr(summary,"backend","backend")}:{target}',target,srcs,
                                    origin='known' if is_known else origin,confidence=float(confidence),stable=bool(stable),
                                    bind_states=True if is_known else None,
                                    metadata={'backend':getattr(summary,'backend',''),'val_nrmse':float(getattr(summary,'val_nrmse',float('nan')))}))
    return out


def relation_evidence_from_algebraic(result, *, origin: str='consensus', confidence: Optional[float]=None,
                                     stable: bool=True) -> RelationEvidence:
    if confidence is None:
        confidence=max(0.0,1.0-min(1.0,float(result.validation_nrmse)))
    return RelationEvidence(f'{result.backend}:{result.target}',str(result.target),list(map(str,result.inputs)),kind='algebraic',
                            origin=origin,confidence=float(confidence),stable=bool(stable),terms=list(result.selected_terms),
                            coefficients=dict(result.coefficients),metadata={'validation_nrmse':float(result.validation_nrmse),
                            'backend':str(result.backend),'intercept':float(result.intercept)})
