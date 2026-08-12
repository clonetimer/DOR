from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from .canonical_ir import (
    Dim, DIMLESS, TIME, VarSpec, CNode, CanonicalIR,
    CanonicalStateWrapper, compile_canonical,
)

@dataclass(frozen=True)
class StorageSpec:
    name: str
    dim: Dim
    min_value: float | None = 0.0
    max_value: float | None = None

@dataclass(frozen=True)
class InputSpec:
    name: str
    dim: Dim

@dataclass(frozen=True)
class DriveSpec:
    id: str
    target: str
    input: str
    gain_dim: Dim
    init: float = 0.01

@dataclass(frozen=True)
class ConditionalDriveSpec:
    id: str
    target: str
    input: str
    condition: str
    threshold: float
    gain_dim: Dim
    init: float = 0.01

@dataclass(frozen=True)
class FluxSpec:
    """Positive constitutive flux routed from one storage to another or the environment.

    A single constitutive law is evaluated from ``source``. Independent positive source/target
    rate gains are allowed because stored states may be expressed in different coordinates
    (e.g. level rather than volume), so the same physical flux can map through different
    storage-capacity factors.
    """
    id: str
    source: str
    target: str | None
    law: str  # linear | sqrt | power | local_neural
    source_gain_dim: Dim
    target_gain_dim: Dim | None = None
    source_gain_init: float = 0.01
    target_gain_init: float = 0.01
    flux_dim: Dim | None = None  # required for local_neural
    exponent: float | None = None  # required for power
    hidden: int = 1
    share_key: str | None = None

@dataclass(frozen=True)
class ConservationIR:
    storages: tuple[StorageSpec, ...]
    inputs: tuple[InputSpec, ...]
    drives: tuple[DriveSpec, ...] = ()
    conditional_drives: tuple[ConditionalDriveSpec, ...] = ()
    fluxes: tuple[FluxSpec, ...] = ()
    dt: float = 1.0
    metadata: tuple[tuple[str,str], ...] = ()


def _law_dim(law: str, source_dim: Dim, flux_dim: Dim | None, exponent: float | None = None) -> Dim:
    if law == 'linear':
        return source_dim
    if law == 'sqrt':
        return source_dim ** Fraction(1,2)
    if law == 'power':
        if exponent is None or exponent<=0: raise ValueError('power flux requires positive exponent')
        return source_dim ** Fraction(str(exponent))
    if law == 'local_neural':
        if flux_dim is None:
            raise ValueError('local_neural flux requires flux_dim')
        return flux_dim
    raise ValueError(f'unknown flux law {law}')


def validate_conservation_ir(ir: ConservationIR) -> bool:
    if not ir.storages:
        raise ValueError('no storages')
    if ir.dt <= 0:
        raise ValueError('dt must be positive')
    sd={s.name:s.dim for s in ir.storages}; idim={x.name:x.dim for x in ir.inputs}
    if len(sd)!=len(ir.storages) or len(idim)!=len(ir.inputs) or set(sd)&set(idim):
        raise ValueError('duplicate storage/input names')
    ids=set()
    for s in ir.storages:
        if s.min_value is not None and s.max_value is not None and s.min_value>s.max_value:
            raise ValueError(f'{s.name}: invalid bounds')
    for d in ir.drives:
        if d.id in ids: raise ValueError('duplicate term id')
        ids.add(d.id)
        if d.target not in sd or d.input not in idim: raise ValueError(f'{d.id}: bad drive reference')
        if idim[d.input]*d.gain_dim != sd[d.target]/TIME:
            raise ValueError(f'{d.id}: drive dimension mismatch')
    for d in ir.conditional_drives:
        if d.id in ids: raise ValueError('duplicate term id')
        ids.add(d.id)
        if d.target not in sd or d.input not in idim or d.condition not in sd:
            raise ValueError(f'{d.id}: bad conditional drive reference')
        if idim[d.input]*d.gain_dim != sd[d.target]/TIME:
            raise ValueError(f'{d.id}: conditional drive dimension mismatch')
    for f in ir.fluxes:
        if f.id in ids: raise ValueError('duplicate term id')
        ids.add(f.id)
        if f.source not in sd or (f.target is not None and f.target not in sd):
            raise ValueError(f'{f.id}: bad flux reference')
        base_dim=_law_dim(f.law,sd[f.source],f.flux_dim,f.exponent)
        if base_dim*f.source_gain_dim != sd[f.source]/TIME:
            raise ValueError(f'{f.id}: source gain dimension mismatch')
        if f.target is not None:
            if f.target_gain_dim is None: raise ValueError(f'{f.id}: missing target gain dimension')
            if base_dim*f.target_gain_dim != sd[f.target]/TIME:
                raise ValueError(f'{f.id}: target gain dimension mismatch')
    return True


def lower_conservation_ir(ir: ConservationIR):
    validate_conservation_ir(ir)
    sd={s.name:s.dim for s in ir.storages}; idim={x.name:x.dim for x in ir.inputs}
    inputs=tuple([VarSpec(k,v) for k,v in sd.items()]+[VarSpec(k,v) for k,v in idim.items()])
    nodes=[]; terms={s.name:[] for s in ir.storages}
    for d in ir.drives:
        nid=f'{d.id}__rate'
        nodes.append(CNode(nid,'positive_gain',(d.input,),{
            'learnable':True,'init':float(d.init),'gain_dim':d.gain_dim,'eps':1e-8}))
        terms[d.target].append((+1,nid))
    for d in ir.conditional_drives:
        gate=f'{d.id}__gate'
        nodes.append(CNode(gate,'threshold_gate',(d.condition,d.input),{'threshold':float(d.threshold),'mode':'ge'}))
        nid=f'{d.id}__rate'
        nodes.append(CNode(nid,'positive_gain',(gate,),{'learnable':True,'init':float(d.init),'gain_dim':d.gain_dim,'eps':1e-8}))
        terms[d.target].append((+1,nid))
    for f in ir.fluxes:
        base=f'{f.id}__law'
        if f.law=='linear':
            base=f.source
        elif f.law=='sqrt':
            nodes.append(CNode(base,'sqrt_pos',(f.source,),{'sqrt_eps':0.0}))
        elif f.law=='power':
            nodes.append(CNode(base,'power_pos',(f.source,),{'exponent':float(f.exponent),'eps':0.0}))
        elif f.law=='local_neural':
            nodes.append(CNode(base,'local_neural',(f.source,),{
                'output_dim':f.flux_dim,'hidden':int(f.hidden),'share_key':f.share_key}))
        src=f'{f.id}__source_rate'
        nodes.append(CNode(src,'positive_gain',(base,),{
            'learnable':True,'init':float(f.source_gain_init),'gain_dim':f.source_gain_dim,'eps':1e-8}))
        terms[f.source].append((-1,src))
        if f.target is not None:
            tgt=f'{f.id}__target_rate'
            nodes.append(CNode(tgt,'positive_gain',(base,),{
                'learnable':True,'init':float(f.target_gain_init),'gain_dim':f.target_gain_dim,'eps':1e-8}))
            terms[f.target].append((+1,tgt))
    rate_nodes=[]
    outputs=[]; expected=[]
    for s in ir.storages:
        tt=terms[s.name]
        if not tt:
            cur=f'{s.name}__rate_zero'; nodes.append(CNode(cur,'zero_like',(s.name,),{}))
        else:
            sign,nid=tt[0]
            if sign<0:
                cur=f'{s.name}__term0_neg'; nodes.append(CNode(cur,'negate',(nid,),{}))
            else:
                cur=nid
            for j,(sign,nid) in enumerate(tt[1:],1):
                nxt=f'{s.name}__rate_{j}'
                nodes.append(CNode(nxt,'add' if sign>0 else 'subtract',(cur,nid),{})); cur=nxt
        rate_nodes.append(cur)
        raw=f'{s.name}__euler'
        nodes.append(CNode(raw,'euler_state_update',(s.name,cur),{'dt':float(ir.dt),'dt_dim':TIME}))
        out=f'{s.name}__next'
        if s.min_value is not None or s.max_value is not None:
            nodes.append(CNode(out,'clamp',(raw,),{'min_value':s.min_value,'max_value':s.max_value}))
        else:
            # keep a distinct deterministic output node without adding a new identity primitive
            nodes.append(CNode(out,'positive_gain',(raw,),{'learnable':False,'value':1.0,'gain_dim':DIMLESS}))
        outputs.append(out); expected.append((out,s.dim))
    meta=(('source_ir','conservation'),('dt',repr(float(ir.dt))))+tuple(ir.metadata)
    cir=CanonicalIR(inputs,tuple(nodes),tuple(outputs),tuple(expected),meta)
    return cir,tuple(s.name for s in ir.storages),tuple(rate_nodes)


def compile_conservation_ir(ir: ConservationIR):
    cir,names,rate_nodes=lower_conservation_ir(ir)
    return CanonicalStateWrapper(compile_canonical(cir),names,rate_nodes),cir
