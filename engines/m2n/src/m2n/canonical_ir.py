from __future__ import annotations
from dataclasses import dataclass, field, asdict
from fractions import Fraction
from typing import Any
import hashlib, json
import torch
from torch import nn
import torch.nn.functional as F

@dataclass(frozen=True)
class Dim:
    powers: tuple[tuple[str, Fraction], ...] = ()
    @staticmethod
    def of(**powers):
        items=[]
        for k,v in powers.items():
            f=Fraction(v)
            if f: items.append((str(k),f))
        return Dim(tuple(sorted(items)))
    @staticmethod
    def from_pairs(pairs):
        return Dim(tuple(sorted((str(k),Fraction(v)) for k,v in pairs if Fraction(v))))
    def _d(self): return dict(self.powers)
    def __mul__(self,other:'Dim'):
        d=self._d()
        for k,v in other.powers:d[k]=d.get(k,Fraction(0))+v
        return Dim.of(**{k:v for k,v in d.items() if v})
    def __truediv__(self,other:'Dim'):
        d=self._d()
        for k,v in other.powers:d[k]=d.get(k,Fraction(0))-v
        return Dim.of(**{k:v for k,v in d.items() if v})
    def __pow__(self,p): return Dim.of(**{k:v*Fraction(p) for k,v in self.powers})
    def serial(self): return [[k,[v.numerator,v.denominator]] for k,v in self.powers]
    def __str__(self):
        if not self.powers:return '1'
        return '*'.join(f'{k}^{v}' if v!=1 else k for k,v in self.powers)

DIMLESS=Dim(); TIME=Dim.of(time=1)

@dataclass(frozen=True)
class VarSpec:
    name:str
    dim:Dim

@dataclass(frozen=True)
class CNode:
    id:str
    op:str
    inputs:tuple[str,...]=()
    params:dict[str,Any]=field(default_factory=dict)

@dataclass(frozen=True)
class CanonicalIR:
    inputs:tuple[VarSpec,...]
    nodes:tuple[CNode,...]
    outputs:tuple[str,...]
    expected_output_dims:tuple[tuple[str,Dim],...]
    metadata:tuple[tuple[str,str],...]=()

class LocalLaw(nn.Module):
    def __init__(self,hidden=1):
        super().__init__();self.f1=nn.Linear(1,hidden);self.f2=nn.Linear(hidden,1)
    def forward(self,x):return F.softplus(self.f2(torch.tanh(self.f1(x[:,None]))).squeeze(1))

class SignedLocalLaw(nn.Module):
    """Small scalar constitutive law with signed output.

    This is intentionally separate from ``LocalLaw`` so existing positive-flux K3
    semantics (e.g. E61) remain frozen.  Use only where the constitutive quantity
    is allowed to change sign, such as force/torque-like local laws.
    """
    def __init__(self,hidden=1):
        super().__init__();self.f1=nn.Linear(1,hidden);self.f2=nn.Linear(hidden,1)
    def forward(self,x):return self.f2(torch.tanh(self.f1(x[:,None]))).squeeze(1)

class SignedGainLocalLaw(nn.Module):
    """Three-parameter scalar nonlinear gain with zero output at zero.

    For hidden=1 the network has three parameters.  The output is ``x`` times
    a positive learned gain, so it preserves input sign and cannot create an
    additive bias.  It is useful as a capacity-matched local actuation-law
    control without overlapping an explicit constant-load term.
    """
    def __init__(self,hidden=1):
        super().__init__();self.f1=nn.Linear(1,hidden,bias=True);self.f2=nn.Linear(hidden,1,bias=False)
    def forward(self,x):
        ax=torch.abs(x)[:,None]
        gain=F.softplus(self.f2(torch.tanh(self.f1(ax))).squeeze(1))
        return x*gain

class DissipativeLocalLaw(nn.Module):
    """Local signed resistance constrained to oppose motion.

    The law sees only local velocity information ``(|v|, sign(v))``.  It predicts
    a non-negative magnitude and re-applies ``sign(v)`` so ``r(v)*v >= 0`` and
    ``r(0)=0`` by construction.  The output layer intentionally has no bias: for
    hidden=1 this gives three neural parameters; together with one drive gain and
    one explicit constant-load parameter the E62 K3 candidate has five trainable
    parameters, matching the asymmetric viscous+Coulomb K2 candidate.
    """
    def __init__(self,hidden=1):
        super().__init__();self.f1=nn.Linear(2,hidden,bias=False);self.f2=nn.Linear(hidden,1,bias=False)
    def forward(self,x):
        ax=torch.abs(x); sx=torch.sign(x)
        z=torch.stack((ax,sx),dim=1)
        mag=F.softplus(self.f2(torch.tanh(self.f1(z))).squeeze(1))
        return sx*mag


class StribeckResistanceLaw(nn.Module):
    """Finite K2 Stribeck resistance family, not a neural residual.

    r(v)=b*v + [Fc + dF*exp(-(abs(v)/vs)^2)]*tanh(v/vsmooth)

    All five parameters are constrained positive. ``dF`` is the breakaway
    excess, so breakaway magnitude is ``Fc+dF``.  This is a compact reusable
    engineering constitutive family for translational/rotational resistance.
    """
    def __init__(self, viscous=1.0, coulomb=1.0, breakaway_delta=0.5,
                 stribeck_velocity=0.02, smoothing_velocity=1e-3):
        super().__init__()
        def raw(x):
            x=max(float(x),1e-8)
            return torch.log(torch.expm1(torch.tensor(x)))
        self.raw_viscous=nn.Parameter(raw(viscous))
        self.raw_coulomb=nn.Parameter(raw(coulomb))
        self.raw_breakaway_delta=nn.Parameter(raw(breakaway_delta))
        self.raw_stribeck_velocity=nn.Parameter(raw(stribeck_velocity))
        self.raw_smoothing_velocity=nn.Parameter(raw(smoothing_velocity))
    @staticmethod
    def _pos(x):
        return F.softplus(x)+1e-8
    def physical_parameters(self):
        return {
            'viscous': self._pos(self.raw_viscous),
            'coulomb': self._pos(self.raw_coulomb),
            'breakaway_delta': self._pos(self.raw_breakaway_delta),
            'stribeck_velocity': self._pos(self.raw_stribeck_velocity),
            'smoothing_velocity': self._pos(self.raw_smoothing_velocity),
        }
    def forward(self,v):
        p=self.physical_parameters()
        av=torch.abs(v)
        stribeck=p['coulomb']+p['breakaway_delta']*torch.exp(-torch.square(av/p['stribeck_velocity']))
        return p['viscous']*v + stribeck*torch.tanh(v/p['smoothing_velocity'])

class CanonicalGraph(nn.Module):
    def __init__(self,ir:CanonicalIR):
        super().__init__();self.ir=ir
        self.raw_gains=nn.ParameterDict();self.signed_gains=nn.ParameterDict();self.signed_constants=nn.ParameterDict();self.local_modules=nn.ModuleDict();self.local_module_key={}
        for n in ir.nodes:
            if n.op=='positive_gain' and n.params.get('learnable',False):
                init=max(float(n.params.get('init',1.0)),1e-8)
                raw=torch.log(torch.expm1(torch.tensor(init)))
                self.raw_gains[n.id]=nn.Parameter(raw.clone())
            elif n.op=='signed_gain' and n.params.get('learnable',False):
                self.signed_gains[n.id]=nn.Parameter(torch.tensor(float(n.params.get('init',0.0))))
            elif n.op=='signed_constant_like' and n.params.get('learnable',False):
                self.signed_constants[n.id]=nn.Parameter(torch.tensor(float(n.params.get('init',0.0))))
            elif n.op in ('local_neural','signed_local_neural','signed_gain_local_neural','dissipative_local_neural'):
                key=str(n.params.get('share_key') or n.id)
                cls=LocalLaw if n.op=='local_neural' else (SignedLocalLaw if n.op=='signed_local_neural' else (SignedGainLocalLaw if n.op=='signed_gain_local_neural' else DissipativeLocalLaw))
                if key not in self.local_modules:self.local_modules[key]=cls(int(n.params.get('hidden',1)))
                self.local_module_key[n.id]=key
            elif n.op=='stribeck_resistance':
                key=str(n.params.get('share_key') or n.id)
                if key not in self.local_modules:
                    self.local_modules[key]=StribeckResistanceLaw(
                        n.params.get('viscous_init',1.0), n.params.get('coulomb_init',1.0),
                        n.params.get('breakaway_delta_init',0.5), n.params.get('stribeck_velocity_init',0.02),
                        n.params.get('smoothing_velocity_init',1e-3))
                self.local_module_key[n.id]=key
    def positive_gain(self,node_id):
        n=next(x for x in self.ir.nodes if x.id==node_id)
        eps=float(n.params.get('eps',1e-8))
        return F.softplus(self.raw_gains[node_id])+eps
    def forward_env(self,**inputs):
        env=dict(inputs)
        for n in self.ir.nodes:
            xs=[env[k] for k in n.inputs]
            if n.op=='add':env[n.id]=xs[0]+xs[1]
            elif n.op=='subtract':env[n.id]=xs[0]-xs[1]
            elif n.op=='negate':env[n.id]=-xs[0]
            elif n.op=='zero_like':env[n.id]=torch.zeros_like(xs[0])
            elif n.op=='positive_gain':
                if n.params.get('learnable',False):g=self.positive_gain(n.id)
                else:g=torch.as_tensor(float(n.params['value']),dtype=xs[0].dtype,device=xs[0].device)
                env[n.id]=g*xs[0]
            elif n.op=='signed_gain':
                if n.params.get('learnable',False):g=self.signed_gains[n.id]
                else:g=torch.as_tensor(float(n.params['value']),dtype=xs[0].dtype,device=xs[0].device)
                env[n.id]=g*xs[0]
            elif n.op=='signed_constant_like':
                if n.params.get('learnable',False):c=self.signed_constants[n.id]
                else:c=torch.as_tensor(float(n.params['value']),dtype=xs[0].dtype,device=xs[0].device)
                env[n.id]=torch.ones_like(xs[0])*c
            elif n.op in ('local_neural','signed_local_neural','signed_gain_local_neural','dissipative_local_neural','stribeck_resistance'):env[n.id]=self.local_modules[self.local_module_key[n.id]](xs[0])
            elif n.op=='sign':env[n.id]=torch.sign(xs[0])
            elif n.op=='abs':env[n.id]=torch.abs(xs[0])
            elif n.op=='multiply':env[n.id]=xs[0]*xs[1]
            elif n.op=='sqrt_pos':env[n.id]=torch.sqrt(torch.clamp(xs[0],min=0)+float(n.params.get('sqrt_eps',1e-8)))
            elif n.op=='power_pos':env[n.id]=torch.pow(torch.clamp(xs[0],min=0)+float(n.params.get('eps',0.0)),float(n.params['exponent']))
            elif n.op=='clamp':
                lo=n.params.get('min_value',None); hi=n.params.get('max_value',None)
                env[n.id]=torch.clamp(xs[0],min=None if lo is None else float(lo),max=None if hi is None else float(hi))
            elif n.op=='threshold_gate':
                th=float(n.params['threshold']); mode=n.params.get('mode','ge')
                if mode!='ge':raise ValueError(f'unsupported threshold mode {mode}')
                env[n.id]=torch.where(xs[0]>=th,xs[1],torch.zeros_like(xs[1]))
            elif n.op=='sqrt_pos_diff_scaled':
                q=float(n.params['coeff'])*torch.sqrt(torch.clamp(xs[0]-xs[1],min=0)+float(n.params.get('sqrt_eps',1e-8)))
                if len(xs)==3:q=q*xs[2]
                env[n.id]=q
            elif n.op=='signed_sqrt_diff_scaled':
                dp=xs[0]-xs[1]
                q=float(n.params['coeff'])*torch.sign(dp)*torch.sqrt(torch.abs(dp)+float(n.params.get('sqrt_eps',1e-8)))
                if len(xs)==3:q=q*xs[2]
                env[n.id]=q
            elif n.op=='euler_state_update':env[n.id]=xs[0]+float(n.params['dt'])*xs[1]
            else:raise KeyError(n.op)
        return env
    def forward(self,**inputs):
        env=self.forward_env(**inputs)
        return {k:env[k] for k in self.ir.outputs}

class CanonicalStateWrapper(nn.Module):
    def __init__(self,graph:CanonicalGraph,state_names:tuple[str,...],rate_nodes:tuple[str,...]):
        super().__init__();self.graph=graph;self.state_names=state_names;self.rate_nodes=rate_nodes
    @property
    def raw_gains(self):return self.graph.raw_gains
    @property
    def local_modules(self):return self.graph.local_modules
    def positive_gain(self,id):return self.graph.positive_gain(id)
    def forward(self,state:torch.Tensor,**inputs):
        if state.ndim!=2 or state.shape[1]!=len(self.state_names):raise ValueError('state shape mismatch')
        kw={name:state[:,i] for i,name in enumerate(self.state_names)};kw.update(inputs)
        out=self.graph(**kw)
        return torch.stack([out[f'{name}__next'] for name in self.state_names],dim=1)
    def rate(self,state:torch.Tensor,**inputs):
        if state.ndim!=2 or state.shape[1]!=len(self.state_names):raise ValueError('state shape mismatch')
        kw={name:state[:,i] for i,name in enumerate(self.state_names)};kw.update(inputs)
        env=self.graph.forward_env(**kw)
        return torch.stack([env[k] for k in self.rate_nodes],dim=1)

def _dim_from_serial(v):
    if isinstance(v,Dim):return v
    return Dim.from_pairs(v)

def validate_canonical(ir:CanonicalIR):
    env={v.name:v.dim for v in ir.inputs};known=set(env);ids=set()
    for n in ir.nodes:
        if n.id in known or n.id in ids:raise ValueError(f'duplicate id {n.id}')
        miss=[x for x in n.inputs if x not in known]
        if miss:raise ValueError(f'{n.id}: unavailable {miss}')
        xs=[env[k] for k in n.inputs]
        if n.op in ('add','subtract'):
            if len(xs)!=2 or xs[0]!=xs[1]:raise ValueError(f'{n.id}: {n.op} dimensions mismatch')
            out=xs[0]
        elif n.op in ('negate','zero_like'):
            if len(xs)!=1:raise ValueError(f'{n.id}: unary op arity')
            out=xs[0]
        elif n.op in ('positive_gain','signed_gain'):
            out=xs[0]*_dim_from_serial(n.params.get('gain_dim',DIMLESS))
        elif n.op=='signed_constant_like':
            if len(xs)!=1:raise ValueError(f'{n.id}: signed_constant_like arity')
            out=_dim_from_serial(n.params['output_dim'])
        elif n.op in ('local_neural','signed_local_neural','signed_gain_local_neural','dissipative_local_neural','stribeck_resistance'):
            if len(xs)!=1:raise ValueError(f'{n.id}: local constitutive arity')
            out=_dim_from_serial(n.params['output_dim'])
        elif n.op=='sign':
            if len(xs)!=1:raise ValueError(f'{n.id}: sign arity')
            out=DIMLESS
        elif n.op=='abs':
            if len(xs)!=1:raise ValueError(f'{n.id}: abs arity')
            out=xs[0]
        elif n.op=='multiply':
            if len(xs)!=2:raise ValueError(f'{n.id}: multiply arity')
            out=xs[0]*xs[1]
        elif n.op=='sqrt_pos':
            if len(xs)!=1:raise ValueError(f'{n.id}: sqrt_pos arity')
            out=xs[0]**Fraction(1,2)
        elif n.op=='power_pos':
            if len(xs)!=1:raise ValueError(f'{n.id}: power_pos arity')
            exponent=Fraction(str(n.params['exponent']))
            if exponent<=0:raise ValueError(f'{n.id}: nonpositive exponent')
            out=xs[0]**exponent
        elif n.op=='clamp':
            if len(xs)!=1:raise ValueError(f'{n.id}: clamp arity')
            if n.params.get('min_value') is None and n.params.get('max_value') is None:raise ValueError(f'{n.id}: empty clamp')
            out=xs[0]
        elif n.op=='threshold_gate':
            if len(xs)!=2:raise ValueError(f'{n.id}: threshold_gate arity')
            if n.params.get('mode','ge')!='ge':raise ValueError(f'{n.id}: unsupported threshold mode')
            out=xs[1]
        elif n.op in ('sqrt_pos_diff_scaled','signed_sqrt_diff_scaled'):
            if len(xs) not in (2,3) or xs[0]!=xs[1]:raise ValueError(f'{n.id}: bad flow inputs')
            scale=DIMLESS if len(xs)==2 else xs[2]
            coeff=_dim_from_serial(n.params['coeff_dim'])
            out=(xs[0]**Fraction(1,2))*coeff*scale
        elif n.op=='euler_state_update':
            if len(xs)!=2:raise ValueError(f'{n.id}: state update arity')
            dt_dim=_dim_from_serial(n.params.get('dt_dim',TIME))
            if xs[0]!=xs[1]*dt_dim:raise ValueError(f'{n.id}: state != dt*rate')
            out=xs[0]
        else:raise ValueError(f'unknown op {n.op}')
        env[n.id]=out;known.add(n.id);ids.add(n.id)
    exp=dict(ir.expected_output_dims)
    for o in ir.outputs:
        if o not in env:raise ValueError(f'missing output {o}')
        if o not in exp:raise ValueError(f'missing expected dimension {o}')
        if env[o]!=exp[o]:raise ValueError(f'output {o}: {env[o]} != {exp[o]}')
    return env

def compile_canonical(ir:CanonicalIR):validate_canonical(ir);return CanonicalGraph(ir)

def canonical_signature(ir:CanonicalIR)->str:
    def dimj(d):return d.serial()
    obj={
      'inputs':[{'name':v.name,'dim':dimj(v.dim)} for v in ir.inputs],
      'nodes':[{'id':n.id,'op':n.op,'inputs':list(n.inputs),'params':_json_params(n.params)} for n in ir.nodes],
      'outputs':list(ir.outputs),'expected':[[k,dimj(d)] for k,d in ir.expected_output_dims],
      'metadata':[list(x) for x in ir.metadata],
    }
    s=json.dumps(obj,sort_keys=True,separators=(',',':'))
    return hashlib.sha256(s.encode()).hexdigest()

def _json_params(p):
    out={}
    for k,v in p.items():
        if isinstance(v,Dim):out[k]=v.serial()
        elif isinstance(v,tuple) and v and isinstance(v[0],tuple):
            try:out[k]=Dim.from_pairs(v).serial()
            except Exception:out[k]=v
        else:out[k]=v
    return out

# ---- lowering adapters --------------------------------------------------
def lower_e04(ir,contract)->CanonicalIR:
    inputs=tuple(VarSpec(k,Dim.from_pairs(v)) for k,v in contract.input_dimensions.items())
    nodes=[]
    for n in ir.nodes:
        p=dict(n.params)
        if n.op=='positive_gain':p={'learnable':bool(p.get('learnable',False)),'init':p.get('init',1.0),'value':p.get('value'), 'gain_dim':Dim.from_pairs(p.get('gain_dimension',())), 'eps':1e-8}
        elif n.op=='euler_state_update':p={'dt':float(p['dt']),'dt_dim':Dim.from_pairs(p.get('dt_dimension',(('t',1),)))}
        nodes.append(CNode(n.id,n.op,tuple(n.inputs),p))
    expected=tuple((k,Dim.from_pairs(v)) for k,v in contract.expected_output_dimensions.items())
    return CanonicalIR(inputs,tuple(nodes),tuple(ir.outputs),expected,(('source_ir','e04_dag'),))

def lower_e16(ir)->tuple[CanonicalIR,tuple[str,...]]:
    state_dims={s.name:Dim.from_pairs(s.dim.powers) for s in ir.states};input_dims={x.name:Dim.from_pairs(x.dim.powers) for x in ir.inputs}
    inputs=tuple([VarSpec(k,v) for k,v in state_dims.items()]+[VarSpec(k,v) for k,v in input_dims.items()])
    nodes=[];terms={s.name:[] for s in ir.states}
    rate_dim={k:v/TIME for k,v in state_dims.items()}
    for x in ir.local_neurals:
        nid=x.id;nodes.append(CNode(nid,'local_neural',(x.state,),{'output_dim':rate_dim[x.state],'hidden':int(x.hidden),'share_key':x.share_key}))
        terms[x.state].append((-1,nid))
    for d in ir.drives:
        nodes.append(CNode(d.id,'positive_gain',(d.input,),{'learnable':True,'init':float(d.init),'gain_dim':Dim.from_pairs(d.gain_dim.powers),'eps':1e-6}))
        terms[d.target].append((+1,d.id))
    for e in ir.edges:
        ins=(e.source,e.target) if e.scale_input is None else (e.source,e.target,e.scale_input)
        nodes.append(CNode(e.id,'sqrt_pos_diff_scaled',ins,{'coeff':float(e.coeff),'coeff_dim':Dim.from_pairs(e.coeff_dim.powers),'sqrt_eps':1e-8}))
        terms[e.source].append((-1,e.id));terms[e.target].append((+1,e.id))
    # deterministic accumulation for each state
    rate_nodes=[]
    for s in ir.states:
        tt=terms[s.name]
        if not tt:
            z=f'{s.name}__rate_zero';nodes.append(CNode(z,'zero_like',(s.name,),{}));cur=z
        else:
            sign,nid=tt[0]
            if sign<0:
                cur=f'{s.name}__term0_neg';nodes.append(CNode(cur,'negate',(nid,),{}))
            else:cur=nid
            for j,(sign,nid) in enumerate(tt[1:],1):
                nxt=f'{s.name}__rate_{j}'
                nodes.append(CNode(nxt,'add' if sign>0 else 'subtract',(cur,nid),{}));cur=nxt
        rate_nodes.append(cur)
        out=f'{s.name}__next';nodes.append(CNode(out,'euler_state_update',(s.name,cur),{'dt':float(ir.dt),'dt_dim':TIME}))
    outs=tuple(f'{s.name}__next' for s in ir.states);expected=tuple((f'{s.name}__next',state_dims[s.name]) for s in ir.states)
    cir=CanonicalIR(inputs,tuple(nodes),outs,expected,(('source_ir','e16_component'),('dt',repr(float(ir.dt)))))
    return cir,tuple(s.name for s in ir.states),tuple(rate_nodes)

def compile_e16_via_canonical(ir):
    cir,names,rate_nodes=lower_e16(ir);g=compile_canonical(cir);return CanonicalStateWrapper(g,names,rate_nodes),cir
