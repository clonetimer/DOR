from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import math
import torch
from torch import nn
import torch.nn.functional as F

from .canonical_ir import Dim


def _inv_softplus(x: float) -> float:
    x=max(float(x),1e-8)
    return math.log(math.expm1(x)) if x < 50.0 else x


@dataclass(frozen=True)
class PortSpec:
    name: str
    dim: Dim


@dataclass(frozen=True)
class StableModalLTISpec:
    """Stable scalar-input/scalar-output low-order LTI block.

    ``order=3`` is represented as one stable real mode in parallel with one
    stable second-order mode.  This covers the common third-order case with one
    real pole and one real/complex pole pair while keeping stability structural:
    ``decay>0``, ``wn>0`` and ``zeta>0``.

    All modal states carry the input port dimension; signed output coefficients
    map those states to the declared output port dimension.  The representation
    is intentionally modal rather than device-specific.
    """
    id: str
    input_dim: Dim
    output_dim: Dim
    order: int = 3
    decay_init: float = 1.0
    wn_init: float = 2.0
    zeta_init: float = 0.7
    output_coeff_init: tuple[float, ...] = (1.0, 0.0, 0.0)


@dataclass(frozen=True)
class StaticLawSpec:
    """Local memoryless constitutive slot between dynamic blocks.

    Supported finite families are intentionally small.  ``monotone_local_neural``
    is a local K3 escape with a positive derivative by construction; it is not a
    generic network over the full trajectory.
    """
    id: str
    input_dim: Dim
    output_dim: Dim
    law: str  # affine | odd_cubic | piecewise_linear | monotone_piecewise3 | smooth_diode | monotone_local_neural
    init: tuple[float, ...] = ()
    hidden: int = 1


@dataclass(frozen=True)
class BlockLNLIR:
    input: PortSpec
    output: PortSpec
    first: StableModalLTISpec
    static: StaticLawSpec
    second: StableModalLTISpec
    dt: float
    metadata: tuple[tuple[str,str], ...] = ()


def validate_block_lnl_ir(ir: BlockLNLIR) -> bool:
    if not (ir.dt > 0):
        raise ValueError('dt must be positive')
    if ir.input.name == ir.output.name:
        raise ValueError('input/output names must differ')
    for b in (ir.first, ir.second):
        if b.order not in (1,3):
            raise ValueError(f'{b.id}: supported stable modal orders are 1 and 3')
        if b.decay_init <= 0:
            raise ValueError(f'{b.id}: decay_init must be positive')
        if b.order == 3 and (b.wn_init <= 0 or b.zeta_init <= 0):
            raise ValueError(f'{b.id}: wn/zeta must be positive')
        if len(b.output_coeff_init) != b.order:
            raise ValueError(f'{b.id}: output coefficient count != order')
    if ir.first.input_dim != ir.input.dim:
        raise ValueError('first LTI input dimension mismatch')
    if ir.first.output_dim != ir.static.input_dim:
        raise ValueError('first LTI -> static dimension mismatch')
    if ir.static.output_dim != ir.second.input_dim:
        raise ValueError('static -> second LTI dimension mismatch')
    if ir.second.output_dim != ir.output.dim:
        raise ValueError('second LTI output dimension mismatch')
    allowed={'affine','odd_cubic','piecewise_linear','monotone_piecewise3','smooth_diode','monotone_local_neural'}
    if ir.static.law not in allowed:
        raise ValueError(f'unsupported static law {ir.static.law}')
    if ir.static.law=='monotone_local_neural' and ir.static.hidden < 1:
        raise ValueError('monotone local neural hidden must be >=1')
    return True


class StableModalLTI(nn.Module):
    """Exact-ZOH stable modal LTI block with learnable modal parameters."""
    def __init__(self,spec:StableModalLTISpec,dt:float):
        super().__init__(); self.spec=spec; self.dt=float(dt); self.order=spec.order
        self.raw_decay=nn.Parameter(torch.tensor(_inv_softplus(spec.decay_init),dtype=torch.get_default_dtype()))
        if self.order==3:
            self.raw_wn=nn.Parameter(torch.tensor(_inv_softplus(spec.wn_init),dtype=torch.get_default_dtype()))
            self.raw_zeta=nn.Parameter(torch.tensor(_inv_softplus(spec.zeta_init),dtype=torch.get_default_dtype()))
        else:
            self.register_parameter('raw_wn',None); self.register_parameter('raw_zeta',None)
        self.output_coeff=nn.Parameter(torch.tensor(spec.output_coeff_init,dtype=torch.get_default_dtype()))

    @property
    def decay(self): return F.softplus(self.raw_decay)+1e-8
    @property
    def wn(self):
        if self.order!=3: raise AttributeError('wn exists only for order=3')
        return F.softplus(self.raw_wn)+1e-8
    @property
    def zeta(self):
        if self.order!=3: raise AttributeError('zeta exists only for order=3')
        return F.softplus(self.raw_zeta)+1e-8

    def continuous_matrices(self):
        a=self.decay; z=a*0.0
        if self.order==1:
            A=(-a).reshape(1,1); B=a.reshape(1,1)
            return A,B
        w=self.wn; ze=self.zeta
        A=torch.stack((
            torch.stack((-a,z,z)),
            torch.stack((z,z,w)),
            torch.stack((z,-w,-2.0*ze*w)),
        ))
        B=torch.stack((a,z,w)).reshape(3,1)
        return A,B

    def discrete_matrices(self):
        A,B=self.continuous_matrices(); n=self.order
        top=torch.cat((A,B),dim=1)
        bottom=torch.zeros((1,n+1),dtype=A.dtype,device=A.device)
        aug=torch.cat((top,bottom),dim=0)
        E=torch.matrix_exp(aug*self.dt)
        return E[:n,:n],E[:n,n]

    def poles(self):
        A,_=self.continuous_matrices()
        return torch.linalg.eigvals(A)

    def filter(self,u:torch.Tensor,state0:torch.Tensor|None=None,*,return_state:bool=False):
        if u.ndim!=1: raise ValueError('StableModalLTI expects a 1-D scalar sequence')
        Ad,Bd=self.discrete_matrices(); n=self.order
        x=torch.zeros(n,dtype=u.dtype,device=u.device) if state0 is None else state0.to(dtype=u.dtype,device=u.device)
        if x.shape!=(n,): raise ValueError('state0 shape mismatch')
        ys=[]; states=[]
        C=self.output_coeff.to(dtype=u.dtype,device=u.device)
        for ut in u:
            x=Ad@x+Bd*ut
            ys.append(torch.dot(C,x)); states.append(x)
        y=torch.stack(ys) if ys else torch.empty_like(u)
        if return_state:
            st=torch.stack(states) if states else torch.empty((0,n),dtype=u.dtype,device=u.device)
            return y,st
        return y

    def impulse_response(self,length:int=512):
        if int(length)<1: raise ValueError('impulse length must be positive')
        Ad,Bd=self.discrete_matrices(); x=torch.zeros(self.order,dtype=Ad.dtype,device=Ad.device)
        C=self.output_coeff.to(dtype=Ad.dtype,device=Ad.device); ys=[]
        for k in range(int(length)):
            x=Ad@x+Bd*(1.0 if k==0 else 0.0)
            ys.append(torch.dot(C,x))
        return torch.stack(ys)

    def impulse_tail_ratio(self,length:int=512,tail_fraction:float=.125):
        h=self.impulse_response(length); n=max(1,int(len(h)*float(tail_fraction)))
        return torch.linalg.vector_norm(h[-n:])/(torch.linalg.vector_norm(h)+1e-15)

    def filter_fft(self,u:torch.Tensor,impulse_length:int=512):
        """Differentiable zero-state causal rollout via truncated impulse response.

        This path is intended for long records.  Callers must audit
        ``impulse_tail_ratio`` and reject an impulse length whose tail remains
        material.  It therefore accelerates execution without silently changing
        the dynamic contract.
        """
        if u.ndim!=1: raise ValueError('StableModalLTI expects a 1-D scalar sequence')
        h=self.impulse_response(int(impulse_length)).to(dtype=u.dtype,device=u.device)
        n=int(u.numel()+h.numel()-1); nfft=1 << (n-1).bit_length()
        y=torch.fft.irfft(torch.fft.rfft(u,nfft)*torch.fft.rfft(h,nfft),nfft)
        return y[:u.numel()]


class AffineStatic(nn.Module):
    def __init__(self,init:Sequence[float]=()):
        super().__init__(); vals=list(init) if init else [1.0,0.0]
        if len(vals)!=2: raise ValueError('affine init requires (gain,bias)')
        self.gain=nn.Parameter(torch.tensor(float(vals[0]))); self.bias=nn.Parameter(torch.tensor(float(vals[1])))
    def forward(self,x): return self.gain*x+self.bias


class OddCubicStatic(nn.Module):
    def __init__(self,init:Sequence[float]=()):
        super().__init__(); vals=list(init) if init else [1.0,0.1]
        if len(vals)!=2: raise ValueError('odd_cubic init requires (a1,a3)')
        self.a1=nn.Parameter(torch.tensor(float(vals[0]))); self.a3=nn.Parameter(torch.tensor(float(vals[1])))
    def forward(self,x): return self.a1*x+self.a3*x**3


class PiecewiseLinearStatic(nn.Module):
    def __init__(self,init:Sequence[float]=()):
        super().__init__(); vals=list(init) if init else [0.0,1.0,0.5,0.0]
        if len(vals)!=4: raise ValueError('piecewise init requires (bias,slope,delta_slope,threshold)')
        self.bias=nn.Parameter(torch.tensor(float(vals[0]))); self.slope=nn.Parameter(torch.tensor(float(vals[1])))
        self.delta_slope=nn.Parameter(torch.tensor(float(vals[2]))); self.threshold=nn.Parameter(torch.tensor(float(vals[3])))
    def forward(self,x): return self.bias+self.slope*x+self.delta_slope*F.relu(x-self.threshold)


class MonotonePiecewise3Static(nn.Module):
    """Eight-parameter finite K2 monotone four-segment law.

    Parameters are one bias, one positive base slope, three positive slope
    increments and three learned knee locations.  This is deliberately a
    finite engineering grammar member rather than a neural hidden layer.
    """
    def __init__(self,init:Sequence[float]=()):
        super().__init__()
        if init:
            vals=list(init)
            if len(vals)!=8: raise ValueError('monotone_piecewise3 init requires 8 values')
            bias,base,*rest=vals; weights=rest[:3]; knots=rest[3:]
        else:
            bias,base=0.0,0.5; weights=[0.25,0.25,0.25]; knots=[-0.5,0.0,0.5]
        self.bias=nn.Parameter(torch.tensor(float(bias)))
        self.raw_base=nn.Parameter(torch.tensor(_inv_softplus(float(base))))
        self.raw_weight=nn.Parameter(torch.tensor([_inv_softplus(float(x)) for x in weights]))
        self.knot=nn.Parameter(torch.tensor([float(x) for x in knots]))
    def forward(self,x):
        base=F.softplus(self.raw_base)+1e-8; w=F.softplus(self.raw_weight)+1e-8
        return self.bias+base*x+torch.sum(w[None,:]*F.relu(x[:,None]-self.knot[None,:]),dim=1)


class SmoothDiodeStatic(nn.Module):
    """Finite K2 one-sided smooth-knee constitutive family."""
    def __init__(self,init:Sequence[float]=()):
        super().__init__(); vals=list(init) if init else [0.0,0.5,0.5,3.0,0.0]
        if len(vals)!=5: raise ValueError('smooth_diode init requires (bias,base_gain,diode_gain,sharpness,threshold)')
        self.bias=nn.Parameter(torch.tensor(float(vals[0]))); self.base_gain=nn.Parameter(torch.tensor(float(vals[1])))
        self.raw_diode_gain=nn.Parameter(torch.tensor(_inv_softplus(float(vals[2]))))
        self.raw_sharp=nn.Parameter(torch.tensor(_inv_softplus(float(vals[3])))); self.threshold=nn.Parameter(torch.tensor(float(vals[4])))
    def forward(self,x):
        dg=F.softplus(self.raw_diode_gain)+1e-8; k=F.softplus(self.raw_sharp)+1e-8
        return self.bias+self.base_gain*x+dg*F.softplus(k*(x-self.threshold))/k


class MonotoneLocalStaticNN(nn.Module):
    """K3 local static law with positive derivative by construction.

    For ``hidden=1`` the module has five trainable parameters, matching the
    finite ``smooth_diode`` K2 family.  More hidden knees can be requested only
    when the experiment explicitly permits additional local capacity.
    """
    def __init__(self,hidden:int=1):
        super().__init__(); h=int(hidden)
        if h<1: raise ValueError('hidden must be >=1')
        self.bias=nn.Parameter(torch.tensor(0.0)); self.raw_base=nn.Parameter(torch.tensor(_inv_softplus(0.5)))
        self.raw_weight=nn.Parameter(torch.full((h,),_inv_softplus(0.5)))
        self.raw_slope=nn.Parameter(torch.full((h,),_inv_softplus(2.0)))
        self.knot=nn.Parameter(torch.linspace(-0.25,0.25,h))
    def forward(self,x):
        base=F.softplus(self.raw_base)+1e-8
        w=F.softplus(self.raw_weight)+1e-8; s=F.softplus(self.raw_slope)+1e-8
        z=x[:,None]-self.knot[None,:]
        knees=F.softplus(s[None,:]*z)/s[None,:]
        return self.bias+base*x+torch.sum(w[None,:]*knees,dim=1)


def compile_static_law(spec:StaticLawSpec)->nn.Module:
    if spec.law=='affine': return AffineStatic(spec.init)
    if spec.law=='odd_cubic': return OddCubicStatic(spec.init)
    if spec.law=='piecewise_linear': return PiecewiseLinearStatic(spec.init)
    if spec.law=='monotone_piecewise3': return MonotonePiecewise3Static(spec.init)
    if spec.law=='smooth_diode': return SmoothDiodeStatic(spec.init)
    if spec.law=='monotone_local_neural': return MonotoneLocalStaticNN(spec.hidden)
    raise ValueError(spec.law)


class BlockLNLModel(nn.Module):
    """Compiled LTI -> local static law -> LTI computation graph."""
    def __init__(self,ir:BlockLNLIR):
        super().__init__(); validate_block_lnl_ir(ir); self.ir=ir
        self.first=StableModalLTI(ir.first,ir.dt); self.static=compile_static_law(ir.static); self.second=StableModalLTI(ir.second,ir.dt)

    def forward(self,u:torch.Tensor,*,state1:torch.Tensor|None=None,state2:torch.Tensor|None=None,return_internal:bool=False):
        z1,s1=self.first.filter(u,state1,return_state=True)
        z2=self.static(z1)
        y,s2=self.second.filter(z2,state2,return_state=True)
        if return_internal:
            return {'y':y,'first_output':z1,'static_output':z2,'first_state':s1,'second_state':s2}
        return y

    def forward_fast(self,u:torch.Tensor,*,impulse_length:int=512,return_internal:bool=False):
        z1=self.first.filter_fft(u,impulse_length); z2=self.static(z1); y=self.second.filter_fft(z2,impulse_length)
        if return_internal:
            return {'y':y,'first_output':z1,'static_output':z2,
                    'first_tail_ratio':self.first.impulse_tail_ratio(impulse_length),
                    'second_tail_ratio':self.second.impulse_tail_ratio(impulse_length)}
        return y

    def pole_real_parts(self):
        return torch.cat((self.first.poles().real,self.second.poles().real))


def compile_block_lnl_ir(ir:BlockLNLIR):
    validate_block_lnl_ir(ir)
    return BlockLNLModel(ir)


def block_architecture_summary(ir:BlockLNLIR,model:BlockLNLModel)->dict:
    return {
        'source_ir':'block_lnl',
        'topology':['stable_lti',ir.static.law,'stable_lti'],
        'orders':[ir.first.order,ir.second.order],
        'dt':float(ir.dt),
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'static_parameters':sum(p.numel() for p in model.static.parameters() if p.requires_grad),
        'metadata':dict(ir.metadata),
    }
