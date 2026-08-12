from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
import hashlib, json, math
import torch
from torch import nn
import torch.nn.functional as F
from .canonical_ir import Dim


@dataclass(frozen=True)
class PeakAxisSpec:
    """One dimensionless coordinate in a separable peaked constitutive law.

    ``scale`` converts the physical input into a dimensionless quantity before
    applying ``transform``.  This keeps optimization well-conditioned without
    changing the physical topology.  ``center``/``width`` live in transformed,
    dimensionless coordinates.
    """
    name: str
    dim: Dim
    transform: str = 'identity'   # identity | log
    scale: float = 1.0
    center_init: float = 0.0
    width_init: float = 1.0
    learn_center: bool = True

    def validate(self) -> 'PeakAxisSpec':
        if not self.name:
            raise ValueError('axis name must be non-empty')
        if self.transform not in {'identity','log'}:
            raise ValueError(f'{self.name}: unsupported transform {self.transform!r}')
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError(f'{self.name}: scale must be finite positive')
        if not math.isfinite(self.center_init):
            raise ValueError(f'{self.name}: center_init must be finite')
        if not math.isfinite(self.width_init) or self.width_init <= 0:
            raise ValueError(f'{self.name}: width_init must be finite positive')
        return self


@dataclass(frozen=True)
class SeparablePeakIR:
    """Generic multivariate peaked response surface.

    y = A * exp[-1/2 * sum_j ((z_j-c_j)/w_j)^2]

    where ``z_j`` is either x/scale or log(x/scale).  This is a generic local
    constitutive primitive for bounded response surfaces (conversion efficiency,
    reaction yield, resonance envelope, etc.); it contains no EUV/device names.
    The finite architecture family is represented by ``active_axes``.
    """
    axes: tuple[PeakAxisSpec, ...]
    active_axes: tuple[str, ...]
    output_name: str
    output_dim: Dim
    amplitude_init: float = 1.0
    metadata: tuple[tuple[str,str], ...] = ()

    def validate(self) -> 'SeparablePeakIR':
        if not self.axes:
            raise ValueError('at least one axis is required')
        for a in self.axes:
            a.validate()
        names=[a.name for a in self.axes]
        if len(names)!=len(set(names)):
            raise ValueError('duplicate peak axis names')
        if not self.active_axes:
            raise ValueError('active_axes must be non-empty')
        if len(self.active_axes)!=len(set(self.active_axes)):
            raise ValueError('duplicate active axes')
        missing=set(self.active_axes)-set(names)
        if missing:
            raise ValueError(f'unknown active axes: {sorted(missing)}')
        if not self.output_name:
            raise ValueError('output_name must be non-empty')
        if not math.isfinite(self.amplitude_init) or self.amplitude_init <= 0:
            raise ValueError('amplitude_init must be finite positive')
        return self


class SeparablePeakGraph(nn.Module):
    def __init__(self, ir: SeparablePeakIR):
        super().__init__(); ir.validate(); self.ir=ir
        self.raw_amplitude=nn.Parameter(self._raw_pos(ir.amplitude_init))
        self.centers=nn.ParameterDict(); self.raw_widths=nn.ParameterDict()
        self._fixed_centers={}
        amap={a.name:a for a in ir.axes}
        for name in ir.active_axes:
            a=amap[name]
            if a.learn_center:
                self.centers[name]=nn.Parameter(torch.tensor(float(a.center_init),dtype=torch.float64))
            else:
                self._fixed_centers[name]=float(a.center_init)
            self.raw_widths[name]=nn.Parameter(self._raw_pos(a.width_init))

    @staticmethod
    def _raw_pos(x: float) -> torch.Tensor:
        x=max(float(x),1e-12)
        return torch.log(torch.expm1(torch.tensor(x,dtype=torch.float64)))

    @staticmethod
    def _pos(raw: torch.Tensor) -> torch.Tensor:
        return F.softplus(raw)+1e-12

    def amplitude(self): return self._pos(self.raw_amplitude)
    def width(self,name): return self._pos(self.raw_widths[name])
    def center(self,name):
        return self.centers[name] if name in self.centers else torch.as_tensor(self._fixed_centers[name],dtype=self.raw_amplitude.dtype,device=self.raw_amplitude.device)

    def _z(self,a:PeakAxisSpec,x:torch.Tensor):
        z=x/float(a.scale)
        if a.transform=='log':
            z=torch.log(torch.clamp(z,min=1e-30))
        return z

    def forward(self, **inputs):
        amap={a.name:a for a in self.ir.axes}
        missing=[n for n in self.ir.active_axes if n not in inputs]
        if missing: raise ValueError(f'missing peak inputs: {missing}')
        q=None
        for name in self.ir.active_axes:
            z=self._z(amap[name],inputs[name])
            term=torch.square((z-self.center(name))/self.width(name))
            q=term if q is None else q+term
        y=self.amplitude()*torch.exp(-0.5*q)
        return {self.ir.output_name:y}

    def set_normalized_parameters(self, *, amplitude: float, centers: dict[str,float], widths: dict[str,float]) -> None:
        """Load fitted parameters expressed in the IR's normalized coordinates."""
        with torch.no_grad():
            self.raw_amplitude.copy_(self._raw_pos(float(amplitude)).to(self.raw_amplitude.device))
            for name in self.ir.active_axes:
                if name in self.centers:
                    self.centers[name].copy_(torch.as_tensor(float(centers[name]),dtype=self.centers[name].dtype,device=self.centers[name].device))
                self.raw_widths[name].copy_(self._raw_pos(float(widths[name])).to(self.raw_widths[name].device))

    def physical_parameters(self) -> dict[str,float]:
        out={'amplitude':float(self.amplitude().detach())}
        amap={a.name:a for a in self.ir.axes}
        for name in self.ir.active_axes:
            a=amap[name]; c=float(self.center(name).detach()); w=float(self.width(name).detach())
            if a.transform=='log':
                out[f'{name}_center_physical']=float(a.scale*math.exp(c))
                out[f'{name}_log_width']=w
            else:
                out[f'{name}_center_physical']=float(a.scale*c)
                out[f'{name}_width_physical']=float(a.scale*w)
        return out


def compile_separable_peak_ir(ir:SeparablePeakIR)->SeparablePeakGraph:
    ir.validate(); return SeparablePeakGraph(ir)


def separable_peak_signature(ir:SeparablePeakIR)->str:
    ir.validate()
    obj=asdict(ir)
    def conv(x:Any):
        if isinstance(x,Dim): return x.serial()
        if isinstance(x,dict): return {k:conv(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)): return [conv(v) for v in x]
        return x
    s=json.dumps(conv(obj),sort_keys=True,separators=(',',':'))
    return hashlib.sha256(s.encode()).hexdigest()
