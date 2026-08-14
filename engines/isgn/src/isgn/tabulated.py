from __future__ import annotations

"""Bounded empirical one-dimensional constitutive relations.

``Tabulated1D`` is intentionally narrow.  It is for a mechanism scope where
engineering knowledge has already identified a scalar constitutive coordinate
``x`` and scalar response ``y``, but the shape y=f(x) is not known analytically.
The table is evidence-backed only inside the observed x support and inside the
same declared regime.  It never extrapolates and it refuses multivalued samples
instead of averaging away hysteresis/mode mixing.

This is not a generic predictive fallback and it does not discover graph
structure.  It completes one already-declared local constitutive relation.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, Mapping, Sequence
import numpy as np
from scipy.interpolate import PchipInterpolator

from .contracts import AlgebraicDataset


def _column(ds: AlgebraicDataset, name: str) -> np.ndarray:
    if name not in ds.variable_names:
        raise KeyError(name)
    return np.asarray(ds.values[:, ds.variable_names.index(name)], dtype=float)


@dataclass
class Tabulated1DResult:
    input: str
    target: str
    method: str
    x_knots: list[float]
    y_knots: list[float]
    support_min: float
    support_max: float
    regime: str
    validation_rmse: float
    validation_nrmse: float
    validation_max_abs: float
    validation_rows: int
    candidate_only: bool = True
    certification_allowed: bool = False
    truth_channels_used: bool = False
    status: str = "PASS"
    metadata: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Tabulated1DResult":
        return cls(**dict(payload))

    def support_mask(self, x: Sequence[float] | np.ndarray) -> np.ndarray:
        z=np.asarray(x,dtype=float)
        return np.isfinite(z) & (z >= float(self.support_min)) & (z <= float(self.support_max))

    def predict(self, x: Sequence[float] | np.ndarray, *, strict: bool=True) -> np.ndarray:
        z=np.asarray(x,dtype=float)
        mask=self.support_mask(z)
        if strict and not bool(np.all(mask)):
            bad=z[~mask]
            lo=float(np.nanmin(bad)) if len(bad) else float('nan')
            hi=float(np.nanmax(bad)) if len(bad) else float('nan')
            raise ValueError(
                f"OUT_OF_SUPPORT: Tabulated1D support=[{self.support_min},{self.support_max}], "
                f"requested=[{lo},{hi}]"
            )
        out=np.full(z.shape,np.nan,dtype=float)
        if not np.any(mask):
            return out
        xk=np.asarray(self.x_knots,dtype=float); yk=np.asarray(self.y_knots,dtype=float)
        if self.method == "linear":
            out[mask]=np.interp(z[mask],xk,yk)
        elif self.method == "pchip":
            out[mask]=PchipInterpolator(xk,yk,extrapolate=False)(z[mask])
        else:  # defensive for corrupted serialized payloads
            raise ValueError(f"unsupported Tabulated1D method {self.method!r}")
        return out

    def predict_dataset(self, ds: AlgebraicDataset, *, strict: bool=True) -> np.ndarray:
        ds.validate()
        if str(ds.regime) != str(self.regime):
            raise ValueError(f"CONTEXT_MISMATCH: fitted regime={self.regime!r}, requested regime={ds.regime!r}")
        return self.predict(_column(ds,self.input),strict=strict)


def _prepare_knots(ds: AlgebraicDataset, *, input: str, target: str,
                   duplicate_atol: float, min_unique_points: int) -> tuple[np.ndarray,np.ndarray,Dict[str,Any]]:
    ds.validate()
    x=_column(ds,input); y=_column(ds,target)
    good=np.isfinite(x)&np.isfinite(y)
    x=x[good]; y=y[good]
    if len(x) < int(min_unique_points):
        raise ValueError("INSUFFICIENT_SUPPORT: too few finite constitutive samples")
    order=np.argsort(x,kind="mergesort"); x=x[order]; y=y[order]
    unique,starts,counts=np.unique(x,return_index=True,return_counts=True)
    yk=[]; conflict=[]; max_spread=0.0
    for u,s,n in zip(unique,starts,counts):
        vals=y[s:s+n]
        spread=float(np.ptp(vals)) if len(vals)>1 else 0.0
        max_spread=max(max_spread,spread)
        if spread > float(duplicate_atol):
            conflict.append({'x':float(u),'count':int(n),'spread':spread})
        yk.append(float(np.mean(vals)))
    if conflict:
        first=conflict[0]
        raise ValueError(
            "MULTIVALUED_RELATION: repeated constitutive input has conflicting outputs; "
            f"first x={first['x']} spread={first['spread']} > duplicate_atol={duplicate_atol}. "
            "Partition hysteresis/regime/context before fitting a 1-D relation."
        )
    if len(unique) < int(min_unique_points):
        raise ValueError("INSUFFICIENT_SUPPORT: too few unique constitutive input values")
    if not np.all(np.diff(unique)>0):
        raise ValueError("invalid constitutive support: x knots must be strictly increasing")
    return np.asarray(unique,float),np.asarray(yk,float),{
        'finite_rows':int(len(x)), 'unique_points':int(len(unique)),
        'duplicate_groups':int(np.sum(counts>1)), 'max_duplicate_spread':float(max_spread),
    }


def fit_tabulated_1d(train: AlgebraicDataset, validation: AlgebraicDataset, *,
                     input: str, target: str, method: str="pchip",
                     duplicate_atol: float=1e-8, min_unique_points: int=6) -> Tabulated1DResult:
    """Fit a bounded 1-D empirical constitutive relation and validate in-scope.

    The train and validation datasets must declare the same ``regime``.  Every
    validation input must lie inside the training support.  This makes
    interpolation evidence explicit and prevents a lookup table from silently
    becoming an extrapolating black box.
    """
    method=str(method).lower()
    if method not in {"linear","pchip"}:
        raise ValueError("method must be 'linear' or 'pchip'")
    train.validate(); validation.validate()
    if str(train.regime) != str(validation.regime):
        raise ValueError(f"CONTEXT_MISMATCH: train regime={train.regime!r}, validation regime={validation.regime!r}")
    if input == target:
        raise ValueError("input and target must be different variables")
    xk,yk,prep=_prepare_knots(train,input=input,target=target,duplicate_atol=duplicate_atol,min_unique_points=min_unique_points)
    vx=_column(validation,input); vy=_column(validation,target)
    good=np.isfinite(vx)&np.isfinite(vy); vx=vx[good]; vy=vy[good]
    if len(vx)<3:
        raise ValueError("INSUFFICIENT_SUPPORT: too few finite validation samples")
    support=(vx>=xk[0])&(vx<=xk[-1])
    if not np.all(support):
        raise ValueError(
            f"OUT_OF_SUPPORT: validation contains {int(np.sum(~support))}/{len(vx)} rows outside "
            f"training support [{xk[0]},{xk[-1]}]"
        )
    if method == 'linear': pred=np.interp(vx,xk,yk)
    else: pred=PchipInterpolator(xk,yk,extrapolate=False)(vx)
    err=np.asarray(pred)-vy
    rmse=float(np.sqrt(np.mean(err*err)))
    nrmse=float(rmse/(np.std(vy)+1e-12))
    mx=float(np.max(np.abs(err)))
    return Tabulated1DResult(
        input=str(input),target=str(target),method=method,
        x_knots=xk.tolist(),y_knots=yk.tolist(),support_min=float(xk[0]),support_max=float(xk[-1]),
        regime=str(train.regime),validation_rmse=rmse,validation_nrmse=nrmse,validation_max_abs=mx,
        validation_rows=int(len(vx)),metadata={**prep,'duplicate_atol':float(duplicate_atol),
            'min_unique_points':int(min_unique_points),'train_variant':str(train.variant),
            'validation_variant':str(validation.variant),'scope':'declared-1d-constitutive'}
    )
