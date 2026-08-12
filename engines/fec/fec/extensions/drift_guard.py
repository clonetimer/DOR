from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Iterable
import math
import numpy as np

@dataclass(frozen=True)
class CalibrationHealth:
    n: int
    alpha: float
    exceedances: int
    empirical_fpr: float
    ci_low: float
    ci_high: float
    status: str
    action: str


def _wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float,float]:
    if n <= 0:
        return (float('nan'), float('nan'))
    p = k / n
    den = 1.0 + z*z/n
    cen = (p + z*z/(2*n))/den
    half = z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/den
    return max(0.0, cen-half), min(1.0, cen+half)


def audit_p_values(
    p_values: Iterable[float], *, alpha: float = 0.05,
    min_samples: int = 40, tolerance: float = 0.0,
) -> CalibrationHealth:
    """Audit false-alarm calibration on an externally trusted normal sentinel batch.

    This function does not infer whether arbitrary online samples are healthy.
    The caller must provide a batch whose normal/healthy provenance is trusted.
    It reports anti-conservative, conservative, healthy, or insufficient-data.
    """
    p = np.asarray(list(p_values), dtype=float)
    p = p[np.isfinite(p)]
    if not (0 < alpha < 1):
        raise ValueError('alpha must be in (0,1)')
    if min_samples < 1:
        raise ValueError('min_samples must be >=1')
    if len(p) < min_samples:
        return CalibrationHealth(len(p),alpha,int(np.sum(p<=alpha)),float(np.mean(p<=alpha)) if len(p) else float('nan'),float('nan'),float('nan'),'insufficient_data','collect_trusted_normal_sentinel')
    if np.any((p < 0) | (p > 1)):
        raise ValueError('p-values must be within [0,1]')
    k = int(np.sum(p <= alpha)); n=len(p); fpr=k/n; lo,hi=_wilson_interval(k,n)
    if lo > alpha + tolerance:
        status='anti_conservative'; action='block_or_refresh_calibration'
    elif hi < max(0.0, alpha - tolerance):
        status='conservative'; action='review_power_and_reference_specificity'
    else:
        status='healthy'; action='continue'
    return CalibrationHealth(n,alpha,k,float(fpr),float(lo),float(hi),status,action)


def calibration_strategy_decision(*, has_external_group: bool=False, groups_adequate: bool=False,
                                  has_shift_covariate: bool=False, ratio_validated: bool=False,
                                  has_trusted_refresh: bool=False, sentinel_health: str|None=None) -> dict:
    """Conservative function-first selector for optional calibration extensions."""
    if sentinel_health == 'anti_conservative':
        if has_trusted_refresh:
            return {'mode':'trusted_refresh','ready':True,'reason':'current reference is anti-conservative and a trusted refresh exists'}
        if has_external_group and groups_adequate:
            return {'mode':'mondrian','ready':True,'reason':'current reference is anti-conservative; external groups are available'}
        if has_shift_covariate and ratio_validated:
            return {'mode':'weighted_conformal','ready':True,'reason':'current reference is anti-conservative; validated shift model is available'}
        return {'mode':'unsupported','ready':False,'reason':'calibration is anti-conservative but no safe adaptation contract is available'}
    if has_external_group and groups_adequate:
        return {'mode':'mondrian','ready':True,'reason':'external grouping contract available'}
    if has_shift_covariate and ratio_validated:
        return {'mode':'weighted_conformal','ready':True,'reason':'validated covariate-shift contract available'}
    if has_trusted_refresh:
        return {'mode':'trusted_refresh','ready':True,'reason':'trusted refresh available'}
    return {'mode':'static','ready':True,'reason':'no verified shift adaptation contract; keep immutable reference and audit it'}


def audit_p_value_series(
    p_values: Iterable[float], *, alpha: float = 0.05,
    min_samples: int = 40, tolerance: float = 0.0,
    block_size: int = 1, n_bootstrap: int = 2000, random_state: int = 0,
) -> CalibrationHealth:
    """Calibration-health audit for a serial trusted-normal sentinel.

    With ``block_size=1`` this is identical to :func:`audit_p_values`.
    For ``block_size>1`` it uses a circular moving-block bootstrap on the
    false-alarm indicator ``1[p<=alpha]`` to avoid pretending overlapping or
    autocorrelated sentinel windows are independent Bernoulli trials.

    This is an operational diagnostic, not a replacement for a formal
    time-series conformal theorem. The block length must be declared from the
    sampling/windowing design or another defensible dependence analysis.
    """
    p = np.asarray(list(p_values), dtype=float)
    p = p[np.isfinite(p)]
    if block_size <= 1:
        return audit_p_values(p, alpha=alpha, min_samples=min_samples, tolerance=tolerance)
    if not (0 < alpha < 1):
        raise ValueError('alpha must be in (0,1)')
    if block_size < 1 or n_bootstrap < 100:
        raise ValueError('block_size must be >=1 and n_bootstrap >=100')
    if len(p) < min_samples:
        return CalibrationHealth(len(p),alpha,int(np.sum(p<=alpha)),float(np.mean(p<=alpha)) if len(p) else float('nan'),float('nan'),float('nan'),'insufficient_data','collect_trusted_normal_sentinel')
    if np.any((p < 0) | (p > 1)):
        raise ValueError('p-values must be within [0,1]')
    y = (p <= alpha).astype(float); n=len(y); k=int(y.sum()); fpr=float(y.mean())
    b=min(int(block_size),n); nblocks=int(np.ceil(n/b)); rng=np.random.default_rng(random_state)
    means=np.empty(int(n_bootstrap),float)
    idx=np.arange(n)
    for r in range(int(n_bootstrap)):
        starts=rng.integers(0,n,size=nblocks)
        boot=np.concatenate([y[(np.arange(s,s+b)%n)] for s in starts])[:n]
        means[r]=boot.mean()
    blo,bhi=np.quantile(means,[.025,.975])
    # Never let finite-sample bootstrap degeneracy (e.g. all-zero indicators)
    # create a narrower interval than the ordinary Wilson interval. For a
    # dependence diagnostic we prefer a conservative envelope.
    wlo,whi=_wilson_interval(k,n)
    lo=min(float(blo),float(wlo)); hi=max(float(bhi),float(whi))
    if lo > alpha + tolerance:
        status='anti_conservative';action='block_or_refresh_calibration'
    elif hi < max(0.0, alpha - tolerance):
        status='conservative';action='review_power_and_reference_specificity'
    else:
        status='healthy';action='continue'
    return CalibrationHealth(n,alpha,k,fpr,float(lo),float(hi),status,action)
