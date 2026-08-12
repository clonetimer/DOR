from __future__ import annotations
import numpy as np
from ..contracts import PublicDataset
from ..phm import NominalMechanism, _design, _ridge


def estimate_physical_parameter(system: str, nominal: NominalMechanism, ds: PublicDataset) -> np.ndarray:
    out=[]
    for k in range(len(ds.X)):
        local={}
        for target,sources in nominal.graph.items():
            P,y=_design(ds,k,target,sources,nominal.state_names,nominal.input_names,nominal.operators,nominal.operator_parameters)
            th=_ridge(P,y); local[target]={s:float(v) for s,v in zip(sources,th)}
        try:
            if system=="thermal": C=1.0/local["T"]["P"]; val=-local["T"]["T"]*C
            elif system=="tank": val=1.0/local["h"]["qin"]
            elif system=="motor": J=-1.0/local["w"]["TL"]; val=-local["w"]["w"]*J
            elif system=="msd": m=1.0/local["v"]["F"]; val=-local["v"]["x"]*m
            else: raise KeyError(system)
        except (KeyError,ZeroDivisionError,FloatingPointError): val=np.nan
        out.append(val)
    return np.asarray(out,float)


def nominal_physical_parameter(system: str, nominal: NominalMechanism) -> float:
    c=nominal.coefficients
    if system=="thermal": C=1.0/c["T"]["P"]; return -c["T"]["T"]*C
    if system=="tank": return 1.0/c["h"]["qin"]
    if system=="motor": J=-1.0/c["w"]["TL"]; return -c["w"]["w"]*J
    if system=="msd": m=1.0/c["v"]["F"]; return -c["v"]["x"]*m
    raise KeyError(system)
