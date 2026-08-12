from __future__ import annotations
import importlib
import importlib.metadata as md

COMPONENTS={"DOR":"dor-diagnostic-stack","ISGN":"isgn-toolkit","M2N":"m2n-toolkit","FEC":"fec-toolkit"}

def stack_info():
    out={}
    for name,dist in COMPONENTS.items():
        try: out[name]={"distribution":dist,"version":md.version(dist),"installed":True}
        except md.PackageNotFoundError: out[name]={"distribution":dist,"version":None,"installed":False}
    return out
