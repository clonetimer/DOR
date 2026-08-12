from __future__ import annotations
from dataclasses import dataclass, asdict
from importlib import import_module
from typing import Any, Callable, Dict, Optional

@dataclass(frozen=True)
class AdapterDescriptor:
    name: str
    module: str
    factory: str
    description: str = ""
    kind: str = "dataset"

_REGISTRY: Dict[str,AdapterDescriptor] = {}

def register_adapter(name: str, module: str, factory: str, *, description: str="", kind: str="dataset") -> None:
    _REGISTRY[str(name)] = AdapterDescriptor(str(name),str(module),str(factory),str(description),str(kind))

def get_adapter(name: str) -> Callable[...,Any]:
    try: d=_REGISTRY[str(name)]
    except KeyError as e: raise KeyError(f"unknown adapter {name!r}; available={sorted(_REGISTRY)}") from e
    obj=getattr(import_module(d.module), d.factory)
    return obj

def available_adapters() -> tuple[str,...]: return tuple(sorted(_REGISTRY))

def adapter_manifest() -> Dict[str,Dict[str,Any]]: return {k:asdict(v) for k,v in sorted(_REGISTRY.items())}

# Built-ins are registered by module path only; importing this registry does not import device code.
register_adapter("ehss", "isgn.bridges.ehss", "load_ehss_scenarios", description="EHSS diagnosis dataset bridge")
register_adapter("uci-hydraulic", "isgn.bridges.uci_hydraulic", "load_uci_hydraulic", description="UCI hydraulic condition-monitoring bridge")
register_adapter("euv", "isgn.bridges.euv", "load_euv_scope", description="EUV simulator clean-room bridge")
