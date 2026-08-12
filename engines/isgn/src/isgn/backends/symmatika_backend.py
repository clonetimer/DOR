"""External SymMatika relation-discovery adapter.

Scope is intentionally narrow: algebraic/constitutive relations that remain
GRAMMAR_UNCOVERED after system-graph completion. It does not replace the ISGN
state-equation or certification layers.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Sequence
import json, os, shutil, subprocess, tempfile

@dataclass
class SymbolicCandidate:
    expression: str
    error: float

@dataclass
class SymbolicDiscoveryResult:
    backend: str
    candidates: List[SymbolicCandidate]
    task_csv: str
    metadata: dict
    def to_dict(self):
        d=asdict(self); return d

class SymMatikaExternalBackend:
    name='symmatika'
    def __init__(self, executable: Optional[str]=None):
        self.executable=executable or os.environ.get('ISGN_SYMMATIKA_BRIDGE') or shutil.which('isgn_symmatika_bridge')
    @classmethod
    def available(cls):
        exe=os.environ.get('ISGN_SYMMATIKA_BRIDGE') or shutil.which('isgn_symmatika_bridge')
        return bool(exe and Path(exe).exists())
    def run_explicit(self,csv_path:str|Path,variables:Sequence[str],*,population:int=10000,depth:int=3,timeout:int=3600):
        if not self.executable:
            raise RuntimeError('SymMatika bridge unavailable. Build external/symmatika/isgn_symmatika_bridge.cpp and set ISGN_SYMMATIKA_BRIDGE=/path/to/isgn_symmatika_bridge')
        cmd=[str(self.executable),str(csv_path),'1',','.join(variables),str(int(population)),str(int(depth))]
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout,check=True)
        try: payload=json.loads(p.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(f'SymMatika bridge did not return JSON. stderr={p.stderr[-1000:]} stdout={p.stdout[-1000:]}') from e
        c=[SymbolicCandidate(str(x['expression']),float(x['error'])) for x in payload.get('candidates',[])]
        return SymbolicDiscoveryResult('symmatika',c,str(csv_path),{'population':population,'depth':depth,'stderr':p.stderr[-4000:]})
