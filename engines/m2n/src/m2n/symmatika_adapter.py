from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
import csv
import json
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np

from .discovery import CandidateProposal, CandidateProposalBatch, BackendUnavailableError


SYMMATIKA_OPERATOR_ORDER = (
    'ADD','SUBTRACT','MULTIPLY','DIVIDE','POWER','SIN','SINH','ASIN','COS','COSH','ACOS','TAN',
    'TANH','ATAN','LOG','LN','EXP','SQRT'
)


@dataclass(frozen=True)
class SymMatikaExplicitRequest:
    """Explicit y=f(x) symbolic-discovery request for the upstream SymMatika source tree.

    M2N deliberately exposes only an explicit-mapping adapter in v1.3.  The
    upstream project also supports implicit relations, but those require a
    different data contract and are not silently mapped onto this interface.
    """
    x: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    target_name: str
    population_size: int = 10_000
    max_depth: int = 3
    allowed_ops: tuple[bool, ...] = (True,) * 18
    all_variables_relevant: bool = True
    multiple_distributions: bool = False

    def validate(self) -> 'SymMatikaExplicitRequest':
        x=np.asarray(self.x,dtype=float); y=np.asarray(self.y,dtype=float).reshape(-1)
        if x.ndim != 2 or x.shape[0] < 5:
            raise ValueError('x must have shape (samples, features) with at least 5 samples')
        if y.shape[0] != x.shape[0]:
            raise ValueError('y must share the sample axis with x')
        if len(self.feature_names) != x.shape[1]:
            raise ValueError('feature_names must match x columns')
        if not self.target_name.strip() or any(not s.strip() for s in self.feature_names):
            raise ValueError('feature/target names must be non-empty')
        if len(set(self.feature_names + (self.target_name,))) != len(self.feature_names)+1:
            raise ValueError('feature/target names must be unique')
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError('SymMatika request contains non-finite data')
        if self.population_size < 10:
            raise ValueError('population_size must be >=10')
        if not 1 <= self.max_depth <= 5:
            raise ValueError('max_depth must be in [1,5]')
        if len(self.allowed_ops) != 18:
            raise ValueError('allowed_ops must contain 18 booleans in upstream SymMatika operator order')
        return self


@dataclass(frozen=True)
class SymMatikaSourceProbe:
    source_root: str | None
    source_ready: bool
    cmake_available: bool
    cxx_available: bool
    reason: str

    @property
    def ready(self) -> bool:
        return self.source_ready and self.cmake_available and self.cxx_available

    def as_dict(self):
        return asdict(self) | {'ready': self.ready}


@dataclass(frozen=True)
class SymMatikaPreparedJob:
    job_dir: str
    dataset_csv: str
    driver_cpp: str
    contract_json: str
    sample_count: int
    warning: str

    def as_dict(self):
        return asdict(self)


def probe_symmatika_source(source_root: str | os.PathLike | None) -> SymMatikaSourceProbe:
    cmake=shutil.which('cmake') is not None
    cxx=any(shutil.which(x) is not None for x in ('clang++','g++','c++'))
    if source_root is None:
        return SymMatikaSourceProbe(None,False,cmake,cxx,'source_root not supplied; upstream SymMatika is a C++ source project, not a Python package')
    root=Path(source_root)
    required=('CMakeLists.txt','model.h','main.cpp')
    missing=[x for x in required if not (root/x).is_file()]
    if missing:
        return SymMatikaSourceProbe(str(root),False,cmake,cxx,'missing upstream files: '+', '.join(missing))
    return SymMatikaSourceProbe(str(root),True,cmake,cxx,'source tree shape is compatible with the M2N source-job adapter')


def _cpp_string(s: str) -> str:
    return '"' + s.replace('\\','\\\\').replace('"','\\"') + '"'


def _driver_source(req: SymMatikaExplicitRequest, dataset_path: str) -> str:
    vars_cpp=', '.join(_cpp_string(s) for s in (*req.feature_names, req.target_name))
    ops=', '.join('true' if x else 'false' for x in req.allowed_ops)
    return f'''#include <iostream>\n#include <iomanip>\n#include "model.h"\n\nint main() {{\n    DataSet data = DataSet({_cpp_string(dataset_path)}, true, {{{vars_cpp}}});\n    Model model = Model(data, {req.population_size}, {req.max_depth}, {{{ops}}}, {'true' if req.all_variables_relevant else 'false'}, {'true' if req.multiple_distributions else 'false'});\n    model.run();\n    auto final_candidates = model.return_final_candidates();\n    std::cout << "M2N_SYMMATIKA_BEGIN\\n";\n    std::cout << std::setprecision(17);\n    for (const auto &item : final_candidates) {{\n        std::cout << "M2N_SYMMATIKA_CANDIDATE\\t" << item.second << "\\t" << item.first << "\\n";\n    }}\n    std::cout << "M2N_SYMMATIKA_END\\n";\n    return 0;\n}}\n'''


def prepare_symmatika_job(request: SymMatikaExplicitRequest, output_dir: str | os.PathLike) -> SymMatikaPreparedJob:
    request.validate()
    out=Path(output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    dataset=out/'m2n_symmatika_explicit.csv'
    arr=np.column_stack([np.asarray(request.x,dtype=float),np.asarray(request.y,dtype=float).reshape(-1)])
    # Upstream README describes numeric columns and variable names passed separately.
    np.savetxt(dataset,arr,delimiter=',',fmt='%.17g')
    driver=out/'m2n_symmatika_driver.cpp'
    driver.write_text(_driver_source(request,str(dataset)),encoding='utf-8')
    contract=out/'m2n_symmatika_contract.json'
    contract.write_text(json.dumps({
        'backend':'symmatika','mode':'explicit','status':'UNADMITTED_PROPOSAL_GENERATOR',
        'feature_names':list(request.feature_names),'target_name':request.target_name,
        'population_size':request.population_size,'max_depth':request.max_depth,
        'allowed_ops':dict(zip(SYMMATIKA_OPERATOR_ORDER,request.allowed_ops)),
        'all_variables_relevant':request.all_variables_relevant,
        'multiple_distributions':request.multiple_distributions,
        'sample_count':int(arr.shape[0]),
        'admission_required':['physical_semantics','units','support','executable_ir_binding'],
    },indent=2),encoding='utf-8')
    warn='SymMatika candidates are UNADMITTED proposals. Upstream runtime is external C++23/CMake; v1.3 does not vendor upstream source.'
    if arr.shape[0] >= 10_000:
        warn += ' Upstream documentation recommends <10,000 samples for runtime.'
    return SymMatikaPreparedJob(str(out),str(dataset),str(driver),str(contract),int(arr.shape[0]),warn)


def parse_symmatika_driver_output(stdout: str, target_name: str='y') -> CandidateProposalBatch:
    proposals=[]
    in_block=False
    for raw in stdout.splitlines():
        if raw.strip()=='M2N_SYMMATIKA_BEGIN':
            in_block=True; continue
        if raw.strip()=='M2N_SYMMATIKA_END':
            in_block=False; break
        if not in_block or not raw.startswith('M2N_SYMMATIKA_CANDIDATE\t'):
            continue
        parts=raw.split('\t',2)
        if len(parts)!=3:
            continue
        try: loss=float(parts[1])
        except ValueError: loss=float('nan')
        expr=parts[2].strip()
        proposals.append(CandidateProposal(
            id=f'symmatika_eq_{len(proposals)}',expression=expr,target=target_name,
            backend='symmatika',complexity=None,
            metadata=(('status','UNADMITTED'),('upstream_loss',repr(loss)))
        ))
    if not proposals:
        raise ValueError('no M2N_SYMMATIKA_CANDIDATE records found in driver output')
    return CandidateProposalBatch(
        backend='symmatika', proposals=tuple(proposals),
        warning='SymMatika output is symbolic-discovery evidence only. Apply M2N ProposalAdmissionCheck before adding any expression to K2.'
    )


def _discover_built_executable(build_dir: Path) -> Path:
    candidates=[]
    for p in build_dir.rglob('*'):
        if not p.is_file() or 'CMakeFiles' in p.parts:
            continue
        try: executable=os.access(p,os.X_OK)
        except OSError: executable=False
        if executable and p.suffix not in ('.o','.a','.so','.dylib','.dll'):
            candidates.append(p)
    # Prefer names suggesting the project, then shallow executable paths.
    candidates.sort(key=lambda p:(0 if 'symmat' in p.name.lower() else 1,len(p.parts),p.name))
    if not candidates:
        raise BackendUnavailableError('CMake build completed but no executable could be discovered')
    return candidates[0]


def run_symmatika_from_source(
    request: SymMatikaExplicitRequest,
    source_root: str | os.PathLike,
    output_dir: str | os.PathLike,
    *, cxx: str | None=None, timeout_seconds: int=3600,
) -> CandidateProposalBatch:
    """Build/run a temporary copy of an upstream SymMatika source tree.

    The upstream tree is copied; M2N never edits the caller's source tree.
    This path was adapter-harness tested in v1.3, but the upstream SymMatika
    dependency stack was not available in the Linux execution container.
    """
    request.validate()
    probe=probe_symmatika_source(source_root)
    if not probe.ready:
        raise BackendUnavailableError('SymMatika source runtime is not ready: '+probe.reason)
    out=Path(output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    job=prepare_symmatika_job(request,out/'job')
    copied=out/'upstream_source'
    if copied.exists(): shutil.rmtree(copied)
    shutil.copytree(Path(source_root),copied)
    shutil.copyfile(job.driver_cpp,copied/'main.cpp')
    build=out/'build'
    if build.exists(): shutil.rmtree(build)
    env=os.environ.copy()
    if cxx: env['CXX']=cxx
    cfg=subprocess.run(['cmake','-S',str(copied),'-B',str(build)],capture_output=True,text=True,env=env,timeout=timeout_seconds)
    (out/'cmake_configure.log').write_text(cfg.stdout+'\n--- STDERR ---\n'+cfg.stderr,encoding='utf-8')
    if cfg.returncode:
        raise BackendUnavailableError(f'SymMatika CMake configure failed; see {out/"cmake_configure.log"}')
    bld=subprocess.run(['cmake','--build',str(build),'-j2'],capture_output=True,text=True,env=env,timeout=timeout_seconds)
    (out/'cmake_build.log').write_text(bld.stdout+'\n--- STDERR ---\n'+bld.stderr,encoding='utf-8')
    if bld.returncode:
        raise BackendUnavailableError(f'SymMatika CMake build failed; see {out/"cmake_build.log"}')
    exe=_discover_built_executable(build)
    run=subprocess.run([str(exe)],capture_output=True,text=True,env=env,timeout=timeout_seconds)
    (out/'symmatika_stdout.log').write_text(run.stdout+'\n--- STDERR ---\n'+run.stderr,encoding='utf-8')
    if run.returncode:
        raise RuntimeError(f'SymMatika executable failed with exit code {run.returncode}')
    return parse_symmatika_driver_output(run.stdout,request.target_name)
