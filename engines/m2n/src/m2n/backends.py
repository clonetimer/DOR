from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
import importlib.util
import importlib
from importlib import metadata




@dataclass(frozen=True)
class BackendRuntimeProbe:
    id: str
    installed: bool
    importable: bool
    version: str | None
    error: str | None = None

    @property
    def ready(self) -> bool:
        return self.installed and self.importable

    def as_dict(self):
        return asdict(self) | {'ready': self.ready}


class BackendCapability(str, Enum):
    CANDIDATE_DISCOVERY = 'CANDIDATE_DISCOVERY'
    HYBRID_K3_TRAINING = 'HYBRID_K3_TRAINING'
    CONSTITUTIVE_MODELING = 'CONSTITUTIVE_MODELING'
    PROGRAM_COMPILATION = 'PROGRAM_COMPILATION'
    LOCAL_NAS = 'LOCAL_NAS'
    MODEL_COMPONENT_INFERENCE = 'MODEL_COMPONENT_INFERENCE'


@dataclass(frozen=True)
class BackendDescriptor:
    id: str
    project: str
    capability: tuple[BackendCapability, ...]
    integration_status: str  # NATIVE | OPTIONAL_ADAPTER | EXTERNAL_ADAPTER | CONTRACT_ONLY
    package_name: str | None = None
    install_extra: str | None = None
    code_license_status: str = 'UNVERIFIED'
    code_license: str | None = None
    paper_reference: str | None = None
    notes: str = ''

    def installed(self) -> bool:
        if self.integration_status == 'NATIVE':
            return True
        return bool(self.package_name and importlib.util.find_spec(self.package_name) is not None)

    def installed_version(self) -> str | None:
        if not self.installed() or not self.package_name:
            return None
        try:
            return metadata.version(self.package_name)
        except metadata.PackageNotFoundError:
            try:
                mod=__import__(self.package_name)
                return str(getattr(mod,'__version__','unknown'))
            except Exception:
                return 'unknown'

    def runtime_probe(self) -> BackendRuntimeProbe:
        installed=self.installed()
        version=self.installed_version() if installed else None
        if self.integration_status == 'NATIVE':
            return BackendRuntimeProbe(self.id, True, True, version or 'native', None)
        if self.integration_status == 'EXTERNAL_ADAPTER':
            return BackendRuntimeProbe(self.id, False, False, None, 'external source/runtime must be supplied explicitly; use the backend-specific source probe')
        if not installed or not self.package_name:
            return BackendRuntimeProbe(self.id, installed, False, version, 'package not installed')
        try:
            importlib.import_module(self.package_name)
            return BackendRuntimeProbe(self.id, True, True, version, None)
        except Exception as e:
            return BackendRuntimeProbe(self.id, True, False, version, f'{type(e).__name__}: {e}')

    def as_dict(self):
        d=asdict(self)
        d['capability']=[x.value for x in self.capability]
        probe=self.runtime_probe()
        d['installed']=probe.installed
        d['installed_version']=probe.version
        d['importable']=probe.importable
        d['runtime_error']=probe.error
        return d


_BACKENDS = (
    BackendDescriptor(
        id='m2n-native', project='M2N native finite-family/compiler stack',
        capability=(BackendCapability.CANDIDATE_DISCOVERY, BackendCapability.PROGRAM_COMPILATION),
        integration_status='NATIVE', code_license_status='PROJECT_CONTROLLED',
        notes='Frozen M2N v6.3 capability; novelty is assessed separately from utility.',
    ),
    BackendDescriptor(
        id='pysindy', project='PySINDy',
        capability=(BackendCapability.CANDIDATE_DISCOVERY,),
        integration_status='OPTIONAL_ADAPTER', package_name='pysindy', install_extra='discovery',
        code_license_status='VERIFIED', code_license='MIT',
        paper_reference='Brunton, Proctor & Kutz, PNAS 2016; PySINDy 2.x documentation',
        notes='Sparse equation discovery backend. Proposals remain unadmitted until M2N qualification and IR binding.',
    ),
    BackendDescriptor(
        id='sbmi', project='Simulation-Based Model Inference (SBMI)',
        capability=(BackendCapability.MODEL_COMPONENT_INFERENCE,),
        integration_status='CONTRACT_ONLY',
        code_license_status='UNVERIFIED',
        paper_reference='Schroder & Macke, ICML 2024',
        notes='Feasibility contract available in v1.3. Exact implementation remains external/contract-only; use only when finite model families are too large for transparent enumeration.',
    ),
    BackendDescriptor(
        id='symmatika', project='SymMatika',
        capability=(BackendCapability.CANDIDATE_DISCOVERY,),
        integration_status='EXTERNAL_ADAPTER',
        code_license_status='VERIFIED', code_license='Apache-2.0',
        paper_reference='Scherk & Chen, SymMatika, ICLR 2026 submission / arXiv 2507.03110',
        notes='Apache-2.0 repository license verified. v1.3 provides a C++ source-job adapter; upstream source/dependencies are not vendored and must be supplied externally.',
    ),
    BackendDescriptor(
        id='ude', project='Universal Differential Equations / SciML',
        capability=(BackendCapability.HYBRID_K3_TRAINING,),
        integration_status='CONTRACT_ONLY',
        code_license_status='VERIFIED', code_license='MIT (paper demo repository)',
        paper_reference='Rackauckas et al., Universal Differential Equations, 2020/2021',
        notes='Candidate K3 training backend; M2N remains responsible for whether/where K3 is admissible.',
    ),
    BackendDescriptor(
        id='aphynity', project='APHYNITY',
        capability=(BackendCapability.HYBRID_K3_TRAINING,),
        integration_status='CONTRACT_ONLY',
        code_license_status='VERIFIED', code_license='MIT',
        paper_reference='Yin et al., ICLR 2021',
        notes='MIT repository license verified. No source is vendored; K3 integration is deferred to a dedicated hybrid-training adapter.',
    ),
    BackendDescriptor(
        id='cann', project='Constitutive Artificial Neural Networks',
        capability=(BackendCapability.CONSTITUTIVE_MODELING,),
        integration_status='CONTRACT_ONLY',
        code_license_status='UNVERIFIED',
        paper_reference='Linka & Kuhl, automated constitutive model discovery, 2022/2023',
        notes='Domain-specialist backend candidate for constitutive mechanics.',
    ),
    BackendDescriptor(
        id='neural-compiler', project='The Neural Compiler',
        capability=(BackendCapability.PROGRAM_COMPILATION,),
        integration_status='CONTRACT_ONLY',
        code_license_status='VERIFIED', code_license='MIT',
        paper_reference='Sheneman, arXiv 2605.22498, 2026',
        notes='MIT repository license verified. No source is vendored; compiler adapter remains future optional infrastructure.',
    ),
)


def list_backends() -> tuple[BackendDescriptor, ...]:
    return _BACKENDS


def get_backend(backend_id: str) -> BackendDescriptor:
    for b in _BACKENDS:
        if b.id == backend_id:
            return b
    raise KeyError(backend_id)
