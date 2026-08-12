from __future__ import annotations
from typing import Any, Dict
from .pysindy_backend import PySINDyBackend
from .reference_sindy import ReferenceSINDyBackend
from .sklearn_lasso import SklearnLassoBackend
from .symmatika_backend import SymMatikaExternalBackend

_BACKENDS={'pysindy':PySINDyBackend,'reference-sindy':ReferenceSINDyBackend,'sklearn-lasso':SklearnLassoBackend}

CAPABILITY_MANIFEST: Dict[str, Dict[str, Any]] = {
    'native': {
        'role':'frozen ISGN-v1 hierarchical grammar solver',
        'stable_api':True, 'control_inputs':True, 'partial_prior_present':True,
        'partial_prior_absent':True, 'constitutive_identity_sat_deadzone':True,
        'weak_form':False, 'official_external':False,
    },
    'pysindy': {
        'role':'official PySINDy/SINDYc sparse system-identification backend',
        'stable_api':True, 'control_inputs':True,
        'partial_prior_present':'audit-only in F1', 'partial_prior_absent':'ConstrainedSR3 equality-to-zero',
        'constitutive_identity_sat_deadzone':'ISGN profile wrapper + official PySINDy fit',
        'weak_form':'upstream PySINDy supports WeakPDELibrary; F1 adapter exposure pending',
        'official_external':True, 'recommended_for':['sparse_ode','sindyc','known_absent_constraints'],
    },
    'sklearn-lasso': {
        'role':'independent scale-normalized sparse regression backend using scikit-learn Lasso',
        'stable_api':False, 'control_inputs':True, 'partial_prior_present':'audit-only',
        'partial_prior_absent':True, 'constitutive_identity_sat_deadzone':False,
        'weak_form':False, 'official_external':True, 'recommended_for':['industrial_unit_scaling','independent_sparse_consensus'],
    },
    'reference-sindy': {
        'role':'offline QA STLSQ-SINDYc reference; NOT official PySINDy',
        'stable_api':False, 'control_inputs':True, 'partial_prior_present':'audit-only',
        'partial_prior_absent':True, 'constitutive_identity_sat_deadzone':True,
        'weak_form':False, 'official_external':False,
    },
    'symmatika': {
        'role':'external algebraic/constitutive symbolic-relation discovery for GRAMMAR_UNCOVERED gaps',
        'stable_api':False, 'scope':'relation-discovery-only', 'explicit_relations':True,
        'implicit_relations':'upstream supports; F2 bridge starts with explicit relations',
        'official_external':True,
    },
}

def backend_status() -> Dict[str,bool]:
    return {k:v.available() for k,v in _BACKENDS.items()}

def backend_manifest() -> Dict[str,Dict[str,Any]]:
    status={'native':True,**backend_status(),'symmatika':SymMatikaExternalBackend.available()}
    return {k:{**v,'available':bool(status.get(k,False))} for k,v in CAPABILITY_MANIFEST.items()}

def recommend_backend(*, constitutive_search:bool=False, has_partial_prior:bool=False,
                      noisy:bool=False, require_weak:bool=False) -> Dict[str,str]:
    st=backend_status()
    if require_weak:
        if st.get('pysindy'):
            return {'backend':'pysindy','reason':'PySINDy upstream provides WeakPDELibrary; use advanced PySINDy workflow until F1 weak adapter is exposed'}
        return {'backend':'native','reason':'official PySINDy is unavailable; native is fallback but does not provide weak-form SINDy'}
    if st.get('pysindy'):
        reason='official PySINDy preferred for sparse ODE/SINDYc'
        if has_partial_prior: reason += '; known-absent priors use ConstrainedSR3 and known-present priors are certified above the solver'
        if constitutive_search: reason += '; finite Saturation/DeadZone transforms are profiled by ISGN and each candidate is fit by PySINDy'
        return {'backend':'pysindy','reason':reason}
    return {'backend':'native','reason':'official PySINDy optional dependency is not installed; use frozen native solver'}

def get_backend(name:str):
    if name not in _BACKENDS: raise KeyError(f'unknown external backend {name!r}')
    cls=_BACKENDS[name]
    if not cls.available():
        if name=='pysindy': raise RuntimeError('PySINDy backend unavailable. Install optional dependency: pip install "isgn-toolkit[pysindy]"')
        raise RuntimeError(f'backend {name} unavailable')
    return cls()
