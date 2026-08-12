from copy import deepcopy
import numpy as np

from isgn import project_mechanism_scope
from isgn.api import recover_graph
from isgn.backends import backend_status
from isgn.simulation import PublicDataset, simulate


def _scaled_thermal(ds, scale=1e6):
    U=ds.U.copy()
    U[:,:,0] *= scale
    return PublicDataset(X=ds.X.copy(), U=U, trajectory_id=ds.trajectory_id.copy(),
                         schema=deepcopy(ds.schema), dt=ds.dt, regime=ds.regime,
                         variant=ds.variant)


def test_mechanism_scope_promotes_public_measurement_without_truth_or_resampling():
    ds,_=simulate('msd',4,T=60,seed=9)
    scoped=project_mechanism_scope(ds,target_states=['x'],driver_states=['v'])
    assert scoped.X.shape == (4,60,1)
    assert scoped.U.shape == (4,59,2)
    np.testing.assert_array_equal(scoped.X[:,:,0],ds.X[:,:,0])
    np.testing.assert_array_equal(scoped.U[:,:,-1],ds.X[:,:-1,1])
    assert scoped.schema['mechanism_scope']['truth_used'] is False
    assert scoped.schema['mechanism_scope']['state_estimation_applied'] is False
    assert scoped.schema['inputs'][-1]['name']=='v'
    assert scoped.schema['inputs'][-1]['observation_role']=='measurement_driver'


def test_reference_sindy_normalization_preserves_small_physical_coefficient_edge():
    tr,_=simulate('thermal',10,T=80,seed=1)
    va,_=simulate('thermal',6,T=80,seed=2)
    tr=_scaled_thermal(tr); va=_scaled_thermal(va)
    raw=recover_graph(tr,va,backend='reference-sindy')
    normalized=recover_graph(tr,va,backend='reference-sindy',
                             backend_options={'normalize_columns':True})
    assert 'P' not in raw.summary.graph['T']
    assert set(normalized.summary.graph['T']) == {'T','P','Ta'}
    assert 0 < abs(normalized.summary.coefficients['T']['P']) < 2e-6
    assert normalized.summary.backend_metadata['derivative_nrmse'] < 1e-3


def test_sklearn_lasso_is_registered_and_enforces_known_absent():
    assert backend_status()['sklearn-lasso'] is True
    tr,_=simulate('thermal',10,T=80,seed=1)
    va,_=simulate('thermal',6,T=80,seed=2)
    r=recover_graph(tr,va,backend='sklearn-lasso')
    assert set(r.summary.graph['T']) == {'T','P','Ta'}
    assert r.summary.backend_metadata['derivative_nrmse'] < 1e-3
    blocked=recover_graph(tr,va,backend='sklearn-lasso',known_absent={'T':['P']})
    assert 'P' not in blocked.summary.graph['T']


def test_sklearn_lasso_handles_mixed_engineering_scales():
    tr,_=simulate('thermal',10,T=80,seed=1)
    va,_=simulate('thermal',6,T=80,seed=2)
    r=recover_graph(_scaled_thermal(tr),_scaled_thermal(va),backend='sklearn-lasso')
    assert set(r.summary.graph['T']) == {'T','P','Ta'}
    assert 0 < abs(r.summary.coefficients['T']['P']) < 2e-6


def test_native_empty_selected_topology_does_not_crash_refit():
    tr,_=simulate('thermal',4,T=45,seed=31)
    va,_=simulate('thermal',3,T=45,seed=32)
    # Eliminate every declared source path for the sole state.  An empty graph is
    # a legitimate sparse-selection outcome and must be returned to certification.
    blocked={'T':['T','P','Ta']}
    r=recover_graph(tr,va,backend='native',known_absent=blocked,epochs=30,refit_epochs=10)
    assert r.summary.graph['T'] == []
