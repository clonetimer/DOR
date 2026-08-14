import json
import numpy as np
import pytest

from dor.m2n_forecast_bridge import bind_m2n_forecast_contract


def test_forecast_horizon_is_injected_only_from_dor_deployment_contract():
    m={'deployment':{'forecast_evaluation':{'mode':'RECURSIVE_ROLLOUT','horizon_steps':8}}}
    c={'contract':{'constraints':{}}}
    r=bind_m2n_forecast_contract(m,c)
    assert r['status']=='BOUND'
    assert r['forecast_evaluation']=={'mode':'RECURSIVE_ROLLOUT','horizon_steps':8}
    assert r['device_card']['contract']['constraints']['forecast_evaluation']['horizon_steps']==8
    assert r['safety']['horizon_invented'] is False


def test_forecast_horizon_mismatch_fails_closed():
    m={'deployment':{'forecast_evaluation':{'mode':'RECURSIVE_ROLLOUT','horizon_steps':8}}}
    c={'contract':{'constraints':{'forecast_evaluation':{'mode':'RECURSIVE_ROLLOUT','horizon_steps':4}}}}
    with pytest.raises(ValueError):
        bind_m2n_forecast_contract(m,c)


def test_recursive_rollout_without_dor_deployment_contract_fails_closed():
    c={'contract':{'constraints':{'forecast_evaluation':{'mode':'RECURSIVE_ROLLOUT','horizon_steps':4}}}}
    with pytest.raises(ValueError):
        bind_m2n_forecast_contract({},c)
