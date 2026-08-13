from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from m2n.model_family import qualify_model_families


def _base_spec(contract):
    return {
        'schema_version': '2.4',
        'template': 'generic_mechanism_graph',
        'data': {'split_column': 'split', 'time_column': 't_s'},
        'mechanism': {'inputs': [{'name': 'x'}], 'nodes': [{'id': 'yhat', 'op': 'identity', 'inputs': ['x']}], 'outputs': [{'name': 'yhat'}]},
        'knowledge': {'slots': [{'id': 'local', 'level': 'K2', 'finite_candidates': ['a', 'b'], 'allow_k3_escape': True}]},
        'supervision': {'input_columns': {'x': 'x'}, 'output_columns': {'yhat': 'y'}},
        'qualification': {'independent_context_column': 'cycle_id', 'evidence_scope': 'BLINDED_FOR_VALIDATION'},
        'modeling_contract': contract,
    }


def _statuses(plan):
    return {x.family.value: x.status for x in plan.families}


def test_mechanism_required_static_regression_keeps_black_box_out():
    spec = _base_spec({
        'objective': 'STATIC_REGRESSION',
        'input_representation': 'TABULAR',
        'mechanism_usage': 'REQUIRED',
        'input_columns': ['x'],
        'target_columns': ['y'],
        'required_outputs': ['prediction', 'residual'],
    })
    data = {'x': np.arange(8.0), 'y': np.arange(8.0), 'cycle_id': np.arange(8), 'split': np.array(['train'] * 6 + ['val'] * 2), 't_s': np.zeros(8)}
    plan = qualify_model_families(spec, data=data)
    s = _statuses(plan)
    assert plan.status == 'READY'
    assert s['PHYSICS_PARAMETRIC'] == 'ELIGIBLE'
    assert s['CLASSICAL_ML'] == 'NOT_ELIGIBLE'
    assert s['DEEP_LEARNING'] == 'NOT_ELIGIBLE'
    assert s['SCIML_HYBRID'] == 'CONDITIONAL'


def test_normality_contract_qualifies_statistical_and_classical_without_diagnosis_semantics():
    spec = _base_spec({
        'objective': 'NORMALITY_MODELING',
        'input_representation': 'TABULAR',
        'mechanism_usage': 'OPTIONAL',
        'input_columns': ['x'],
        'prediction_unit': 'INDEPENDENT_CONTEXT',
        'required_outputs': ['score'],
    })
    data = {'x': np.arange(8.0), 'cycle_id': np.arange(8), 'split': np.array(['train'] * 6 + ['val'] * 2), 't_s': np.zeros(8), 'y': np.arange(8.0)}
    s = _statuses(qualify_model_families(spec, data=data))
    assert s['STATISTICAL'] == 'ELIGIBLE'
    assert s['CLASSICAL_ML'] == 'ELIGIBLE'
    assert s['DEEP_LEARNING'] == 'CONDITIONAL'
    assert s['OPERATOR_FIELD'] == 'NOT_APPLICABLE'


def test_categorical_contract_qualifies_tabular_classical_ml_but_does_not_assign_fault_meaning():
    spec = _base_spec({
        'objective': 'CATEGORICAL_PREDICTION',
        'input_representation': 'TABULAR',
        'mechanism_usage': 'OPTIONAL',
        'input_columns': ['x'],
        'label_column': 'label',
        'prediction_unit': 'INDEPENDENT_CONTEXT',
        'required_outputs': ['class_score'],
    })
    data = {'x': np.arange(8.0), 'label': np.array(['a', 'a', 'b', 'b', 'c', 'c', 'a', 'b']), 'cycle_id': np.arange(8), 'split': np.array(['train'] * 6 + ['val'] * 2), 't_s': np.zeros(8), 'y': np.arange(8.0)}
    plan = qualify_model_families(spec, data=data)
    s = _statuses(plan)
    assert s['CLASSICAL_ML'] == 'ELIGIBLE'
    assert s['DEEP_LEARNING'] == 'CONDITIONAL'
    assert s['PHYSICS_PARAMETRIC'] == 'NOT_APPLICABLE'
    assert plan.diagnostics['label_classes'] == 3
    assert 'does not' in plan.boundary.lower() and 'fault meaning' in plan.boundary.lower()


def test_dynamics_requires_time_contract():
    spec = _base_spec({
        'objective': 'DYNAMICS_IDENTIFICATION',
        'input_representation': 'MULTIVARIATE_SEQUENCE',
        'mechanism_usage': 'PREFERRED',
        'input_columns': ['x'],
        'target_columns': ['y'],
        'prediction_unit': 'TRAJECTORY',
        'required_outputs': ['forecast'],
    })
    spec['data'].pop('time_column')
    plan = qualify_model_families(spec, data={'x': np.arange(8.0), 'y': np.arange(8.0), 'cycle_id': np.arange(8), 'split': np.array(['train'] * 6 + ['val'] * 2)})
    s = _statuses(plan)
    assert s['SYSTEM_IDENTIFICATION'] == 'NOT_ELIGIBLE'


def test_practitioner_accepts_valid_modeling_contract(tmp_path: Path):
    from m2n.practitioner import validate_device_card
    spec = _base_spec({
        'objective': 'NORMALITY_MODELING',
        'input_representation': 'TABULAR',
        'mechanism_usage': 'OPTIONAL',
        'input_columns': ['x'],
        'prediction_unit': 'INDEPENDENT_CONTEXT',
    })
    # Make the graph structurally valid for practitioner preflight.
    spec['mechanism'] = {
        'inputs': [{'name': 'x', 'dim': {}}],
        'nodes': [{'id': 'yhat', 'op': 'positive_gain', 'inputs': ['x'], 'params': {'learnable': True, 'init': 1.0, 'gain_dim': {}}}],
        'outputs': [{'name': 'yhat', 'dim': {}}],
    }
    spec['knowledge'] = {'slots': [{'id': 'local', 'level': 'K1', 'role': 'local', 'input_dims': [{}], 'output_dim': {}, 'exact_family': 'gain', 'learnable_parameters': ['g'], 'knowledge_source': 'DECLARED_ENGINEERING'}]}
    p = tmp_path / 'device_card.json'
    p.write_text(json.dumps(spec), encoding='utf-8')
    r = validate_device_card(p)
    assert r.status in ('PASS', 'WARN')
    assert r.diagnostics['modeling_contract']['objective'] == 'NORMALITY_MODELING'
    assert any('model-plan' in a for a in r.actions)
