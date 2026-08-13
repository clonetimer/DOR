from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from m2n.auto_qualification import qualify_tabular_design
from m2n.cli import init_design
from m2n.goal_design import run_goal_driven_design
from m2n.practitioner import inspect_tabular_data, validate_device_card


def _duplicated_context_problem(repeats: int = 60):
    # Six train and two validation contexts. Repeating rows increases row count
    # without adding independent commissioning evidence.
    context_values = np.asarray(['tr0', 'tr1', 'tr2', 'tr3', 'tr4', 'tr5', 'va0', 'va1'], dtype=object)
    x_values = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 3.5])
    y_values = 2.0 * x_values + 1.0
    context = np.repeat(context_values, repeats)
    x = np.repeat(x_values, repeats)
    y = np.repeat(y_values, repeats)
    train_idx = np.arange(0, 6 * repeats)
    validation_idx = np.arange(6 * repeats, 8 * repeats)
    return {'x': x, 'y': y}, context, train_idx, validation_idx


def test_row_duplication_cannot_satisfy_independent_context_gate():
    data, context, tri, vai = _duplicated_context_problem(60)

    # Legacy behavior is intentionally preserved when no context contract is declared.
    legacy = qualify_tabular_design(
        data,
        required_inputs=['x'],
        target_columns=['y'],
        train_idx=tri,
        validation_idx=vai,
    )
    assert legacy.status == 'READY', legacy

    qualified = qualify_tabular_design(
        data,
        required_inputs=['x'],
        target_columns=['y'],
        train_idx=tri,
        validation_idx=vai,
        independent_context=context,
        min_train_independent_contexts=8,
        min_validation_independent_contexts=3,
    )
    assert qualified.status == 'AMBIGUOUS', qualified
    assert qualified.diagnostics['n_train'] == 360
    assert qualified.diagnostics['n_validation'] == 120
    assert qualified.diagnostics['n_train_independent_contexts'] == 6
    assert qualified.diagnostics['n_validation_independent_contexts'] == 2
    assert any('insufficient independent contexts' in x for x in qualified.reasons)

    # Ten times more duplicate rows still provide exactly the same independent evidence.
    data2, context2, tri2, vai2 = _duplicated_context_problem(600)
    qualified2 = qualify_tabular_design(
        data2,
        required_inputs=['x'],
        target_columns=['y'],
        train_idx=tri2,
        validation_idx=vai2,
        independent_context=context2,
        min_train_independent_contexts=8,
        min_validation_independent_contexts=3,
    )
    assert qualified2.status == 'AMBIGUOUS', qualified2
    assert qualified2.diagnostics['n_train_independent_contexts'] == 6
    assert qualified2.diagnostics['n_validation_independent_contexts'] == 2


def test_declared_context_without_minimums_fails_closed():
    data, context, tri, vai = _duplicated_context_problem(60)
    result = qualify_tabular_design(
        data,
        required_inputs=['x'],
        target_columns=['y'],
        train_idx=tri,
        validation_idx=vai,
        independent_context=context,
    )
    assert result.status == 'AMBIGUOUS', result
    assert any('fails closed' in x for x in result.reasons)


def test_context_overlap_invalidates_validation_split():
    data, context, tri, vai = _duplicated_context_problem(60)
    context = context.copy()
    context[vai[:60]] = 'tr0'
    result = qualify_tabular_design(
        data,
        required_inputs=['x'],
        target_columns=['y'],
        train_idx=tri,
        validation_idx=vai,
        independent_context=context,
        min_train_independent_contexts=1,
        min_validation_independent_contexts=1,
    )
    assert result.status == 'UNSUPPORTED', result
    assert result.diagnostics['independent_context_split_overlap_count'] == 1
    assert any('validation is not independent' in x for x in result.reasons)


def test_practitioner_and_design_honor_independent_context_contract(tmp_path: Path):
    project = tmp_path / 'project'
    assert init_design(str(project), 'graph', False) == 0
    spec_path = project / 'device_card.json'
    data_path = project / 'data' / 'normal_data.csv'

    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    spec.setdefault('qualification', {})['independent_context_column'] = 'cycle_id'
    spec['qualification']['min_train_independent_contexts'] = 7
    spec['qualification']['min_validation_independent_contexts'] = 3
    spec_path.write_text(json.dumps(spec), encoding='utf-8')

    with data_path.open('r', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys())
    train_i = 0
    val_i = 0
    for row in rows:
        split = row['split']
        if split == 'train':
            row['cycle_id'] = f'tr{train_i // 40}'
            train_i += 1
        else:
            row['cycle_id'] = f'va{val_i // 60}'
            val_i += 1
    with data_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames + ['cycle_id'])
        w.writeheader()
        w.writerows(rows)

    vr = validate_device_card(spec_path)
    assert vr.status == 'PASS', vr

    ir = inspect_tabular_data(data_path, spec_path)
    assert ir.status == 'WARN', ir
    assert ir.diagnostics['split']['train_independent_contexts'] == 6
    assert ir.diagnostics['split']['validation_independent_contexts'] == 2
    assert any('independent context support is below' in x for x in ir.warnings)

    result = run_goal_driven_design(spec_path, data_path, project / 'output')
    assert result.status == 'AMBIGUOUS', result
    assert result.diagnostics['n_train_independent_contexts'] == 6
    assert result.diagnostics['n_validation_independent_contexts'] == 2
    assert any('insufficient independent contexts' in x for x in result.diagnostics['qualification_reasons'])
    assert not (project / 'output' / 'models').exists()


def test_blinded_validation_scope_allows_controlled_falsification_without_production_claim():
    data, context, tri, vai = _duplicated_context_problem(60)
    result = qualify_tabular_design(
        data,
        required_inputs=['x'],
        target_columns=['y'],
        train_idx=tri,
        validation_idx=vai,
        independent_context=context,
        independent_context_evidence_scope='BLINDED_FOR_VALIDATION',
    )
    assert result.status == 'READY', result
    assert result.diagnostics['n_train_independent_contexts'] == 6
    assert result.diagnostics['n_validation_independent_contexts'] == 2
    assert result.diagnostics['production_independent_context_qualified'] is False
    assert 'controlled architecture falsification' in result.diagnostics['qualification_limitation']


def test_blinded_validation_scope_still_needs_context_to_context_validation():
    data, context, tri, vai = _duplicated_context_problem(60)
    # Collapse the two validation contexts into one independent run.  Row count
    # remains 120; controlled context-to-context falsification is no longer possible.
    context = context.copy()
    context[vai] = 'va0'
    result = qualify_tabular_design(
        data,
        required_inputs=['x'],
        target_columns=['y'],
        train_idx=tri,
        validation_idx=vai,
        independent_context=context,
        independent_context_evidence_scope='BLINDED_FOR_VALIDATION',
    )
    assert result.status == 'AMBIGUOUS', result
    assert result.diagnostics['n_validation_independent_contexts'] == 1
    assert any('at least two disjoint train and two validation independent contexts' in x for x in result.reasons)


def test_inspect_accepts_cycle_local_time_resets(tmp_path: Path):
    project = tmp_path / 'project_time'
    assert init_design(str(project), 'graph', False) == 0
    spec_path = project / 'device_card.json'
    data_path = project / 'data' / 'normal_data.csv'

    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    spec.setdefault('qualification', {}).update({
        'independent_context_column': 'cycle_id',
        'evidence_scope': 'BLINDED_FOR_VALIDATION',
    })
    spec.setdefault('data', {})['time_column'] = 't_s'
    spec_path.write_text(json.dumps(spec), encoding='utf-8')

    with data_path.open('r', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f)); fieldnames = list(rows[0].keys())
    counters = {'train': 0, 'val': 0}
    for row in rows:
        split = row['split']
        i = counters[split]; counters[split] += 1
        # Multiple independent cycles with local time reset to zero each cycle.
        block = 40 if split == 'train' else 60
        prefix = 'tr' if split == 'train' else 'va'
        row['cycle_id'] = f'{prefix}{i // block}'
        row['t_s'] = str(i % block)
    with data_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames + ['cycle_id', 't_s'])
        w.writeheader(); w.writerows(rows)

    result = inspect_tabular_data(data_path, spec_path)
    assert result.status in ('PASS', 'WARN'), result
    td = result.diagnostics['time_diagnostics']['t_s']
    assert td['context_aware'] is True
    assert td['duplicate_or_reverse_steps_within_context'] == 0
    assert td['contexts_checked'] >= 4
    assert not any('not strictly increasing' in e for e in result.errors)


def test_blinded_validation_scope_requires_explicit_independent_context():
    data, _, tri, vai = _duplicated_context_problem(60)
    result = qualify_tabular_design(
        data,
        required_inputs=['x'],
        target_columns=['y'],
        train_idx=tri,
        validation_idx=vai,
        independent_context_evidence_scope='BLINDED_FOR_VALIDATION',
    )
    assert result.status == 'UNSUPPORTED', result
    assert any('requires an explicit independent-context column' in x for x in result.reasons)


def test_schema_2_3_is_accepted_by_practitioner_and_design_runtime(tmp_path: Path):
    project = tmp_path / 'schema23'
    assert init_design(str(project), 'graph', False) == 0
    spec_path = project / 'device_card.json'
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    spec['schema_version'] = '2.3'
    spec_path.write_text(json.dumps(spec), encoding='utf-8')
    checked = validate_device_card(spec_path)
    assert checked.status in ('PASS', 'WARN'), checked
    result = run_goal_driven_design(spec_path, project / 'data' / 'normal_data.csv', project / 'output')
    assert result.status in ('SELECTED', 'AMBIGUOUS', 'FIXED'), result
