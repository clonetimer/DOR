#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json, subprocess, sys, tempfile

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / 'examples' / 'nasa_adapt_v06_source_constrained_fixture'
SCHEMA = ROOT / 'schemas' / 'root_cause_result_0.6.schema.json'

sys.path.insert(0, str(ROOT / 'src'))
from dor.rootcause import infer_root_causes_from_files

out = infer_root_causes_from_files(FIX/'fault_ontology.json', FIX/'localization.json', FIX/'evidence.csv')
assert out['contract'] == 'DOR-ROOT-CAUSE-RESULT/0.6'
assert out['status'] == 'SUPPORTED'
assert out['root_cause_set'] == ['EY260_failed_open']
assert out['safety']['root_cause_claimed'] is True
assert out['safety']['propagation_inferred'] is True
assert out['safety']['test_truth_used_for_reasoning'] is False
# Lightweight schema contract checks without requiring jsonschema dependency.
schema = json.loads(SCHEMA.read_text(encoding='utf-8'))
for k in schema['required']:
    assert k in out, f'missing required field: {k}'
assert out['status'] in schema['properties']['status']['enum']

print(json.dumps({
    'state':'PASS',
    'contract':out['contract'],
    'status':out['status'],
    'root_cause_set':out['root_cause_set'],
    'fixture_kind':'source-constrained synthetic timing fixture',
    'real_adapt_accuracy_claimed':False,
}, ensure_ascii=False, indent=2))
