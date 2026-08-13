from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from m2n.phm_handoff import write_evidence_csv, export_fec_contact_pack


def test_fec_windowing_never_crosses_independent_context_boundaries(tmp_path: Path):
    time = np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=float)
    context = np.asarray(['cycle_a'] * 4 + ['cycle_b'] * 4, dtype=object)
    target = np.zeros((8, 1), dtype=float)
    predictions = {
        'm1': np.asarray([[0.1], [0.2], [0.3], [0.4], [1.1], [1.2], [1.3], [1.4]]),
        'm2': np.asarray([[0.2], [0.3], [0.4], [0.5], [1.2], [1.3], [1.4], [1.5]]),
    }
    evidence = tmp_path / 'evidence.csv'
    write_evidence_csv(
        evidence,
        time=time,
        independent_context=context,
        output_names=('p_down',),
        target=target,
        predictions=predictions,
        support_distance=np.zeros(8),
        support_ood=np.zeros(8, dtype=bool),
        target_scales={'p_down': 1.0},
    )
    with evidence.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert rows[0]['m2n_independent_context'] == 'cycle_a'
    assert rows[4]['m2n_independent_context'] == 'cycle_b'
    assert rows[0]['time_s'] == '0.0' and rows[4]['time_s'] == '0.0'
    assert all(r['m2n_mechanism_evidence_in_support'] == '1' for r in rows)

    contacts = tmp_path / 'contacts.csv'
    contract = tmp_path / 'contract.json'
    export_fec_contact_pack(evidence, contacts, contract, window_s=10.0, min_samples=4)
    out = pd.read_csv(contacts)
    assert len(out) == 2
    assert set(out['m2n_independent_context'].astype(str)) == {'cycle_a', 'cycle_b'}
    assert np.allclose(out['time_s'], [1.5, 1.5])
    assert np.allclose(out['m2n_mechanism_evidence_in_support_fraction'], [1.0, 1.0])
