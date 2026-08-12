from pathlib import Path
import csv, json
import numpy as np

from fec.extensions.class_score_bridge import fit_class_score_reference, infer_class_prediction_sets


def _write_scores(path: Path, *, n=80, mask=(1,1), include_labels=True, seed=5):
    rng=np.random.default_rng(seed)
    fields=['score__NORMAL','score__FAULT_A','mask__vibration','mask__pressure'] + (['true_label'] if include_labels else [])
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for i in range(n):
            y='NORMAL' if i%2==0 else 'FAULT_A'
            # confident but not degenerate scores
            p_true=float(rng.uniform(.72,.95)); p_other=1-p_true
            row={'score__NORMAL':p_true if y=='NORMAL' else p_other,'score__FAULT_A':p_true if y=='FAULT_A' else p_other,
                 'mask__vibration':mask[0],'mask__pressure':mask[1]}
            if include_labels: row['true_label']=y
            w.writerow(row)


def test_class_score_reference_roundtrip_and_prediction_sets(tmp_path:Path):
    cal=tmp_path/'cal.csv'; _write_scores(cal,n=80,mask=(1,1),include_labels=True)
    ref=tmp_path/'ref.json'
    fit_class_score_reference(cal,ref,alpha=.1,min_group_size=20,provenance='unit-test')
    payload=json.loads(ref.read_text())
    assert payload['contract']=='FEC-CLASS-CONFORMAL-REFERENCE/1.0'
    assert payload['class_frame']==['NORMAL','FAULT_A']
    assert payload['groups'][0]['mask']==[1,1]

    scores=tmp_path/'scores.csv'; _write_scores(scores,n=20,mask=(1,1),include_labels=False,seed=9)
    out=tmp_path/'out.csv'; infer_class_prediction_sets(scores,ref,out)
    rows=list(csv.DictReader(out.open()))
    assert len(rows)==20
    assert all(r['calibration_supported']=='True' for r in rows)
    assert all(r['prediction_set_status'] in ('CLASS_SET_SUPPORTED','ABSTAIN') for r in rows)
    assert all(r['prediction_set'].startswith('[') for r in rows)


def test_unseen_modality_mask_abstains_rowwise(tmp_path:Path):
    cal=tmp_path/'cal.csv'; _write_scores(cal,n=60,mask=(1,1),include_labels=True)
    ref=tmp_path/'ref.json'; fit_class_score_reference(cal,ref,min_group_size=20)
    scores=tmp_path/'partial.csv'; _write_scores(scores,n=8,mask=(1,0),include_labels=False)
    out=tmp_path/'out.csv'; infer_class_prediction_sets(scores,ref,out)
    rows=list(csv.DictReader(out.open()))
    assert all(r['calibration_supported']=='False' for r in rows)
    assert all(r['prediction_set_status']=='ABSTAIN' for r in rows)
    assert all('lacks supported labeled calibration' in r['calibration_reason'] for r in rows)
