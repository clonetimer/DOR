import json, os
from pathlib import Path
import numpy as np

from isgn.io import from_arrays
from isgn.symbolic import export_explicit_relation, validate_expression
from isgn.backends import SymMatikaExternalBackend, backend_manifest


def _relation_ds():
    rng=np.random.RandomState(0); N,T=3,30
    qin=rng.uniform(.1,2.0,size=(N,T-1)); qout=.6*np.sqrt(qin)
    U=np.stack([qin,qout],axis=-1).astype('float32')
    X=np.zeros((N,T,1),dtype='float32')
    schema={'states':[{'name':'h','dim':'level'}],'inputs':[{'name':'qin','dim':'volume_flow'},{'name':'qout','dim':'volume_flow'}]}
    return from_arrays(X,U,schema=schema,dt=.1)


def test_symbolic_export_and_validate(tmp_path):
    ds=_relation_ds(); p=tmp_path/'rel.csv'
    task=export_explicit_relation(ds,inputs=['qin'],target='qout',out_csv=p)
    assert p.exists() and task.rows==3*29
    first=p.read_text().splitlines()[0].split(',')
    assert len(first)==2 and all(float(x)==float(x) for x in first)
    v=validate_expression(ds,inputs=['qin'],target='qout',expression='0.6*sqrt(qin)')
    assert v['nrmse'] < 1e-6


def test_symmatika_bridge_protocol(tmp_path):
    ds=_relation_ds(); task=export_explicit_relation(ds,inputs=['qin'],target='qout',out_csv=tmp_path/'r.csv')
    exe=tmp_path/'bridge.py'
    exe.write_text("#!/usr/bin/env python3\nimport json\nprint(json.dumps({'candidates':[{'expression':'0.6*sqrt(qin)','error':0.0}]}))\n")
    exe.chmod(0o755)
    b=SymMatikaExternalBackend(str(exe))
    r=b.run_explicit(task.csv_path,task.variables,population=10,depth=2,timeout=10)
    assert r.backend=='symmatika' and r.candidates[0].expression=='0.6*sqrt(qin)'


def test_manifest_lists_symmatika():
    assert 'symmatika' in backend_manifest()
