from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import csv, json
import numpy as np
from fec.core.fec_core import ContactSpec, fit_reference, evidence, coordinate, make_root_reference, calibrated_p

@dataclass
class PolicyEvidenceState:
    root_E: float
    p_root: float
    A_mean_present: float
    C_presence: float
    C_effective: float
    S_ctx: float
    S_ref: float
    decision: str

def read_csv_matrix(path, columns):
    rows=[]
    with open(path,'r',encoding='utf-8-sig',newline='') as f:
        r=csv.DictReader(f)
        missing=[c for c in columns if c not in (r.fieldnames or [])]
        if missing: raise KeyError(f'Missing CSV columns: {missing}')
        for row in r: rows.append([float(row[c]) for c in columns])
    if not rows: raise ValueError(f'No rows in {path}')
    return np.asarray(rows,float)

def read_config(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def specs_from_config(cfg):
    cols=[]; specs=[]
    for c in cfg['contacts']:
        cols.append(c['column'])
        specs.append(ContactSpec(c['key'],tuple(c.get('path',[])),c.get('equivalence_id')))
    return cols,specs

def ag_from_config(cfg):
    A=np.asarray([float(c.get('A',1.0)) for c in cfg['contacts']],float)
    G=np.asarray([float(c.get('G',1.0)) for c in cfg['contacts']],float)
    return A,G

def fixed_policy(*,p_root,A,C,S_ctx,S_ref,cfg):
    p=cfg.get('policy',{}); alpha=float(p.get('alpha',0.05))
    if S_ctx < float(p.get('min_S_ctx',0.05)): return 'CONTEXT_UNSUPPORTED'
    if S_ref < float(p.get('min_S_ref',0.05)): return 'REFERENCE_UNSUPPORTED'
    if C < float(p.get('min_C_presence',0.75)): return 'INSUFFICIENT_COVERAGE'
    if A < float(p.get('min_A',0.5)) and p_root <= alpha: return 'ANOMALOUS_WITH_LOW_APPLICABILITY'
    if A < float(p.get('min_A',0.5)): return 'LOW_APPLICABILITY'
    if p_root <= alpha: return 'ANOMALOUS_EVIDENCE'
    return 'NORMAL_EVIDENCE'

def build_model(normal_leaf,normal_root,specs,A,G,cfg):
    leaf_ref=fit_reference(normal_leaf)
    st_cal=coordinate(evidence(normal_root,leaf_ref),specs,A=A,G=G,operators=cfg.get('operators'),expected_contact_count=int(cfg.get('expected_contact_count',len(specs))))
    cal=cfg.get('calibration',{})
    root_ref=make_root_reference(st_cal.root_E,specs,A=A,G=G,scope=cal.get('scope','root'),dependency_set=cal.get('dependency_set',[]),provenance=cal.get('provenance','trusted-normal'))
    return leaf_ref,root_ref

def infer_rows(X,leaf_ref,root_ref,specs,A,G,cfg,S_ctx=None,S_ref=None):
    st=coordinate(evidence(X,leaf_ref),specs,A=A,G=G,operators=cfg.get('operators'),expected_contact_count=int(cfg.get('expected_contact_count',len(specs))))
    p=calibrated_p(root_ref,st.root_E,specs,A=A,G=G)
    if S_ctx is None: S_ctx=np.ones(len(X))
    if S_ref is None: S_ref=np.ones(len(X))
    out=[]
    for i in range(len(X)):
        d=fixed_policy(p_root=float(p[i]),A=st.A_mean_present,C=st.C_presence,S_ctx=float(S_ctx[i]),S_ref=float(S_ref[i]),cfg=cfg)
        out.append(PolicyEvidenceState(float(st.root_E[i]),float(p[i]),float(st.A_mean_present),float(st.C_presence),float(st.C_effective),float(S_ctx[i]),float(S_ref[i]),d))
    return out

def write_states(path,states):
    fields=list(asdict(states[0]).keys())
    with open(path,'w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for s in states: w.writerow(asdict(s))
