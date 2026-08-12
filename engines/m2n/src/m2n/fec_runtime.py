from __future__ import annotations

from pathlib import Path
import importlib.util, json, sys
import numpy as np


def _load_fec_core(fec_root: str|Path):
    root=Path(fec_root)
    candidates=[root/'core'/'fec_core.py',root/'vendor'/'FEC_Starter_Kit_v1.0'/'core'/'fec_core.py']
    path=next((p for p in candidates if p.is_file()),None)
    if path is None: raise FileNotFoundError('FEC core/fec_core.py not found under supplied FEC root')
    spec=importlib.util.spec_from_file_location('_m2n_external_fec_core',path); mod=importlib.util.module_from_spec(spec); assert spec.loader; sys.modules[spec.name]=mod; spec.loader.exec_module(mod)
    return mod


def _support_fit(X: np.ndarray):
    med=np.median(X,axis=0); mad=np.median(np.abs(X-med[None,:]),axis=0)*1.4826; std=np.std(X,axis=0); floor=np.maximum(np.abs(med)*1e-9,1e-12); scale=np.where(mad>floor,mad,np.where(std>floor,std,floor)); d=np.sqrt(np.mean(((X-med[None,:])/scale[None,:])**2,axis=1)); return med,scale,d


def calibrate_fec_from_contacts(normal_leaf_csv: str|Path, normal_root_csv: str|Path, fec_root: str|Path, output_json: str|Path, *, trusted_claim: str, scope: str='m2n:device') -> Path:
    import pandas as pd
    fec=_load_fec_core(fec_root); leaf=pd.read_csv(normal_leaf_csv); root=pd.read_csv(normal_root_csv)
    contacts=tuple(c for c in leaf.columns if c.startswith('m2n_best_finite_residual__'))+('m2n_model_disagreement',)
    if not contacts: raise ValueError('no M2N contact columns found')
    missing=[c for c in contacts if c not in root.columns]
    if missing: raise KeyError(f'root calibration missing contacts: {missing}')
    Xl=leaf.loc[:,contacts].to_numpy(float); Xr=root.loc[:,contacts].to_numpy(float)
    specs=tuple(fec.ContactSpec(c,(c,'m2n'),None) for c in contacts); A=np.ones(len(specs)); G=np.ones(len(specs)); operators=('mean','mean','mean')
    leaf_ref=fec.fit_reference(Xl); st=fec.coordinate(fec.evidence(Xr,leaf_ref),specs,A=A,G=G,operators=operators,expected_contact_count=len(specs))
    root_ref=fec.make_root_reference(st.root_E,specs,A=A,G=G,scope=scope,dependency_set=[trusted_claim],provenance='m2n-practitioner-controlled-normal-root')
    if not fec.calibration_admission({trusted_claim},root_ref): raise PermissionError('FEC calibration admission rejected trusted claim')
    med,scale,dist=_support_fit(Xr)
    m2n_ref=np.asarray(root['m2n_normal_support_distance'],float)
    payload={'contract':'M2N-FEC-RUNTIME-REFERENCE/2.0','contacts':list(contacts),'operators':list(operators),'leaf_ref':{'center':leaf_ref.center.tolist(),'deviations':[x.tolist() for x in leaf_ref.deviations]},'root_ref':{'scores':root_ref.scores.tolist(),'scope':root_ref.scope,'dependency_set':sorted(root_ref.dependency_set),'provenance':root_ref.provenance},'contact_support':{'median':med.tolist(),'scale':scale.tolist(),'distances':dist.tolist()},'m2n_support_reference':m2n_ref.tolist(),'policy':{'alpha':0.05,'min_S_ref':0.05,'min_A':0.5,'min_C_presence':0.75}}
    p=Path(output_json); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(payload,indent=2),encoding='utf-8'); return p


def infer_fec_contacts(contact_csv: str|Path, reference_json: str|Path, fec_root: str|Path, output_csv: str|Path) -> Path:
    import pandas as pd
    fec=_load_fec_core(fec_root); df=pd.read_csv(contact_csv); d=json.loads(Path(reference_json).read_text(encoding='utf-8'))
    contacts=tuple(d['contacts']); X=df.loc[:,contacts].to_numpy(float); specs=tuple(fec.ContactSpec(c,(c,'m2n'),None) for c in contacts); A=np.ones(len(specs)); G=np.ones(len(specs)); ops=tuple(d['operators'])
    leaf=fec.EmpiricalReference(np.asarray(d['leaf_ref']['center'],float),tuple(np.asarray(x,float) for x in d['leaf_ref']['deviations']))
    root=fec.RootCalibrationReference(np.asarray(d['root_ref']['scores'],float),tuple((s.path,s.equivalence_id,1.0,1) for s in specs),d['root_ref']['scope'],frozenset(d['root_ref']['dependency_set']),d['root_ref']['provenance'])
    E=fec.evidence(X,leaf); st=fec.coordinate(E,specs,A=A,G=G,operators=ops,expected_contact_count=len(specs)); p=fec.calibrated_p(root,st.root_E,specs,A=A,G=G)
    cs=d['contact_support']; med=np.asarray(cs['median']); scale=np.asarray(cs['scale']); refd=np.asarray(cs['distances']); dist=np.sqrt(np.mean(((X-med[None,:])/scale[None,:])**2,axis=1)); S_contact=fec.support_p(refd,dist)
    S_m2n=fec.support_p(np.asarray(d['m2n_support_reference'],float),np.asarray(df['m2n_normal_support_distance'],float)); S_m2n=S_m2n*(1-np.clip(np.asarray(df.get('m2n_normal_support_ood_fraction',0.0),float),0,1)); S_ref=np.minimum(S_contact,S_m2n)
    pol=d['policy']; decisions=[]
    for i in range(len(df)):
        if S_ref[i] < pol['min_S_ref']: dec='REFERENCE_UNSUPPORTED'
        elif st.C_presence < pol['min_C_presence']: dec='INSUFFICIENT_COVERAGE'
        elif st.A_mean_present < pol['min_A'] and p[i] <= pol['alpha']: dec='ANOMALOUS_WITH_LOW_APPLICABILITY'
        elif st.A_mean_present < pol['min_A']: dec='LOW_APPLICABILITY'
        elif p[i] <= pol['alpha']: dec='ANOMALOUS_EVIDENCE'
        else: dec='NORMAL_EVIDENCE'
        decisions.append(dec)
    out=pd.DataFrame({'time_s':df['time_s'] if 'time_s' in df else np.arange(len(df)),'root_E':st.root_E,'p_root':p,'A_mean_present':st.A_mean_present,'C_presence':st.C_presence,'C_effective':st.C_effective,'S_ref':S_ref,'decision':decisions})
    for j,c in enumerate(contacts): out['E_contact__'+c]=E[:,j]
    pth=Path(output_csv); pth.parent.mkdir(parents=True,exist_ok=True); out.to_csv(pth,index=False); return pth
