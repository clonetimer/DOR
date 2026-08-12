from __future__ import annotations
from pathlib import Path
from typing import Sequence
import json
import numpy as np
import pandas as pd


def _as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool: return s
    return s.astype(str).str.lower().isin(["true","1","yes"])


def arbitrate_anomaly_csv(fec_csv: str | Path, output_csv: str | Path, summary_json: str | Path | None = None) -> Path:
    """Map FEC scalar-score calibration states to DOR detection decisions.

    Supported FEC `NORMAL/ABNORMAL` rows become DOR `SUPPORTED`; unsupported
    calibration contexts become `ABSTAIN`. DOR does not synthesize a fault class
    from a scalar anomaly score.
    """
    df=pd.read_csv(fec_csv)
    required={"decision","calibration_supported"}
    missing=required-set(df.columns)
    if missing: raise ValueError(f"FEC anomaly output missing columns: {sorted(missing)}")
    supported=_as_bool(df.calibration_supported)
    status=[]; diagnosis=[]; reason=[]
    for ok,d in zip(supported,df.decision.astype(str)):
        if ok and d in {"NORMAL","ABNORMAL"}:
            status.append("SUPPORTED"); diagnosis.append(d); reason.append("FEC calibration at the DOR-selected declared reference level supported the scalar anomaly decision")
        else:
            status.append("ABSTAIN"); diagnosis.append("UNRESOLVED"); reason.append("DOR/FEC reference applicability unsupported or power-limited; DOR refuses a normal/abnormal claim")
    out=df.copy(); out["diagnosis_status"]=status; out["diagnosis"]=diagnosis; out["diagnosis_reason"]=reason
    op=Path(output_csv); op.parent.mkdir(parents=True,exist_ok=True); out.to_csv(op,index=False)
    if summary_json is not None:
        summary={
            "contract":"DOR-DIAGNOSIS-BUNDLE/0.4",
            "task":"ANOMALY_DETECTION",
            "n":len(out),
            "supported_fraction":float((out.diagnosis_status=="SUPPORTED").mean()),
            "abstain_fraction":float((out.diagnosis_status=="ABSTAIN").mean()),
            "abnormal_fraction_among_supported":float((out.loc[out.diagnosis_status=="SUPPORTED","diagnosis"]=="ABNORMAL").mean()) if (out.diagnosis_status=="SUPPORTED").any() else None,
            "prohibited_claims":[
                "scalar anomaly score or calibrated p-value is not a fault-class probability",
                "ABNORMAL is detection evidence, not physical root cause",
                "unsupported context is an explicit abstention, not NORMAL",
            ],
        }
        Path(summary_json).write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return op


def _parse_set(x) -> set[str]:
    if isinstance(x,float) and np.isnan(x): return set()
    try: return set(map(str,json.loads(str(x))))
    except Exception: return set()



def arbitrate_anomaly_portfolio(
    fec_csvs: Sequence[str | Path], output_csv: str | Path, summary_json: str | Path | None = None,
    *, id_column: str | None = None,
) -> Path:
    """Conservative arbitration across multiple calibrated anomaly detectors.

    DOR supports a scalar NORMAL/ABNORMAL claim only when every retained model
    has calibration support and all retained models agree. Any supported-model
    disagreement is preserved as ALTERNATIVES; no supported model means ABSTAIN.
    """
    if not fec_csvs:
        raise ValueError("at least one FEC anomaly output is required")
    frames=[]
    for p in fec_csvs:
        df=pd.read_csv(p)
        if "decision" not in df or "calibration_supported" not in df:
            raise ValueError(f"invalid FEC anomaly output: {p}")
        frames.append((Path(p).stem,df.copy()))
    base=frames[0][1]
    if id_column is None:
        for candidate in ("segment","sample_id","id"):
            if candidate in base.columns:
                id_column=candidate; break
    ids = base[id_column].astype(str).tolist() if id_column and id_column in base.columns else [str(i) for i in range(len(base))]
    for name,df in frames[1:]:
        other = df[id_column].astype(str).tolist() if id_column and id_column in df.columns else [str(i) for i in range(len(df))]
        if other != ids:
            raise ValueError(f"row/id mismatch in {name}")
    rows=[]
    for i,sid in enumerate(ids):
        decisions=[]; all_supported=True
        for _,df in frames:
            ok=bool(_as_bool(pd.Series([df.iloc[i]["calibration_supported"]])).iloc[0])
            d=str(df.iloc[i]["decision"])
            if not ok or d not in {"NORMAL","ABNORMAL"}:
                all_supported=False
            if ok and d in {"NORMAL","ABNORMAL"}:
                decisions.append(d)
        if not decisions:
            status="ABSTAIN"; diagnosis="UNRESOLVED"; dset=[]; why="no retained detector has supported calibrated evidence"
        elif all_supported and len(set(decisions))==1:
            status="SUPPORTED"; diagnosis=decisions[0]; dset=[decisions[0]]; why="all retained calibrated detectors agree"
        else:
            status="ALTERNATIVES"; diagnosis="UNRESOLVED"; dset=sorted(set(decisions)); why="retained calibrated detectors disagree or some retained detector is context-unsupported"
        row={"diagnosis_status":status,"diagnosis":diagnosis,"diagnosis_set":json.dumps(dset,ensure_ascii=False),"diagnosis_reason":why}
        if id_column: row[id_column]=sid
        rows.append(row)
    out=pd.DataFrame(rows); op=Path(output_csv); op.parent.mkdir(parents=True,exist_ok=True); out.to_csv(op,index=False)
    if summary_json is not None:
        summary={"contract":"DOR-DIAGNOSIS-BUNDLE/0.4","task":"ANOMALY_DETECTION","n":len(out),
                 "supported_fraction":float((out.diagnosis_status=="SUPPORTED").mean()),
                 "alternatives_fraction":float((out.diagnosis_status=="ALTERNATIVES").mean()),
                 "abstain_fraction":float((out.diagnosis_status=="ABSTAIN").mean()),
                 "prohibited_claims":["portfolio agreement is a DOR policy result, not a new conformal coverage theorem","ABNORMAL is detection evidence, not physical root cause"]}
        Path(summary_json).write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return op

def arbitrate_class_portfolio(
    fec_csvs: Sequence[str | Path], output_csv: str | Path, summary_json: str | Path | None = None,
    *, id_column: str = "sample_id", true_label_column: str = "true_label",
) -> Path:
    """Conservative DOR arbitration over one or more FEC class prediction-set files."""
    if not fec_csvs: raise ValueError("at least one FEC class output is required")
    frames=[]
    for p in fec_csvs:
        df=pd.read_csv(p,dtype={true_label_column:str} if true_label_column else None)
        if id_column not in df or "prediction_set" not in df or "calibration_supported" not in df:
            raise ValueError(f"invalid FEC class output: {p}")
        df=df.copy(); df["_set"]=df.prediction_set.map(_parse_set); df["_supported"]=_as_bool(df.calibration_supported)
        frames.append((Path(p).stem,df))
    base=frames[0][1]
    ids=base[id_column].astype(str).tolist()
    for name,df in frames[1:]:
        if df[id_column].astype(str).tolist()!=ids: raise ValueError(f"row/id mismatch in {name}")
    rows=[]
    for i,sid in enumerate(ids):
        sets=[]
        for _,df in frames:
            sets.append(df.iloc[i]["_set"] if bool(df.iloc[i]["_supported"]) else set())
        nonempty=[s for s in sets if s]
        if not nonempty:
            status="ABSTAIN"; final=set(); why="all retained calibrated models are unsupported or empty"
        elif len(nonempty)==len(sets) and all(len(s)==1 for s in sets) and len({next(iter(s)) for s in sets})==1:
            status="SUPPORTED"; final=set(nonempty[0]); why="all retained calibrated models agree on the same singleton"
        else:
            status="ALTERNATIVES"; final=set.union(*nonempty); why="model/calibration ambiguity remains; preserve union of non-empty calibrated sets"
        row={id_column:sid,"diagnosis_status":status,"diagnosis_set":json.dumps(sorted(final),ensure_ascii=False),"diagnosis_set_size":len(final),"diagnosis_reason":why}
        if true_label_column and true_label_column in base.columns: row[true_label_column]=str(base.iloc[i][true_label_column])
        if "split_group" in base.columns: row["split_group"]=base.iloc[i]["split_group"]
        rows.append(row)
    out=pd.DataFrame(rows); op=Path(output_csv); op.parent.mkdir(parents=True,exist_ok=True); out.to_csv(op,index=False)
    if summary_json is not None:
        summary={"contract":"DOR-DIAGNOSIS-BUNDLE/0.4","task":"FAULT_CLASSIFICATION","n":len(out),
                 "supported_fraction":float((out.diagnosis_status=="SUPPORTED").mean()),"alternatives_fraction":float((out.diagnosis_status=="ALTERNATIVES").mean()),"abstain_fraction":float((out.diagnosis_status=="ABSTAIN").mean()),
                 "prohibited_claims":["portfolio union does not inherit a new formal coverage guarantee","class set is not physical root-cause probability"]}
        if true_label_column and true_label_column in out:
            covered=[str(y) in _parse_set(s) for y,s in zip(out[true_label_column],out.diagnosis_set)]
            summary["empirical_set_coverage_all_rows"]=float(np.mean(covered))
            supported_mask=(out.diagnosis_status=="SUPPORTED") | (out.diagnosis_status=="ALTERNATIVES")
            if supported_mask.any():
                sc=[c for c,m in zip(covered,supported_mask) if bool(m)]
                summary["empirical_set_coverage_supported_rows"]=float(np.mean(sc))
                summary["mean_prediction_set_size_supported_rows"]=float(out.loc[supported_mask,"diagnosis_set_size"].mean())
        Path(summary_json).write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return op
