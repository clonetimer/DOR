import json, subprocess, sys
from pathlib import Path
import numpy as np
from isgn.contracts import AlgebraicDataset, PublicDataset
from isgn.io import save_algebraic_npz, save_npz


def test_reduce_physics_cli(tmp_path):
    rng=np.random.default_rng(1); n=500
    p1=rng.uniform(2e6,8e6,n); p2=rng.uniform(1e6,5e6,n); q=rng.uniform(.001,.01,n); P=(p1-p2)*q
    schema={'variables':[{'name':'p1','dim':'pressure'},{'name':'p2','dim':'pressure'},{'name':'q','dim':'volume_flow'},{'name':'P','dim':'power'}]}
    tr=AlgebraicDataset(np.column_stack([p1,p2,q,P]),np.arange(n),schema)
    va=AlgebraicDataset(np.column_stack([p1[:250],p2[:250],q[:250],P[:250]]),np.arange(250),schema)
    a=tmp_path/'tr.npz'; b=tmp_path/'va.npz'; out=tmp_path/'out.json'; save_algebraic_npz(tr,a); save_algebraic_npz(va,b)
    cmd=[sys.executable,'-m','isgn.cli','reduce-physics','--train',str(a),'--val',str(b),'--inputs','p1,p2,q','--target','P','--max-order','2','--max-terms','2','--out',str(out)]
    r=subprocess.run(cmd,capture_output=True,text=True); assert r.returncode==0, r.stderr
    d=json.loads(out.read_text()); assert d['pairwise_reductions'][0]['transform']=='difference'


def test_recover_missing_trajectory_cli(tmp_path):
    dt=.2; T=31; n=6; rng=np.random.RandomState(2); X=np.zeros((n,T,2)); U=np.zeros((n,T-1,1))
    for k in range(n):
        X[k,0]=[rng.uniform(-1,1),rng.uniform(-.4,.4)]; t=np.arange(T-1)*dt; U[k,:,0]=.8*np.sin(.7*t+rng.uniform(0,6))
        for j in range(T-1):
            th,w=X[k,j]; u=U[k,j,0]; X[k,j+1]=[th+dt*w,w+dt*(-3*np.sin(th)-.35*w+u)]
    X += np.random.RandomState(44).normal(0,.015,X.shape)
    schema={'states':[{'name':'theta','dim':'angle'},{'name':'omega','dim':'speed'}],'inputs':[{'name':'u','dim':'acceleration'}]}
    ds=PublicDataset(X,U,np.arange(n),schema,dt,'cli-test'); tr=tmp_path/'tr.npz'; va=tmp_path/'va.npz'; save_npz(ds,tr); save_npz(ds,va); out=tmp_path/'out.json'
    known=json.dumps({'theta':{'omega':1.0},'omega':{'sin(theta)':-3.0,'u':1.0}}); cands=json.dumps({'omega':['omega','theta','cos(theta)']})
    cmd=[sys.executable,'-m','isgn.cli','recover-missing-trajectory','--train',str(tr),'--val',str(va),'--known-terms',known,'--unknown-targets','omega','--candidates',cands,'--epochs','80','--batch-windows','12','--window-steps','6','--out',str(out)]
    r=subprocess.run(cmd,capture_output=True,text=True); assert r.returncode==0, r.stderr
    d=json.loads(out.read_text()); assert d['metadata']['derivatives_used_for_training'] is False; assert 'omega' in d['distilled_terms']
