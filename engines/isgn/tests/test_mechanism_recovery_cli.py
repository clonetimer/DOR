import json, subprocess, sys
from pathlib import Path
import numpy as np
from isgn.contracts import PublicDataset
from isgn.io import save_npz


def _dataset(seed=0,n=4,T=121,dt=.02):
    rng=np.random.default_rng(seed); X=np.zeros((n,T,2)); U=np.zeros((n,T-1,1))
    for k in range(n):
        th=float(rng.uniform(-1,1)); w=float(rng.uniform(-.3,.3)); t=np.arange(T-1)*dt; u=.5*np.sin(.8*t+rng.uniform(0,6.28))
        X[k,0]=[th,w]; U[k,:,0]=u
        def f(z,uu): return np.array([z[1],-3*np.sin(z[0])-.35*z[1]+uu])
        for j in range(T-1):
            z=X[k,j]; h=dt; q=u[j]; k1=f(z,q);k2=f(z+.5*h*k1,q);k3=f(z+.5*h*k2,q);k4=f(z+h*k3,q);X[k,j+1]=z+h*(k1+2*k2+2*k3+k4)/6
    return PublicDataset(X,U,np.arange(n),{'states':[{'name':'theta','dim':'angle'},{'name':'omega','dim':'speed'}],'inputs':[{'name':'u','dim':'acceleration'}]},dt,'test')

def test_recover_missing_cli(tmp_path):
    tr=tmp_path/'tr.npz'; va=tmp_path/'va.npz'; out=tmp_path/'out.json'
    save_npz(_dataset(1),tr); save_npz(_dataset(2),va)
    cmd=[sys.executable,'-m','isgn.cli','recover-missing','--train',str(tr),'--val',str(va),'--target','omega','--known-terms',json.dumps({'sin(theta)':-3.0,'u':1.0}),'--candidates',json.dumps(['theta','omega','cos(theta)','square(theta)']),'--out',str(out)]
    p=subprocess.run(cmd,capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    d=json.loads(out.read_text()); assert d['selected_missing_terms']==['omega']; assert abs(d['missing_coefficients']['omega']+.35)<.02
