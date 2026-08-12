"""Minimal API example: fixed partial physics, recover one missing term."""
import numpy as np
from isgn.contracts import PublicDataset
from isgn.mechanism_recovery import recover_missing_terms


def make(seed, n=8, T=251, dt=.02):
    rng=np.random.default_rng(seed); X=np.zeros((n,T,2)); U=np.zeros((n,T-1,1))
    for k in range(n):
        X[k,0]=[rng.uniform(-1.2,1.2),rng.uniform(-.4,.4)]
        t=np.arange(T-1)*dt; u=.6*np.sin(.8*t+rng.uniform(0,6.28)); U[k,:,0]=u
        def f(z,q): return np.array([z[1],-3*np.sin(z[0])-.35*z[1]+q])
        for j in range(T-1):
            z=X[k,j]; h=dt; q=u[j]; k1=f(z,q);k2=f(z+.5*h*k1,q);k3=f(z+.5*h*k2,q);k4=f(z+h*k3,q);X[k,j+1]=z+h*(k1+2*k2+2*k3+k4)/6
    return PublicDataset(X,U,np.arange(n),{'states':[{'name':'theta','dim':'angle'},{'name':'omega','dim':'speed'}],'inputs':[{'name':'u','dim':'acceleration'}]},dt,'example')

train, val = make(1), make(2,n=4)
result = recover_missing_terms(
    train, val, target='omega',
    known_terms={'sin(theta)':-3.0,'u':1.0},
    candidate_terms=['theta','omega','cos(theta)','square(theta)','product(theta,omega)'],
)
print(result.to_dict())
