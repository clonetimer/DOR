"""Trajectory-level partial-physics recovery for a noisy damped pendulum."""
import numpy as np
from isgn.contracts import PublicDataset
from isgn.trajectory_recovery import recover_missing_dynamics_trajectory

def make(seed,n=8,T=37,dt=.2,noise=.03):
    rng=np.random.RandomState(seed); X=np.zeros((n,T,2)); U=np.zeros((n,T-1,1))
    for k in range(n):
        X[k,0]=[rng.uniform(-1.5,1.5),rng.uniform(-.5,.5)]; tt=np.arange(T-1)*dt; ph=rng.uniform(0,2*np.pi,3)
        U[k,:,0]=.7*np.sin(.6*tt+ph[0])+.3*np.sin(1.7*tt+ph[1])+.15*np.sin(2.9*tt+ph[2])
        def f(z,u): return np.array([z[1],-3*np.sin(z[0])-.35*z[1]+u])
        for j in range(T-1):
            z=X[k,j]; u=U[k,j,0]; h=dt; k1=f(z,u); k2=f(z+.5*h*k1,u); k3=f(z+.5*h*k2,u); k4=f(z+h*k3,u)
            X[k,j+1]=z+h*(k1+2*k2+2*k3+k4)/6
    X += np.random.RandomState(seed+100).normal(0,noise,X.shape)
    schema={'states':[{'name':'theta','dim':'angle'},{'name':'omega','dim':'speed'}],'inputs':[{'name':'u','dim':'acceleration'}]}
    return PublicDataset(X,U,np.arange(n),schema,dt,'example')

tr,va=make(1,10),make(2,4)
r=recover_missing_dynamics_trajectory(tr,va,
    known_terms={'theta':{'omega':1.0},'omega':{'sin(theta)':-3.0,'u':1.0}},
    unknown_targets=['omega'],
    candidate_terms={'omega':['omega','theta','cos(theta)','square(theta)','square(omega)','product(theta,omega)']},
    epochs=150,batch_windows=24,window_steps=8,seed=0)
print(r.to_dict())
