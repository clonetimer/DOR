import numpy as np
from isgn.contracts import PublicDataset
from isgn.mechanism_recovery import evaluate_term, recover_missing_terms


def _pendulum(seed=1,n=10,T=301,dt=.02,gL=3.0,b=.35):
    rng=np.random.RandomState(seed); X=np.zeros((n,T,2)); U=np.zeros((n,T-1,1))
    for k in range(n):
        th=rng.uniform(-1.5,1.5); w=rng.uniform(-.5,.5); ph=rng.uniform(0,2*np.pi,3)
        tt=np.arange(T-1)*dt; u=.7*np.sin(.6*tt+ph[0])+.3*np.sin(1.7*tt+ph[1])+.15*np.sin(2.9*tt+ph[2])
        X[k,0]=[th,w]; U[k,:,0]=u
        def f(z,uu): return np.array([z[1],-gL*np.sin(z[0])-b*z[1]+uu])
        for j in range(T-1):
            z=X[k,j]; h=dt; uu=u[j]
            k1=f(z,uu); k2=f(z+.5*h*k1,uu); k3=f(z+.5*h*k2,uu); k4=f(z+h*k3,uu)
            X[k,j+1]=z+h*(k1+2*k2+2*k3+k4)/6
    schema={'states':[{'name':'theta','dim':'angle'},{'name':'omega','dim':'speed'}],
            'inputs':[{'name':'u','dim':'acceleration'}]}
    return PublicDataset(X,U,np.arange(n),schema,dt,'benchmark')


def test_term_library():
    ds=_pendulum(n=2,T=40)
    assert np.allclose(evaluate_term(ds,'square(theta)'),evaluate_term(ds,'theta')**2)
    assert np.allclose(evaluate_term(ds,'product(theta,omega)'),evaluate_term(ds,'theta')*evaluate_term(ds,'omega'))


def test_recover_missing_damping_with_known_gravity():
    tr=_pendulum(seed=1); va=_pendulum(seed=2,n=4)
    r=recover_missing_terms(tr,va,target='omega',known_terms={'sin(theta)':-3.0,'u':1.0},
                            candidate_terms=['theta','omega','cos(theta)','square(theta)','square(omega)','product(theta,omega)'])
    assert r.selected_missing_terms == ['omega']
    assert abs(r.missing_coefficients['omega'] + .35) < .01
