#!/usr/bin/env python3
from __future__ import annotations
from copy import deepcopy
from typing import Dict,Tuple
import numpy as np
from ..simulation import PublicDataset,PrivateTruth,SYSTEM_SCHEMAS,PARAMS,_input_trajectory,_initial_state,_rhs,private_truth

DYNAMIC_TRUTH={
 'thermal':('FirstOrderLag',{'tau':0.6},0),
 'tank':('Delay',{'steps':4.0},0),
 'motor':('SecondOrderOscillator',{'wn':1.4,'zeta':0.35},0),
 'msd':('Identity',{},0),
}

def _schema(system):
 s=deepcopy(SYSTEM_SCHEMAS[system])
 # Only the actuator/command path is dynamic-searchable. Other exogenous quantities are contexts.
 if system=='tank': s['inputs'][1]['role']='context'
 if system=='motor': s['inputs'][1]['role']='context'
 return s

def simulate_dynamic(system,n,T=201,dt=.05,seed=0,regime='iid',substeps=5)->Tuple[PublicDataset,PrivateTruth]:
 rng=np.random.RandomState(seed); schema=_schema(system); D=len(schema['states']); Ud=len(schema['inputs']); X=np.zeros((n,T,D),np.float32); U=np.zeros((n,T-1,Ud),np.float32)
 op,pars,idx=DYNAMIC_TRUTH[system]; tt=np.arange(T-1)*dt
 for k in range(n):
  uk=_input_trajectory(system,tt,rng,regime,'clean'); x=_initial_state(system,rng,uk[0],regime,'clean'); X[k,0]=x; U[k]=uk
  # latent states initialized at the first public command to minimize arbitrary startup mismatch.
  z=float(uk[0,idx]); zd=0.0; delay=[float(uk[0,idx])]*8; h=dt/substeps
  for j in range(T-1):
   cmd=float(uk[j,idx]);
   if op=='Delay':
    d=int(pars['steps']); delay=delay[1:]+[cmd]; phys=delay[-(d+1)]
    up=uk[j].copy(); up[idx]=phys
    # physical state RK4, command held on sample interval
    for _ in range(substeps):
     k1=_rhs(system,x,up,'clean'); k2=_rhs(system,x+.5*h*k1,up,'clean'); k3=_rhs(system,x+.5*h*k2,up,'clean'); k4=_rhs(system,x+h*k3,up,'clean'); x=x+h*(k1+2*k2+2*k3+k4)/6
   else:
    for _ in range(substeps):
     def latent(q,v):
      if op=='Identity': return 0.0,0.0,cmd
      if op=='FirstOrderLag': return (cmd-q)/pars['tau'],0.0,q
      return v, pars['wn']**2*(cmd-q)-2*pars['zeta']*pars['wn']*v, q
     # augmented RK4 for x + latent q/v
     q0=z; v0=zd; lq1,lv1,phys1=latent(q0,v0); up1=uk[j].copy(); up1[idx]=phys1; k1=_rhs(system,x,up1,'clean')
     q2=q0+.5*h*lq1; v2=v0+.5*h*lv1; lq2,lv2,phys2=latent(q2,v2); up2=uk[j].copy(); up2[idx]=phys2; k2=_rhs(system,x+.5*h*k1,up2,'clean')
     q3=q0+.5*h*lq2; v3=v0+.5*h*lv2; lq3,lv3,phys3=latent(q3,v3); up3=uk[j].copy(); up3[idx]=phys3; k3=_rhs(system,x+.5*h*k2,up3,'clean')
     q4=q0+h*lq3; v4=v0+h*lv3; lq4,lv4,phys4=latent(q4,v4); up4=uk[j].copy(); up4[idx]=phys4; k4=_rhs(system,x+h*k3,up4,'clean')
     x=x+h*(k1+2*k2+2*k3+k4)/6; z=q0+h*(lq1+2*lq2+2*lq3+lq4)/6; zd=v0+h*(lv1+2*lv2+2*lv3+lv4)/6
   X[k,j+1]=x
 pub=PublicDataset(X=X,U=U,trajectory_id=np.arange(n),schema=schema,dt=dt,regime=regime,variant='dynamic_joint')
 truth=private_truth(system,'clean'); truth.modes=dict(truth.modes); truth.modes.update({'dynamic_op':op,**pars}); truth.variant='dynamic_joint'
 return pub,truth
