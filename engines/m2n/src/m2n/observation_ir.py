from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class ObservationContract:
    state_names: tuple[str,...]
    observed_names: tuple[str,...]
    initialization_window: int = 10
    min_observability_sv: float = 1e-3
    def __post_init__(self):
        s=set(self.state_names);o=set(self.observed_names)
        if not o.issubset(s):raise ValueError('observed state not in state_names')
        if len(o)!=len(self.observed_names):raise ValueError('duplicate observed state')
    @property
    def hidden_names(self):return tuple(x for x in self.state_names if x not in set(self.observed_names))

class ObservedStateModel:
    def __init__(self,state_model,contract:ObservationContract):
        self.model=state_model;self.contract=contract
        self.idx={n:i for i,n in enumerate(contract.state_names)}
        self.obs_idx=[self.idx[n] for n in contract.observed_names]
        self.hid_idx=[self.idx[n] for n in contract.hidden_names]
    def observe(self,state):return state[:,self.obs_idx]
    def rollout(self,state0,inputs_seq):
        z=state0;zs=[z]
        T=next(iter(inputs_seq.values())).shape[0]
        for t in range(T):
            kw={k:v[t] for k,v in inputs_seq.items()};z=self.model(z,**kw);zs.append(z)
        return torch.stack(zs,dim=0)
    def _state0(self,obs0,hidden):
        # batch=1 initializer protocol
        vals=[];hi=0;oi={idx:j for j,idx in enumerate(self.obs_idx)}
        for i in range(len(self.contract.state_names)):
            if i in oi:vals.append(obs0[:,oi[i]])
            else:vals.append(hidden[:,hi]);hi+=1
        return torch.stack(vals,dim=1)
    def observability_singular_values(self,observations,inputs_seq):
        H=min(self.contract.initialization_window,observations.shape[0]);nh=len(self.hid_idx)
        if nh==0:return torch.tensor([float('inf')])
        h=torch.zeros((1,nh),dtype=observations.dtype,requires_grad=True)
        def f(hv):
            z0=self._state0(observations[0],hv)
            z=self.rollout(z0,{k:v[:H-1] for k,v in inputs_seq.items()})
            return self.observe(z[:,0,:]).reshape(-1)
        J=torch.autograd.functional.jacobian(f,h,vectorize=True).reshape(-1,nh)
        return torch.linalg.svdvals(J)
    def initialize(self,observations,inputs_seq,steps=120,lr=.08):
        H=min(self.contract.initialization_window,observations.shape[0]);nh=len(self.hid_idx)
        if nh==0:
            return {'state0':self._state0(observations[0],torch.empty((1,0),dtype=observations.dtype)),'supported':True,'min_sv':float('inf'),'loss':0.}
        sv=self.observability_singular_values(observations[:H],{k:v[:H-1] for k,v in inputs_seq.items()});minsv=float(torch.min(sv))
        h=torch.zeros((1,nh),dtype=observations.dtype,requires_grad=True);opt=torch.optim.Adam([h],lr=lr)
        target=observations[:H]; target2=target[:,0,:]
        for _ in range(steps):
            opt.zero_grad();z0=self._state0(target[0],h);z=self.rollout(z0,{k:v[:H-1] for k,v in inputs_seq.items()});pred=self.observe(z[:,0,:]);loss=((pred-target2)**2).mean();loss.backward();opt.step()
        z0=self._state0(target[0],h.detach());return {'state0':z0,'supported':minsv>=self.contract.min_observability_sv,'min_sv':minsv,'loss':float(loss.detach())}
