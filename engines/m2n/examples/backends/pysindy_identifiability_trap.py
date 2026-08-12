"""Why a sparse equation proposal is not automatically physical truth.

Here y=0.5*x^2 exactly along the single observed trajectory.  Polynomial
libraries therefore contain collinear terms.  Different coefficient
representations can fit the same trajectory, illustrating why M2N keeps
external symbolic output UNADMITTED until identifiability/physics checks.
"""
import numpy as np
from m2n import DynamicsDiscoveryRequest, PySINDyCandidateBackend, BackendUnavailableError

t=np.linspace(0,2,401)
x=np.column_stack([np.exp(-t),0.5*np.exp(-2*t)])
xd=np.column_stack([-np.exp(-t),-1.0*np.exp(-2*t)])
req=DynamicsDiscoveryRequest(x=x,x_dot=xd,t=t,feature_names=('x','y'),polynomial_degree=2,threshold=0.02,alpha=1e-6)
try:
    batch=PySINDyCandidateBackend.propose(req)
except BackendUnavailableError as e:
    print(e)
    raise SystemExit(0)
for p in batch.proposals:
    print(p.target,'=',p.expression)
print('NOTE: y=0.5*x^2 on this trajectory, so polynomial terms are structurally confounded.')
print('M2N action: keep proposals UNADMITTED until richer excitation / physical admission resolves the ambiguity.')
