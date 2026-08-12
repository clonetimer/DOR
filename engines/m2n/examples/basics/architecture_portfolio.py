"""When K2 remains ambiguous, keep candidate identity and expose model-form spread."""
import torch
from torch import nn
from m2n import ArchitecturePortfolio

class Gain(nn.Module):
    def __init__(self,g): super().__init__(); self.g=float(g)
    def forward(self,x): return x*self.g

portfolio=ArchitecturePortfolio(
    {'candidate_low':Gain(0.9),'candidate_high':Gain(1.1)},
    unresolved_slots=('local_gain_family',),
)
x=torch.tensor([1.0,2.0,3.0])
out=portfolio(x)
print('candidate outputs:')
for k,v in portfolio.forward_all(x).items(): print(' ',k,v.tolist())
print('mean:',out['mean'].tolist())
print('std :',out['std'].tolist())
print('interpretation:',portfolio.contract.interpretation)
