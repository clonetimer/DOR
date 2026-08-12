"""K1 example: topology is known, continuous parameters remain learnable."""
import torch
from m2n import Dim, CoordinateSpec, InputSpec, DriveSpec, ResistanceSpec, SecondOrderMechanicsIR, compile_second_order_ir

L=Dim.of(length=1)
T=Dim.of(time=1)
A=L/(T**2)

ir=SecondOrderMechanicsIR(
    coordinate=CoordinateSpec('q','v',L),
    inputs=(InputSpec('command_accel',A),),
    drive=DriveSpec('actuator','command_accel',Dim(),init=1.2),
    resistance=ResistanceSpec('drag','viscous',viscous_gain_init=0.25),
    dt=0.01,
)
model,canonical=compile_second_order_ir(ir)
state=torch.tensor([[0.0,0.5],[0.2,-0.3]],dtype=torch.float32)
u=torch.tensor([0.8,-0.2],dtype=torch.float32)
print('IR source      :',dict(canonical.metadata)['source_ir'])
print('trainable count:',sum(p.numel() for p in model.parameters() if p.requires_grad))
print('rate [qdot,vdot]:')
print(model.rate(state,command_accel=u))
