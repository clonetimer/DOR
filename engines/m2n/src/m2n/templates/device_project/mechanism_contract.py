"""Edit this file first when onboarding a real device."""
from m2n import (
    Dim, KnowledgeLevel, KnowledgeSource, MechanismSlot, SlotEvidence,
    CoordinateSpec, InputSpec, DriveSpec, ResistanceSpec, SecondOrderMechanicsIR,
    FieldAssignment, FamilyPatch, SlotBinding, plan_neural_architecture,
)

L=Dim.of(length=1)
A=L/(Dim.of(time=1)**2)


def build_contract():
    friction=MechanismSlot(
        id='friction',
        level=KnowledgeLevel.K2,
        role='local mechanical resistance',
        input_dims=(L/Dim.of(time=1),),
        output_dim=A,
        finite_candidates=('viscous','viscous_coulomb'),
        knowledge_source=KnowledgeSource.FINITE_ENGINEERING_HYPOTHESIS,
    )
    evidence={'friction':SlotEvidence(
        data_qualification='READY',
        observation_qualification='SUPPORTED',
        hard_mechanism_status='COMPLETE',
        finite_status='PENDING',
    )}
    return plan_neural_architecture((friction,),evidence)


def build_ir_and_bindings():
    skeleton=SecondOrderMechanicsIR(
        coordinate=CoordinateSpec('q','v',L),
        inputs=(InputSpec('u',A),),
        drive=DriveSpec('actuator','u',Dim(),init=1.0),
        resistance=ResistanceSpec('friction','viscous',viscous_gain_init=0.1,coulomb_gain_init=0.1),
        dt=0.01,
    )
    binding=SlotBinding('friction',(
        FamilyPatch('viscous',(FieldAssignment('resistance.law','viscous'),)),
        FamilyPatch('viscous_coulomb',(FieldAssignment('resistance.law','viscous_coulomb'),)),
    ))
    return skeleton,(binding,)
