from m2n import ModelComponentSlot, SBMIModelSetRequest, assess_sbmi_feasibility

small=SBMIModelSetRequest(slots=(
    ModelComponentSlot('pressure_path',('direct','line_drop')),
    ModelComponentSlot('friction',('viscous','coulomb','stribeck')),
))
print('EHSS-like small set:', assess_sbmi_feasibility(small))

large=SBMIModelSetRequest(slots=tuple(
    ModelComponentSlot(f'slot_{i}',('a','b','c')) for i in range(4)
))
print('81-model family:', assess_sbmi_feasibility(large))
