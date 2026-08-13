import math

import pytest
import torch
from torch import nn

from m2n import (
    CompiledVectorField,
    HybridUDE,
    SymbolicCompilationError,
    SymbolicDomainError,
    SymbolicParameter,
    assess_representation,
    compile_symbolic_expression,
    integrate_ude,
)


def test_safe_compiler_matches_formula_and_exact_gradient():
    module = compile_symbolic_expression(
        "k * x + sin(u)", ("x", "u"), {"k": SymbolicParameter(2.0, trainable=True)}
    )
    x = torch.tensor([1.0, 3.0], requires_grad=True)
    u = torch.tensor([0.0, math.pi / 2])
    y = module(x=x, u=u)
    assert torch.allclose(y, torch.tensor([2.0, 7.0]), atol=1e-6)
    y.sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, 2.0))
    assert torch.allclose(module.trainable_parameters["k"].grad, torch.tensor(4.0))
    assert module.contract.physical_truth_claim == "NOT_ESTABLISHED_BY_COMPILATION"


def test_compiler_rejects_python_escape_and_enforces_domain_policy():
    with pytest.raises(SymbolicCompilationError):
        compile_symbolic_expression("x.__class__", ("x",))
    strict = compile_symbolic_expression("log(x) + 1 / x", ("x",))
    with pytest.raises(SymbolicDomainError):
        strict(x=torch.tensor([0.0]))
    guarded = compile_symbolic_expression("log(x) + 1 / x", ("x",), domain_policy="guarded")
    assert torch.isfinite(guarded(x=torch.tensor([0.0]))).all()
    assert guarded.contract.guarded_semantics is True


def test_compiled_harmonic_oscillator_rk4_and_parameter_gradient():
    params = {"k": SymbolicParameter(1.0, trainable=True)}
    dx = compile_symbolic_expression("v", ("t", "x", "v"), params)
    dv = compile_symbolic_expression("-k * x", ("t", "x", "v"), params)
    field = CompiledVectorField((dx, dv), ("x", "v"))
    model = HybridUDE(field)
    times = torch.linspace(0.0, 1.0, 101)
    trajectory = integrate_ude(model, torch.tensor([1.0, 0.0]), times)
    assert trajectory.shape == (101, 2)
    assert torch.allclose(trajectory[-1], torch.tensor([math.cos(1.0), -math.sin(1.0)]), atol=2e-5)
    trajectory[-1, 0].backward()
    assert field.equations[1].trainable_parameters["k"].grad is not None


def test_partial_neural_correction_and_runtime_qualification():
    zero_x = compile_symbolic_expression("0 * x", ("t", "x", "v"))
    zero_v = compile_symbolic_expression("0 * v", ("t", "x", "v"))
    field = CompiledVectorField((zero_x, zero_v), ("x", "v"))
    correction = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        correction.weight[:] = torch.tensor([[0.0, -1.0]])
    model = HybridUDE(field, correction, correction_indices=(1,))
    out = model(torch.tensor(0.0), torch.tensor([2.0, 3.0]))
    assert torch.allclose(out, torch.tensor([0.0, -3.0]))
    eligible = assess_representation("DIFFERENTIAL_DYNAMICS", {"continuous_time": True, "solver_compatible": True})
    assert eligible.runtime_status == "NATIVE_FIXED_STEP_RK4"
