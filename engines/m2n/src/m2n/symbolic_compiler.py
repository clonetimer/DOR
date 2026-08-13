from __future__ import annotations

"""Safe symbolic-expression lowering to differentiable PyTorch modules.

The compiler intentionally supports a small, auditable expression language.
It is not a Python evaluator and never executes attributes, indexing, imports,
comprehensions, mutation, control flow, or arbitrary calls.  Compilation
exactness is relative to the declared floating-point/domain semantics; it does
not establish that the supplied expression is physically true.
"""

import ast
import hashlib
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import torch
from torch import nn


_UNARY_FUNCTIONS = {
    "sin": torch.sin,
    "cos": torch.cos,
    "tanh": torch.tanh,
    "exp": torch.exp,
    "abs": torch.abs,
    "sign": torch.sign,
}
_BINARY_FUNCTIONS = {
    "minimum": torch.minimum,
    "maximum": torch.maximum,
}
_ALLOWED_BINARY = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_ALLOWED_UNARY = (ast.UAdd, ast.USub)


@dataclass(frozen=True)
class SymbolicParameter:
    value: float
    trainable: bool = False


@dataclass(frozen=True)
class SymbolicCompilationContract:
    expression: str
    expression_sha256: str
    input_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    trainable_parameter_names: tuple[str, ...]
    operations: tuple[str, ...]
    domain_policy: str
    epsilon: float
    guarded_semantics: bool
    compiled_semantics_exactness: str = "WITH_RESPECT_TO_DECLARED_FLOATING_POINT_AND_DOMAIN_SEMANTICS"
    physical_truth_claim: str = "NOT_ESTABLISHED_BY_COMPILATION"

    def to_dict(self) -> dict:
        return asdict(self)


class SymbolicCompilationError(ValueError):
    pass


class SymbolicDomainError(RuntimeError):
    pass


def _parse(expression: str, names: set[str], max_operations: int) -> tuple[ast.Expression, tuple[str, ...]]:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise SymbolicCompilationError(f"invalid symbolic expression: {exc.msg}") from exc
    operations: list[str] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.Expression):
            visit(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise SymbolicCompilationError("only finite real numeric literals are allowed")
            if not torch.isfinite(torch.tensor(float(node.value))):
                raise SymbolicCompilationError("numeric literals must be finite")
        elif isinstance(node, ast.Name):
            if node.id not in names:
                raise SymbolicCompilationError(f"unknown symbol {node.id!r}")
        elif isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINARY):
            operations.append(type(node.op).__name__)
            visit(node.left); visit(node.right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARY):
            operations.append(type(node.op).__name__)
            visit(node.operand)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            expected = 1 if name in _UNARY_FUNCTIONS or name in {"log", "sqrt"} else 2 if name in _BINARY_FUNCTIONS else None
            if expected is None:
                raise SymbolicCompilationError(f"function {name!r} is not allowed")
            if node.keywords or len(node.args) != expected:
                raise SymbolicCompilationError(f"function {name!r} requires exactly {expected} positional argument(s)")
            operations.append(name)
            for arg in node.args:
                visit(arg)
        else:
            raise SymbolicCompilationError(f"unsupported syntax: {type(node).__name__}")
        if len(operations) > max_operations:
            raise SymbolicCompilationError(f"expression exceeds max_operations={max_operations}")

    visit(tree)
    return tree, tuple(operations)


class CompiledSymbolicModule(nn.Module):
    """A frozen-structure, differentiable scalar/tensor symbolic program."""

    def __init__(
        self,
        expression: str,
        input_names: Sequence[str],
        parameters: Mapping[str, SymbolicParameter | float] | None = None,
        *,
        domain_policy: str = "strict",
        epsilon: float = 1e-8,
        max_operations: int = 128,
    ) -> None:
        super().__init__()
        inputs = tuple(str(x) for x in input_names)
        if not inputs or len(set(inputs)) != len(inputs) or any(not x.isidentifier() for x in inputs):
            raise SymbolicCompilationError("input_names must be unique non-empty Python identifiers")
        raw_parameters = dict(parameters or {})
        if set(inputs) & set(raw_parameters):
            raise SymbolicCompilationError("input and parameter names must be disjoint")
        if any(not str(x).isidentifier() for x in raw_parameters):
            raise SymbolicCompilationError("parameter names must be Python identifiers")
        if domain_policy not in {"strict", "guarded"}:
            raise SymbolicCompilationError("domain_policy must be 'strict' or 'guarded'")
        if epsilon <= 0:
            raise SymbolicCompilationError("epsilon must be positive")
        self.expression = str(expression).strip()
        self.input_names = inputs
        self.domain_policy = domain_policy
        self.epsilon = float(epsilon)
        self._tree, operations = _parse(self.expression, set(inputs) | set(raw_parameters), int(max_operations))
        self.trainable_parameters = nn.ParameterDict()
        self._frozen_names: list[str] = []
        trainable: list[str] = []
        for name, raw in raw_parameters.items():
            spec = raw if isinstance(raw, SymbolicParameter) else SymbolicParameter(float(raw), False)
            value = torch.tensor(float(spec.value), dtype=torch.get_default_dtype())
            if not bool(torch.isfinite(value)):
                raise SymbolicCompilationError(f"parameter {name!r} must be finite")
            if spec.trainable:
                self.trainable_parameters[name] = nn.Parameter(value)
                trainable.append(name)
            else:
                self.register_buffer(f"_constant_{name}", value)
                self._frozen_names.append(name)
        self.contract = SymbolicCompilationContract(
            self.expression,
            hashlib.sha256(self.expression.encode("utf-8")).hexdigest(),
            inputs,
            tuple(raw_parameters),
            tuple(trainable),
            operations,
            domain_policy,
            self.epsilon,
            domain_policy == "guarded" and any(x in operations for x in ("Div", "log", "sqrt")),
        )

    def _environment(self, values: Mapping[str, torch.Tensor | float]) -> dict[str, torch.Tensor]:
        missing = sorted(set(self.input_names) - set(values))
        extra = sorted(set(values) - set(self.input_names))
        if missing or extra:
            raise ValueError(f"symbolic inputs mismatch; missing={missing}, extra={extra}")
        tensor_inputs = [v for v in values.values() if isinstance(v, torch.Tensor)]
        if tensor_inputs:
            ref = tensor_inputs[0]
            env = {name: torch.as_tensor(values[name], dtype=ref.dtype, device=ref.device) for name in self.input_names}
        else:
            env = {name: torch.as_tensor(values[name], dtype=torch.get_default_dtype()) for name in self.input_names}
        for name in self._frozen_names:
            env[name] = getattr(self, f"_constant_{name}").to(dtype=next(iter(env.values())).dtype, device=next(iter(env.values())).device)
        for name, value in self.trainable_parameters.items():
            env[name] = value.to(dtype=next(iter(env.values())).dtype, device=next(iter(env.values())).device)
        return env

    def _domain(self, value: torch.Tensor, operation: str) -> torch.Tensor:
        if operation == "division":
            invalid = torch.abs(value) <= self.epsilon
            if self.domain_policy == "guarded":
                sign = torch.where(value < 0, -torch.ones_like(value), torch.ones_like(value))
                return torch.where(invalid, sign * self.epsilon, value)
        elif operation == "log":
            invalid = value <= 0
            if self.domain_policy == "guarded":
                return torch.clamp_min(value, self.epsilon)
        elif operation == "sqrt":
            invalid = value < 0
            if self.domain_policy == "guarded":
                return torch.clamp_min(value, 0.0)
        else:  # pragma: no cover - internal invariant
            raise AssertionError(operation)
        if bool(torch.any(invalid).detach().cpu()):
            raise SymbolicDomainError(f"{operation} received values outside its strict safe domain")
        return value

    def _eval(self, node: ast.AST, env: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(node, ast.Constant):
            ref = next(iter(env.values()))
            return torch.as_tensor(float(node.value), dtype=ref.dtype, device=ref.device)
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, ast.UnaryOp):
            value = self._eval(node.operand, env)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = self._eval(node.left, env), self._eval(node.right, env)
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, ast.Div): return left / self._domain(right, "division")
            if isinstance(node.op, ast.Pow): return torch.pow(left, right)
        if isinstance(node, ast.Call):
            name = node.func.id
            args = [self._eval(x, env) for x in node.args]
            if name == "log": return torch.log(self._domain(args[0], "log"))
            if name == "sqrt": return torch.sqrt(self._domain(args[0], "sqrt"))
            if name in _UNARY_FUNCTIONS: return _UNARY_FUNCTIONS[name](args[0])
            if name in _BINARY_FUNCTIONS: return _BINARY_FUNCTIONS[name](args[0], args[1])
        raise AssertionError(type(node).__name__)  # parser makes this unreachable

    def forward(self, **inputs: torch.Tensor | float) -> torch.Tensor:
        out = self._eval(self._tree.body, self._environment(inputs))
        if not bool(torch.all(torch.isfinite(out)).detach().cpu()):
            raise SymbolicDomainError("symbolic program produced non-finite output")
        return out


def compile_symbolic_expression(
    expression: str,
    input_names: Sequence[str],
    parameters: Mapping[str, SymbolicParameter | float] | None = None,
    **kwargs,
) -> CompiledSymbolicModule:
    return CompiledSymbolicModule(expression, input_names, parameters, **kwargs)


class SymbolicResidualModel(nn.Module):
    """Compose an exact declared symbolic path with a learned additive residual."""

    def __init__(self, symbolic: CompiledSymbolicModule, residual: nn.Module) -> None:
        super().__init__()
        self.symbolic = symbolic
        self.residual = residual

    def forward(self, feature_tensor: torch.Tensor, **symbolic_inputs: torch.Tensor) -> torch.Tensor:
        known = self.symbolic(**symbolic_inputs)
        learned = self.residual(feature_tensor)
        if known.shape != learned.shape:
            raise ValueError(f"symbolic/residual shape mismatch: {tuple(known.shape)} != {tuple(learned.shape)}")
        return known + learned
