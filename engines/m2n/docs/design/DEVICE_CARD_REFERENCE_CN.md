# Device Card v1.0 参考

Device Card 是 M2N v1.4 的主要用户输入。格式为 JSON。

## 顶层字段

```json
{
  "schema_version": "1.0",
  "template": "second_order_mechanics",
  "goal": {...},
  "data": {...},
  "mechanism": {...},
  "knowledge": {...},
  "supervision": {...},
  "qualification": {...},
  "training": {...},
  "selection": {...}
}
```

## `template`

当前支持：

- `second_order_mechanics`
- `separable_peak_response`

模板不是设备类型标签，而是当前已实现的通用机理骨架。

## `knowledge`

使用 K0/K1/K2/K3 描述局部 slot，而不是整个设备。

K2 示例：

```json
{
  "resistance": {
    "level": "K2",
    "finite_candidates": ["viscous", "viscous_coulomb", "stribeck"],
    "knowledge_source": "FINITE_ENGINEERING_HYPOTHESIS"
  }
}
```

如果已知机理是为了盲测而故意隐藏：

```json
"knowledge_source": "BLINDED_FOR_VALIDATION"
```

## `supervision`

机械模板可直接提供加速度列：

```json
{
  "position_column": "position",
  "velocity_column": "velocity",
  "input_columns": {"u": "command"},
  "target_acceleration_column": "acceleration"
}
```

也可以仅提供时间和速度，让 Estimation Contract 派生加速度：

```json
{
  "position_column": "position",
  "velocity_column": "velocity",
  "input_columns": {"u": "command"},
  "derive_acceleration": {
    "time_column": "time",
    "method": "savgol",
    "window_length": 11,
    "polyorder": 3
  }
}
```

派生量会被写入 `qualification.json`，不会伪装成真实传感器通道。

## 输出

`m2n design` 生成：

```text
architecture_plan.json
qualification.json
candidate_space.json
network_blueprints.json
design_result.json
DESIGN_REPORT.md
models/*.pt
reproduce.txt
```

`network_blueprints.json` 会给出实际模块类型、总参数和可训练参数数目。

## v2.0+ `generic_mechanism_graph`

Device Card 也支持 `template: "generic_mechanism_graph"`，用 `mechanism.inputs / nodes / outputs` 声明 CanonicalIR 图，并通过 `knowledge.slots` 与 `bindings` 限定哪些局部节点允许 K2/K3 变化。

## v2.2 `physics_biases`

```json
"physics_biases": [
  {"id":"mass_balance","knowledge_form":"EXACT_CONSERVATION","trust":"HIGH"},
  {"id":"degradation","knowledge_form":"MONOTONICITY","trust":"MEDIUM"}
]
```

M2N 输出 `bias_route_plan.json`，将知识显式区分为 observational / inductive-hard / learning-soft / hybrid。

## v2.2 `task_semantics` / `representation_semantics`

任务级：

```json
"task_semantics": {
  "representation_kind":"FUNCTION_TO_FUNCTION_OPERATOR",
  "multiple_input_functions":true,
  "consistent_input_sensors":true,
  "output_query_coordinate":true
}
```

或者只对某个可 K3 的 slot 声明：

```json
{
  "id":"unknown_dynamics",
  "level":"K2",
  "allow_k3_escape":true,
  "representation_semantics":"DIFFERENTIAL_DYNAMICS",
  "representation_context":{"continuous_time":true,"solver_compatible":true}
}
```

K-level 与 representation 是两条不同轴：K3 只说明知识不足，不意味着一定使用 MLP。

## v2.2 新增设计输出

```text
bias_route_plan.json
representation_plan.json
exact_physics_contract.json
candidate_exact_physics_contracts.json
```

如果 K3 启用 minimal augmentation，还会在 `design_result.json` 中出现 `k3_training_policy` 和 `k3_credit_assignment_audit`。
