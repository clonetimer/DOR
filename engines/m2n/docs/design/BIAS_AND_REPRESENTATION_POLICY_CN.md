# M2N v2.2：Bias Route 与 Representation Semantics

## 1. 为什么要增加这两层

K0/K1/K2/K3 描述的是**认识论状态**：一个机理槽究竟知道多少。它并不能回答两个不同问题：

1. 一条物理知识应该进入数据、网络结构，还是训练约束？
2. K3 中真正未知的对象到底是局部函数、微分动力学、Hamiltonian，还是函数到函数的算子？

v2.2 因此没有增加新的 K-level，而增加两条与 K-level 正交的 policy axis。

## 2. BiasRoutePolicy

`physics_biases` 中的知识被路由为：

- `OBSERVATIONAL`：虚拟传感器、仿真数据、数据增强、实验设计、融合等；
- `INDUCTIVE_HARD`：高可信精确方程、精确守恒、精确边界/对称性等，进入 executable IR/architecture；
- `LEARNING_SOFT`：近似方程、单调性、不可逆性、经验趋势、软守恒等，作为训练约束；
- `HYBRID`：物理模型明确不完整，保留可信核心并让缺失部分进入后续 qualification/learning。

示例：

```json
{
  "physics_biases": [
    {"id":"mass_balance","knowledge_form":"EXACT_CONSERVATION","trust":"HIGH"},
    {"id":"degradation","knowledge_form":"MONOTONICITY","trust":"MEDIUM"},
    {"id":"virtual_temperature","knowledge_form":"VIRTUAL_SENSOR","trust":"MEDIUM"}
  ]
}
```

设计会输出 `bias_route_plan.json`。当前它是**显式路由合同**，不是任意物理 loss 的万能自动执行器；只有 M2N 已支持的 hard/learning/observational action 才会被后续 runtime 实施。

## 3. Representation Semantics

v2.2 定义：

- `LOCAL_FUNCTION` → M2N native constrained local NN；
- `DIFFERENTIAL_DYNAMICS` → UDE-style solver-aware hybrid dynamics；
- `CONSERVATIVE_HAMILTONIAN` → HNN-style conservative architecture；
- `FUNCTION_TO_FUNCTION_OPERATOR` → DeepONet-style branch/trunk operator；
- `REGULAR_GRID_FIELD_OPERATOR` → FNO-style regular-grid field operator；
- `SPATIOTEMPORAL_LATENT_DYNAMICS` → PhyDNet-style latent physical/residual dynamics。

这些名称描述**M2N 学到的架构选择原则**，并不表示工具包 vendoring 或重新实现了相应论文源码。

### HNN eligibility

至少要求：

- canonical `q/p` 可观测或可重建；
- conservative core 有物理依据；
- 主导耗散/外部驱动不存在，或被独立建模。

不满足时 M2N 应拒绝 HNN，而不是因为设备“属于机械系统”就套 HNN。

### DeepONet eligibility

至少要求：多条输入函数、固定/明确的输入 sensor representation，以及输出 query coordinate。

### FNO eligibility

至少要求：重复的 field/PDE operator 任务、规则空间网格、多个参数/初始条件实例。它不是单次 PDE 求解器的默认替代。

### PhyDNet eligibility

用于图像/场序列，且物理规律无法直接施加在原始 observation space、但 latent physical dynamics 假设可信的情况。

## 4. Specialized K3 不再偷偷降级成 local MLP

如果有限 K2 被拒绝，而 Device Card 声明：

```json
"task_semantics": {
  "representation_kind": "DIFFERENTIAL_DYNAMICS",
  "continuous_time": true,
  "solver_compatible": true
}
```

M2N 不会继续把该槽编译成普通 local MLP。当前 native runtime 会返回：

```text
SPECIALIZED_REPRESENTATION_REQUIRED
recommended: UDE_STYLE
```

因此“当前没安装专用 runtime”不会导致架构语义被悄悄改变。

## 5. Exact Physics Contract

设计输出：

- `exact_physics_contract.json`
- `candidate_exact_physics_contracts.json`

它区分：

- frozen/exact IR nodes；
- K1/trainable physics nodes；
- learned neural nodes；
- clamp/epsilon/threshold 等 totalized 或 guarded semantics。

关键声明固定为：

```text
compiled semantics exactness
!=
physical truth
```

即编译器可以精确执行一个工程方程，但无法证明工程方程本身就是真实世界真理。

## 6. Minimal-Augmentation K3

开启：

```json
"k3_policy": {
  "enabled": true,
  "minimal_augmentation": {
    "enabled": true,
    "epochs": 120,
    "fit_slack": 0.05,
    "constraint_weight": 100.0
  }
}
```

后，M2N 先按原 P4 流程获得可行 K3，再进行一个 `MINIMAL_COMPLEMENT_CONSTRAINED_REFINEMENT`：在基本维持可行拟合的前提下，压缩**已经声明的局部 neural complement**。

这是从“最小数据驱动补偿”的思想得到的 M2N 自有训练策略，不声称复现 APHYNITY 的完整约束优化算法或其理论保证。

同时输出：

- `augmentation_nrmse_median`；
- physical parameter restart stability；
- `credit_assignment_warning`。

这些只用于发现 hybrid credit assignment 风险，不能自动证明参数可辨识或具有物理语义。
