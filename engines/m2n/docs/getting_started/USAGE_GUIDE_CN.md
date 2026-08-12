# M2N Practitioner Toolkit 完整使用指南

## 1. M2N 的实际定位

M2N 的目标不是“自动给出一个深度网络名字”，而是：

> **把设备已有机理知识、观测条件和正常数据联合起来，生成一个受约束的可训练架构，或者明确返回有限候选集合 / AMBIGUOUS / UNSUPPORTED。**

它特别适合灰箱/机理驱动建模：硬机理保留为计算图骨架，只让真正不确定的局部关系拥有学习自由度。

---

## 2. 第一步：先定义设计任务，不要直接写模型

在任何代码之前，写出 6 个问题：

1. 这次要预测/重构什么？
2. 设备的状态、输入、输出是什么？
3. 哪些守恒、运动学、拓扑、符号方向、边界条件是可信的？
4. 哪些只是参数未知？
5. 哪些存在有限几个工程候选？
6. 哪些变量在真实部署中可测？

建议把设备拆成局部 slot，例如：

```text
pump_map
valve_metering
pressure_path
cylinder_friction
heat_transfer
conversion_efficiency
```

K-level 必须作用在这些**局部 slot**上，而不是给整台设备打一个“K2”标签。

---

## 3. K0/K1/K2/K3 如何编码

### K0：形式与参数都可信

典型例子：

```text
q_dot = v
已知几何连接关系
固定换算系数
```

要求：`exact_family` 必须给出；不能再包含 learnable/K2/K3 自由度。

### K1：形式可信，连续参数未知

例如：

```text
F = c v
c unknown
```

代码思路：

```python
MechanismSlot(
    id="damping",
    level=KnowledgeLevel.K1,
    exact_family="viscous",
    learnable_parameters=("c",),
    ...
)
```

M2N 不搜索结构，只编译该结构并允许参数训练。

### K2：有限物理候选

例如摩擦：

```text
viscous
viscous + Coulomb
Stribeck
```

或局部响应面：

```text
I only
I + target density
I + target density + alignment
```

代码：

```python
MechanismSlot(
    id="friction",
    level=KnowledgeLevel.K2,
    finite_candidates=("viscous", "viscous_coulomb"),
    knowledge_source=KnowledgeSource.FINITE_ENGINEERING_HYPOTHESIS,
    ...
)
```

随后用 `SlotBinding` 把这些名字映射到实际 IR 局部修改。

### K3：局部未知本构

只有当：

- 已知 hard mechanism 完整；
- 观测支持；
- 正常数据激励充分；
- 冻结的有限 K2 已被明确拒绝；
- 有局部 placement 约束；

才考虑 K3。

本 Practitioner Kit 默认鼓励你先完成 K0/K1/K2。不要把“拟合不好”直接翻译成 K3。

---

## 4. KnowledgeSource：区分“现实不知道”与“故意盲测”

这是使用 M2N 时非常容易混淆的地方。

### `DECLARED_ENGINEERING`
来自设计资料、物理定律、设备手册、可信工程模型。

### `FINITE_ENGINEERING_HYPOTHESIS`
现实中确实只有有限候选，适合 K2。

### `COMMISSIONING_EVIDENCE`
来自正式调试/试验阶段获得的知识。

### `BLINDED_FOR_VALIDATION`
**实验者知道真值，但故意不给 M2N。** 用来测试方法能力边界。Planner 会保留提醒，避免以后误写成“现实中未知”。

### `SIMULATOR_TRUTH_POSTHOC`
只能最终核验。传给 Planner 会直接报错。

---

## 5. 第二步：写 Observation Contract

M2N 不要求你拥有仿真器内部全部状态。

你需要区分：

```text
真实部署可测量量
研发台架额外仪表量
仿真 truth
```

其中只有前两者在明确合同下可以进入设计；`truth_*` 默认只能 posthoc。

如果某个局部机理需要变量 `x, y, z` 才能辨识，但生产传感器只有 `x`，正确结果可能是：

```text
UNSUPPORTED
```

这不是 M2N “失效”，而是说明当前观测合同没有资格对那个机理做物理归因。

`ObservationContract` / `plan_observation()` 可用于需要隐藏状态初始化和局部可观测性审查的状态模型。

---

## 6. 第三步：建立 IR skeleton

当前工具包提供的主要 IR：

### `SecondOrderMechanicsIR`
适合：执行器、转动/平动机械支路。

固定外层：

```text
q_dot = v
v_dot = drive - resistance - load
```

局部 resistance 支持：

```text
viscous
viscous_coulomb
asymmetric_viscous_coulomb
stribeck
local_neural
```

### `ConservationIR`
适合守恒网络、储能/流量结构。

### `FourWayValveIR`
适合四边方向阀计量结构。

### `BlockLNLIR`
适合线性动态块—局部静态本构—线性动态块。

### `SeparablePeakIR`
适合多个物理因素共同形成局部峰值响应面，例如转换效率、反应产率、共振包络。

### `CanonicalIR`
底层通用计算图。只有现有高层 IR 不适合时才建议直接使用。

---

## 7. 第四步：用 SlotBinding 把 K2 变成可执行候选

Planner 只知道：

```text
friction ∈ {viscous, viscous_coulomb}
```

它还不知道这些名字如何改变计算图。

因此你需要：

```python
SlotBinding(
    "friction",
    (
        FamilyPatch("viscous", (
            FieldAssignment("resistance.law", "viscous"),
        )),
        FamilyPatch("viscous_coulomb", (
            FieldAssignment("resistance.law", "viscous_coulomb"),
        )),
    ),
)
```

`lower_architecture_plan()` 会把 K2 交叉乘积展开为真正的 IR 候选，并限制最大候选数，防止退化成无限 NAS。

---

## 8. 第五步：Estimation Contract 由你定义

这是 M2N 非常重要的边界。

`run_design_session()` 不替你决定“一步预测、trajectory rollout、inverse model、output-error”哪个正确。

你需要定义：

```python
def evaluator(candidate, compiled_model):
    # 1. 只用 train 拟合参数
    # 2. 在 normal validation 上计算风险
    # 3. 多个独立 restart
    return [risk1, risk2, risk3]
```

不同设备不能把这些目标函数随便互换。

建议：

- 机械 force-balance：可用 rate/acceleration contract；
- 动态系统：优先 trajectory/free-rollout；
- 逆动力学：明确声明 inverse contract；
- 静态局部本构：直接输出误差可以合理。

**故障数据不得用于正常架构选择。**

---

## 9. 架构选择规则

默认 `ArchitectureSelectionPolicy` 是研究冻结值：

```text
practical_ratio = 0.80
restart_stability = 0.75
```

含义：最佳候选不只是 loss 最小，还必须：

- 有明显 practical advantage；
- 多 restart 下稳定。

这些不是通用工业常数。你可以在项目开始前根据误差尺度预注册新 policy，但不要看完 test 后再改。

---

## 10. SELECTED / AMBIGUOUS 后分别怎么做

### SELECTED

1. 冻结候选 ID；
2. 用 train+val 重新拟合参数；
3. 再打开独立 test；
4. 记录 IR signature、数据 provenance 和参数；
5. 部署时继续检查 support/OOD。

### AMBIGUOUS

不要强行选一个，也不要自动上 K3。

可以构造：

```python
ArchitecturePortfolio(models, unresolved_slots=(... ,))
```

它会保留所有有限候选，并输出：

```text
candidate predictions
mean
std
min
max
```

这里 `std` 表示**模型形式不确定性**，不是测量噪声。

---

## 11. M2N evidence 如何使用

如果有 frozen portfolio，可以使用：

- `RobustSupportReference`
- `portfolio_derivative_evidence()`
- `summarize_portfolio_segment()`

输出：

```text
best finite residual
median finite residual
model disagreement
consensus residual
normal-support distance
```

这些是模型证据，不等于 fault probability。

若后续要做 FEC，应该把这些作为 FEC 的输入之一，而不是把 FEC 逻辑塞回 M2N。

---

## 12. 一个完整的新设备项目应该保存什么

推荐目录：

```text
my_device/
├─ mechanism_contract.py
├─ observation_contract.md
├─ run_design.py
├─ data/
│  ├─ train.*
│  ├─ val.*
│  └─ test.*
├─ results/
│  ├─ architecture_plan.json
│  ├─ design_session.json
│  ├─ selected_model.pt   或 architecture_portfolio.pt
│  ├─ provenance.json
│  └─ test_report.json
└─ README.md
```

最重要的是保存：

- 输入机理知识到底来自哪里；
- 哪些是 `BLINDED_FOR_VALIDATION`；
- 数据 split 在什么时候冻结；
- test 什么时候第一次被打开；
- K2 候选集合是否事先冻结；
- 最终是 SELECTED、FINITE SET、AMBIGUOUS 还是 UNSUPPORTED。

---

## 13. 什么时候应该停止，而不是继续完善 M2N

当你发现：

```text
当前候选在正常数据上无法稳定区分
```

先问：

1. 现有传感器有没有资格区分？
2. 正常任务有没有激发差异？
3. 是否遗漏已知硬机理？
4. 是否应该保留有限 Architecture Set？

而不是直接：

```text
再加一个网络
再换一个 loss
再调阈值
```

M2N 的价值之一，就是允许“无法唯一确定”成为正式工程结果。


## 14. 外部 SciML backend：M2N 负责裁决，不负责重复发明

Practitioner v1.3 将成熟外部方法视为可插拔 backend。核心原则是：

```text
先做 M2N qualification
→ 允许后调用外部 backend
→ backend 只产生 proposal / trained component / compiled module
→ 再回到 M2N 做 admission、support、held-out 与 architecture decision
```

例如 PySINDy 的公式输出首先只是 `UNADMITTED_K2_PROPOSAL`。只有经过工程语义、单位/约束和验证后，才可以进入有限 K2 family。

运行：

```bash
m2n backends
```

详见 `docs/07_EXTERNAL_BACKENDS_CN.md`。

## 15. Novelty Kill 与工具使用要分开

NKT 只回答论文 claim 是否站得住，不回答某个功能是否应该删除。

```text
Capability Ledger: 有用吗？
Novelty Ledger: 原创吗？
```

因此：

- 已知方程编译即使已有 Neural Compiler，也继续保留；
- K3 hybrid learning 即使已有 UDE/APHYNITY，也继续保留并计划调用成熟 backend；
- 方程发现即使已有 SINDy/SymMatika，也可以作为 M2N 候选生成 backend；
- 论文写作时必须正确归因，不把 adopted capability 写成 M2N 首创。

运行：

```bash
m2n nkt
```

详见 `docs/08_NOVELTY_KILL_TEST_CN.md` 和根目录 NKT-1 报告。
