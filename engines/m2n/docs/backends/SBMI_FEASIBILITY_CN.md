# SBMI Feasibility：在 M2N 中何时值得使用

## 1. SBMI 能补什么

Simulation-Based Model Inference (SBMI) 的核心价值是：对由多个离散组件组成的 scientific simulator，同时推断：

```text
model components
+
continuous parameters
```

并保留多个 data-consistent model configurations，而不是强制单一模型。

这和 M2N 的大型 K2/model-set 问题很匹配。

## 2. 为什么 EHSS 当前不需要

EHSS v6.1 只有：

```text
2 pressure-path candidates
×
3 friction candidates
=
6 configurations
```

6 个模型直接跑完、验证完、保留 ArchitecturePortfolio 更透明；为此训练 amortized SBI 网络没有工程收益。

## 3. 什么时候开始有意义

v1.3 提供：

```python
from m2n import (
    ModelComponentSlot, SBMIModelSetRequest,
    assess_sbmi_feasibility,
)

req = SBMIModelSetRequest(
    slots=(
        ModelComponentSlot("valve", ("a", "b", "c")),
        ModelComponentSlot("friction", ("a", "b", "c")),
        ModelComponentSlot("thermal", ("a", "b", "c")),
        ModelComponentSlot("pump", ("a", "b", "c")),
    ),
)
print(assess_sbmi_feasibility(req))
```

4 个三选一 slot 会形成：

```text
3^4 = 81 configurations
```

默认工程启发式：

```text
<= 32  → M2N exact enumeration
> 32   → FEASIBLE_CONTRACT
```

注意：**32 不是 SBMI 论文阈值**，只是 M2N 的透明性/工程成本默认值，项目可改。

## 4. 必要前提

SBMI-style backend 至少要求：

- simulator 可运行；
- simulator 能实例化不同 model-component configuration；
- 连续参数有 prior/bounds；
- 已定义 observation/summary contract；
- 有足够 simulation budget。

任何一项缺失，M2N 返回 `BLOCKED`。

## 5. 一个重要接口差异

M2N 的 K2 slot 通常是：

```text
friction ∈ {viscous, Coulomb, Stribeck}
```

也就是每个 slot **exactly one**。

原始 SBMI 论文用 binary model components 的分布表示。因此真正接入时不能简单把每个 M2N candidate 当独立开关；必须加：

```text
one-hot binary mask per M2N slot
exactly-one constraint
```

v1.3 的 `export_sbmi_contract()` 已显式写出这个边界。

## 6. 当前 feasibility 结论

### 方法适配度

高：

- 大型有限 component family；
- simulator likelihood 不可得；
- 需要 model posterior / multiple configurations；
- 需要同时推断结构和参数。

### 当前工程成熟度

中低：

- 本轮没有找到可直接作为稳定 Python dependency 接入的官方 SBMI reference package；
- 不建议 M2N 自己重写 Grassmann mixture inference；
- 当前通用 `sbi` 生态可以作为 simulation-based inference 基础，但其软件许可与具体部署方式需要独立评审；
- 2026 年已有更可扩展的 model-structure SBI 研究（例如 PRISM），真正出现组合爆炸问题时应重新比较，而不是锁死“必须 SBMI”。

## 7. v1.3 结论

```text
SBMI = FEASIBILITY CONTRACT
不是 runtime backend
```

它已经进入 M2N 的“什么时候值得调用”逻辑，但当前不加入重量级依赖。
