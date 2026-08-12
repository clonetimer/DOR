# M2N Goal-Driven Design：从设备卡片和正常数据直接生成网络

## 这一步为什么重要

M2N 的原始目的不是要求用户先学会 M2N 内部 IR API，而是：

> 给出设备机理知识、观测条件和正常数据，设计出允许的神经/机理混合架构。

v1.3 以前仍要求用户手写 Python `IR skeleton + SlotBinding + evaluator`。v1.4 新增声明式入口，把这些低层步骤下沉到工具内部。

## 最短使用路径

```bash
m2n init-design my_device --template mechanics
cd my_device
m2n design --spec device_card.json --data data/normal_data.csv --out output
```

结果目录包含：

- `architecture_plan.json`：K0/K1/K2/K3 → architecture actions；
- `qualification.json`：最小观测/数据资格检查；
- `candidate_space.json`：实际可执行候选；
- `design_result.json`：SELECTED / AMBIGUOUS / FIXED / UNSUPPORTED；
- `DESIGN_REPORT.md`：人可读设计报告；
- `models/*.pt`：最终模型或有限 Architecture Set；
- `reproduce.txt`：复现命令。

若得到单一模型：

```bash
m2n predict --artifact output/models/<model>.pt --data new_normal_or_operating_data.csv --out prediction.csv
```

## 当前声明式模板

### `second_order_mechanics`

适合：旋转/直线执行机构、机械支路、机电位置系统等。

固定外层拓扑：

```text
q_dot = v
v_dot = drive(input) - resistance(v) - optional_constant_load
```

可让 resistance 处于 K0/K1/K2/K3。K2 原生候选包括 `viscous / viscous_coulomb / asymmetric_viscous_coulomb / stribeck`；K3 可用局部 dissipative neural law。

### `separable_peak_response`

适合：多个物理因素共同决定一个局部峰值响应的本构/响应面。

例如：反应产率、转换效率、共振/工艺窗口等。K2 可由不同 `active_axes` 组合构成。

## 重要边界

1. 声明式 Frontend 不等于自然语言自动恢复完整机理。
2. 默认资格检查是工程 guardrail，不是普适可辨识性定理。
3. `SELECTED` 只表示在当前 Knowledge/Observation/Data Contract 下选择证据充分，不等于物理真理。
4. `AMBIGUOUS/UNSUPPORTED` 是合法设计结果，不会自动退化为 K3。
5. PySINDy/SymMatika 等 backend 只在需要扩候选时介入，不是主流程。
