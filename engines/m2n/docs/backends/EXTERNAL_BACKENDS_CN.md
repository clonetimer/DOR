# M2N 外部 SciML Backend 使用说明（v2.0）

## 1. 核心原则

M2N 不需要比 SINDy/SymMatika 更会找公式，也不需要比 UDE/APHYNITY 更会训练 missing-physics NN。M2N 负责：

```text
Knowledge / Observation / Data qualification
                ↓
        现在允许调用什么 backend？
                ↓
          backend proposal/result
                ↓
           Admission Gate
                ↓
      M2N validation / adjudication
```

外部 backend 永远不能绕过 qualification。

## 2. 当前 registry

| backend | 作用 | v2.0 状态 | 软件处理 |
|---|---|---|---|
| `m2n-native` | 原生有限候选/编译 | NATIVE | M2N 自有 |
| `pysindy` | 稀疏动力学候选 | **OPTIONAL_ADAPTER / full local runtime tested** | MIT；可选依赖 |
| `symmatika` | 显式/隐式 symbolic candidates | **EXTERNAL_ADAPTER** | Apache-2.0；source/job adapter，不 vendoring |
| `sbmi` | scientific model-component + parameter inference | **FEASIBILITY CONTRACT** | 原论文方法可用；本包不重写/不 vendoring |
| `ude` | K3 hybrid trainer | CONTRACT_ONLY | demo repo MIT；不 vendoring |
| `aphynity` | physics + learned remainder | CONTRACT_ONLY | MIT；不 vendoring |
| `cann` | 材料本构专家 | CONTRACT_ONLY | 许可证仍待核验 |
| `neural-compiler` | symbolic program → differentiable module | CONTRACT_ONLY | MIT；不 vendoring |

查看当前环境：

```bash
m2n backends
```

## 3. PySINDy 2.1：完整离线链已验证

v1.3 自带用户提供的两个第三方 wheel（原样保留）：

```text
third_party/PySINDy-2.1.0/pysindy-2.1.0-py3-none-any.whl
third_party/derivative-0.6.3/derivative-0.6.3-py3-none-any.whl
```

在本轮隔离环境中已实际验证：

```text
PySINDy 2.1.0 import       PASS
derivative 0.6.3 import    PASS
显式 x_dot discovery       PASS
PySINDy 内部数值微分        PASS
```

内部数值微分测试恢复：

```text
x' = -1.000 x
y' = -1.700 y
```

### 离线安装

```bash
pip install --no-deps third_party/derivative-0.6.3/derivative-0.6.3-py3-none-any.whl
pip install --no-deps third_party/PySINDy-2.1.0/pysindy-2.1.0-py3-none-any.whl
```

前提是环境已经具备 NumPy / SciPy / scikit-learn / typing-extensions 等基础依赖。

### PyPI 不通时

优先顺序建议：

```text
已有可信本地 wheel
→ PyPI
→ piwheels（若有该版本/平台 wheel）
→ 上游 release/source
```

从 piwheels 获取后仍需核对：包名、版本、Python/ABI/platform tag、SHA-256、许可证和依赖，不把“能下载”当作“可直接信任”。

## 4. SymMatika：为什么是 external adapter

SymMatika 上游是 C++23/CMake 工程，而不是 Python package。M2N v1.3 不复制其源码，而是提供：

- explicit mapping 数据集导出；
- 自动生成 `main.cpp` driver；
- 临时复制用户提供的上游 source tree；
- CMake configure/build/run；
- 通过 `Model::return_final_candidates()` 输出机器可读候选；
- 所有候选仍标记 `UNADMITTED`。

详见 `docs/10_SYMMATIKA_ADAPTER_CN.md`。

## 5. SBMI：什么时候才值得用

SBMI 适合“多个离散模型组件 + 连续参数 + simulator”的组合式模型选择，而且能保留多个 data-consistent model configurations。

但若 M2N 只有很小的有限模型族，例如 EHSS 当前：

```text
2 pressure-path × 3 friction = 6 configurations
```

直接枚举、验证、保留 portfolio 更简单透明。v1.3 默认 exact-enumeration limit 为 32；超过后才提示 `FEASIBLE_CONTRACT`，这只是 M2N 工程启发式，不是 SBMI 论文阈值。

详见 `docs/11_SBMI_FEASIBILITY_CN.md`。

## 6. 不允许的用法

```text
观测不足 → SINDy/SymMatika → 得到公式 → 直接当真实机理
```

```text
K0/K1 已知公式 → 因为 symbolic candidate loss 更低就覆盖工程机理
```

```text
SBMI posterior mode → 直接宣称唯一真实结构
```

```text
backend proposal → 跳过 units/support/IR binding → 进入最终模型
```

## 7. 后续 adoption

v1.3 后不再按“接得越多越好”推进：

1. PySINDy：已可用；
2. SymMatika：external adapter 已可用，等待真实上游环境运行；
3. SBMI：仅在模型族出现组合爆炸时启用；
4. UDE / APHYNITY：真正出现合格 K3 需求时再接；
5. Neural Compiler：native compiler 维护成本成为瓶颈时再接；
6. CANN：材料本构应用出现时再接。
