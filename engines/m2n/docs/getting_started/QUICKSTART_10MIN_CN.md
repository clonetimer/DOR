# M2N 10 分钟快速开始

## 1. 安装

建议 Python 3.10+。

```bash
python -m venv .venv
```

激活环境后：

```bash
pip install dist/m2n_toolkit-2.0.0-py3-none-any.whl
m2n doctor
```

若你的机器已经安装 PyTorch，也可以：

```bash
pip install --no-deps dist/m2n_toolkit-2.0.0-py3-none-any.whl
```

然后确保 `numpy/scipy/torch` 可导入。

## 2. 跑最简单的 K1 示例

```bash
python examples/01_k1_fixed_mechanics.py
```

它展示：已知二阶机械拓扑，只学习驱动/阻尼参数。

## 3. 跑 K2 架构选择

```bash
python examples/02_k2_design_session.py
```

它展示完整闭环：

```text
K2 friction = {viscous, viscous+Coulomb}
                ↓
Architecture Plan
                ↓
2 个可执行网络
                ↓
正常 train/validation
                ↓
SELECTED 或 AMBIGUOUS
```

## 4. 跑受控盲测 EUV 示例

```bash
python examples/03_blinded_multivariate_euv.py
```

这是 v6.3 冻结迁移实验的精简使用版。真实生成公式对实验者已知，但明确标成 `BLINDED_FOR_VALIDATION`，M2N 只能从 3 个有限 K2 响应面中选择。

**不要把该示例的近机器精度理解成真实 EUV 工厂精度。** 它是无真实测量噪声的仿真器内生盲测。

## 5. v2.0 推荐：建立声明式设备设计项目

```bash
m2n init-design my_device --template graph
m2n design --spec my_device/device_card.json --data my_device/data/normal_data.csv --out my_device/output
```

普通使用者优先修改 `device_card.json` 和正常 CSV，不需要先写 Python IR/SlotBinding。高级用户仍可使用 `m2n new-project` 与底层 API。

## 6. 输出怎么看

### `SELECTED`
正常数据对一个有限候选有稳定、显著优势。冻结架构后再打开 test。

### `AMBIGUOUS`
多个候选都可解释数据，或优势不稳定。**不要自动升级 K3。** 可以保留有限 `ArchitecturePortfolio`。

### `UNSUPPORTED`
观测/数据根本不支持该机理归因。先改传感器/任务/目标，不是换更大的 NN。

### `FIXED`
只有 K0/K1 或 K2 已经由工程知识确定，不需要架构竞争；后续只是参数拟合与验证。


## 7. 查看 NKT 与外部 backend

```bash
m2n nkt
m2n backends
```

`m2n nkt` 用于查看哪些能力可以继续作为研究 claim，哪些只是 adopted capability。它**不会删除功能**。

若要让 PySINDy 作为 K2 候选生成器：

```bash
pip install 'm2n-toolkit[discovery]'
python examples/05_pysindy_candidate_backend.py
```

如果没有安装 PySINDy，示例会明确报告缺依赖并停止；不会绕过 M2N qualification。
