# 先读我：如何真正开始使用 M2N

过去的 M2N 包含大量 Exx 验证、实验 ledger、研究报告。这些材料用于证明方法边界，但不适合日常使用。本包把它们拆成了“运行工具”和“研究证据”两层。

## 你日常只需要记住 5 个对象

### 1. `MechanismSlot`
表示一个**局部机理槽位**，例如：摩擦、阀口流量、热交换、转换效率。

### 2. K0/K1/K2/K3
表示你对这个局部机理知道到什么程度：

- **K0**：形式和参数可信，直接固定；
- **K1**：形式可信，参数待估；
- **K2**：有有限个物理合理候选，需要数据选择；
- **K3**：没有可信有限候选，且资格审查通过后，才考虑局部受约束神经化。

### 3. `NeuralArchitecturePlan`
把知识合同变成“下一步应该固定、参数化、搜索 K2、还是阻断”的架构计划。

### 4. IR + `SlotBinding`
IR 是设备计算图骨架；`SlotBinding` 告诉 M2N 某个 K2 候选如何局部修改 IR。

### 5. `run_design_session()`
真正执行候选网络编译 + 正常数据验证 + 架构选择。

## 最短路径

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install dist/m2n_toolkit-2.0.0-py3-none-any.whl
m2n doctor
python examples/02_k2_design_session.py
m2n new-project my_device
cd my_device
python run_design.py
```

如果 `m2n doctor` 输出 `compile: PASS`，工具链已安装成功。

## 新设备不要从“选网络”开始

正确顺序是：

1. 写设备状态/输入/输出；
2. 列出确定硬机理；
3. 把每个局部不确定关系标成 K0/K1/K2/K3；
4. 写清正常数据能观测什么；
5. 才建立候选网络。

如果你第一步就问“应该用 LSTM 还是 Transformer”，那已经偏离 M2N 的定位。

## v2.0 推荐入口

如果你的目标是“拿设备机理和数据直接设计网络”，优先使用声明式入口，而不是从 Python API 开始：

```bash
m2n init-design my_device --template graph
m2n design --spec my_device/device_card.json --data my_device/data/normal_data.csv --out my_device/output
```

然后优先阅读 `14_P1_P5_END_TO_END_CN.md`、`15_GENERIC_MECHANISM_GRAPH_CN.md`、`16_QUALIFIED_K3_CN.md`、`17_PHM_FEC_HANDOFF_CN.md`。
