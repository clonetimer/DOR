# M2N v2.3 故障诊断模型编排

## 1. 目的

闭合此前四工具链中的核心断点：DOR 已能声明“要做故障分类/异常检测”，FEC 已能校准 class/anomaly evidence，但没有工具明确负责在传统 ML、统计模型与普通神经基线之间进行**有限、可拒答的训练与模型选择**。v2.3 把这项职责放入 M2N，而不是增加第五个工具。

## 2. 职责边界

```text
DOR: task / label semantics / fault ontology / final arbitration
  ↓ M2N-DIAGNOSTIC-MODEL-REQUEST/1.0
M2N: feature-view qualification + finite model-family training/selection
  ↓ M2N-DIAGNOSTIC-SCORE-PACK/1.0
FEC: independent calibration / prediction-set validity / abstention
  ↓ calibrated class/evidence state
DOR: class-frame alignment + alternatives/unknown/abstain + root-cause reasoning
```

M2N 不建立 fault ontology，不把类别概率解释为物理根因，也不改变 ISGN/FEC 的职责。

## 3. 当前稳定任务

### FAULT_CLASSIFICATION

有限 registry：

- `logistic_regression`
- `rbf_svm`
- `random_forest`
- `hist_gradient_boosting`
- `mlp`（普通神经基线，不是 physics architecture）

默认主指标 `macro_f1`，可选择 `balanced_accuracy` 或 `min_class_recall`。模型只利用 `validation` 做选择。

### ANOMALY_DETECTION

有限 registry：

- `pca_reconstruction`
- `one_class_svm`
- `isolation_forest`

训练仅允许 trusted-normal rows。若 validation 没有独立 fault label，只输出 `PORTFOLIO`，不通过正常拟合误差制造 winner。模型阈值/正常 reference calibration 仍交给 FEC。

## 4. Mechanism-aware feature views

请求可显式声明：

- `RAW`：原始/工程传感特征；
- `MECHANISM`：ISGN/M2N 产生的参数、机制 residual、物理特征；
- `RAW_PLUS_MECHANISM`：二者合并。

M2N 比较的是“feature view × model family”候选，因此可以回答“机理特征是否真的帮助了当前诊断任务”，而不是预设 physics feature 一定更好。

禁止语义偷换：`MECHANISM` view 获胜只说明这些特征在当前验证合同下有诊断价值，不证明某个 residual 对应真实根因。

## 5. 数据隔离合同

必须显式给出 split column。M2N 不自动随机切分工业故障数据。

```text
train       -> candidate fitting
validation  -> model/view selection only
calibration -> FEC only; never used for M2N selection
test        -> final independent evaluation only
```

对于 run/asset 级数据，调用方应在生成 split column 时保证 run-disjoint / asset-disjoint；v2.3 不会猜测哪些 row 属于同一次物理运行。

## 6. 输出状态

- `FIXED`：资格审查后仅剩一个候选；不是“证明它最好”。
- `SELECTED`：相对所有候选均达到 practical margin + restart stability。
- `AMBIGUOUS`：多个候选无法稳定区分。
- `PORTFOLIO`：当前证据只支持保留有限集合，典型为 normal-only detector。
- `UNSUPPORTED`：数据/标签/特征/分区/质量合同失败。

## 7. 主要制品

- `qualification.json`
- `candidate_registry.json`
- `validation_metrics.json`
- `selection.json`
- `models/*.joblib`
- `diagnostic_model_bundle.json`
- `scores/calibration_scores.csv`（仅唯一模型 + calibration split）
- `scores/test_scores.csv`（如提供 test split）
- `score_contract.json`

## 8. 当前明确不支持

- raw waveform 自动表征学习；
- sequence model/TCN/RNN/Transformer 自动选择；
- 图拓扑/GNN fault classifier；
- fault localization/root cause；
- novel-fault semantics；
- RUL/maintenance optimization；
- 全局 AutoML/hyperparameter sweep。

这些不是“偷偷补齐”的 TODO；后续应由真实设备应用暴露是否需要扩 M2N adapter/backend。
