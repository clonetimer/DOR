# DOR × M2N 外部工具包选择性集成设计

## 1. 技术报告扫描

本轮审计对象为用户提供的两个本地发布包，来源链未独立认证：

| 对象 | SHA-256 | 实际变化 | 创新分类 | 证据等级 |
|---|---|---|---|---|
| DOR Satellite R2 | `66db30da53ae2a8c2b283ab85ff08b4300d36c238ff535aab0ee5eb3ff36dd0b` | 辨识、ISGN 桥、多任务仲裁、垂直验证、冻结门和卫星应用证据 | 架构组合/工程治理 | E1；包内自报测试为 E2，非独立复现 |
| M2N v2.12 | `b08c4707bcc3890b1e89119dd512308269cc458728f7da07c40f1b5e6df7cbe8` | 模型族资格、时序候选、波形表示、独立上下文和工况支持域 | 数据表示、组件组合、资格策略 | E1；包内自报测试为 E2，非独立复现 |

不声明两个包提出了新的基础神经算子。TCN、GRU、Transformer、AR、仿射状态空间和工程时域特征均属于已有机制；本次价值在于受合同约束的组合、比较和拒答流程。

## 2. 比较矩阵

公平条件是相同数据角色、独立上下文划分、训练预算、候选调参预算和部署信息边界。

| 维度 | 当前 DOR/M2N | 外部 DOR R2 | 外部 M2N v2.12 | 集成决策 |
|---|---|---|---|---|
| 多阶段自动设计 | 表格检测/定位/分类/辨识候选 | 强化结果仲裁 | 不拥有诊断语义 | 保留当前设计器，引入仲裁 |
| 机理编译 | 安全符号编译、UDE | 无 | 无 | 必须保留当前实现 |
| 时序选择 | 无一等时序竞争 | 仅应用 RF 时序统计 | Summary/AR/TCN/GRU/Transformer | 引入 M2N 时序支路 |
| 物理时序 | 符号/UDE | 应用 residual | K1/bounded/affine/quaternion | 与现有机制候选并存 |
| 波形表示 | 有限人工特征 | 应用时序统计 | 合同预声明 ROI + 时域特征 | 引入，禁止标签驱动 ROI 搜索 |
| 数据独立性 | 分组评估但合同较弱 | truth quarantine | context/time fail-closed | 引入 M2N 合同 |
| 工况外推 | 切片与 FEC | mission threshold | train-only operating support | 引入，禁止解释为故障概率 |

## 3. 组件机制卡

### Model-family qualification

- 一句话机制：把机理使用方式、任务、表示和数据资格映射为有限模型族的 `ELIGIBLE/CONDITIONAL/NOT_ELIGIBLE`，而不是直接训练任意模型。
- 输入/输出：`DeviceCard + ModelingContract + data profile → ModelFamilyPlan`。
- 作用轴：候选模型族轴；不改变原始观测事实。
- 信息操作：保留合同和数据统计，丢弃无资格候选。
- 主要收益：选择空间约束和拒答，不是新增模型容量。
- 失败边界：资格规则不完整时可能漏掉合适模型；不能宣称全局 NAS 最优。

### Temporal design

- 一句话机制：在独立 trajectory 上比较低复杂度与深度时序候选，并允许 `AMBIGUOUS`。
- 输入 shape：每个 context 为变长 `[T,F]` 序列；分类输出 `[C]`，预测输出逐时间步 `[T,Y]`。
- 核心候选：Summary、Linear AR、TCN、GRU、Transformer，以及合同允许的 K1、bounded、affine、quaternion。
- 状态：GRU/时序卷积/attention 各自保留不同时间依赖；padding mask 防止填充值成为证据。
- 主要风险：小样本深网不稳定、长序列成本、规则采样假设、多速率数据尚未原生覆盖。

### Context support

- 一句话机制：只使用训练 contexts 冻结长度、持续时间和声明工况的经验支持域，把外推样本从架构排名中隔离。
- 核心计算：训练 context 统计标准化后，以 leave-one-out 最近邻最大距离形成经验半径。
- 信息边界：支持域是模型资格域，不是健康、安全或故障边界。

### DOR multi-task arbitration

- 一句话机制：检查检测、分类、定位、根因和辨识之间的集合与语义一致性，输出最深的无矛盾诊断层级。
- 状态：`SUPPORTED / ALTERNATIVES / UNKNOWN / ABSTAIN`。
- 失败边界：仲裁不产生新证据，也不能把 closed-set 分类升级为未知故障识别。

## 4. 架构方案

### 问题合同

- 任务：由设备机理、观测数据和诊断目标设计多阶段故障诊断模型。
- 输入：部署时可用的传感器、机理卡、数据角色、独立 context、局部时间和资源约束。
- 输出：阶段资格、有限候选、冻结模型/组合、校准接口和可拒答诊断。
- 基线失败：当前表格自动设计无法一等比较时序/波形候选，跨任务结果缺少统一一致性门。

### 可证伪假设

> 如果失败来自时序结构或运行工况未被表格汇总正确表达，则在严格独立 context 验证下，时序/波形候选应改善对应切片；若低复杂度 AR/工程特征达到相同效果，则不把收益归因于深层网络。

### 最终计算图

```text
DeviceCard + DatasetManifest + Request
  -> 数据/任务/独立上下文资格
  -> Detection / Localization / Classification / Identification 分解
  -> 表格诊断候选 + ModelFamilyPlan + TemporalDesign
  -> 独立验证、工况支持、资源和稳定性裁决
  -> FEC calibration
  -> DOR multi-task arbitration
  -> vertical validation / freeze gate
```

当前 `diagnostic_modeling`、`symbolic_compiler` 和 `ude` 不被 v2.12 文件覆盖。外部模型制品不进入信任边界；PyTorch 制品仅使用 `weights_only=True` 加载。

## 5. 实验计划

| ID | 实验 | 回答的问题 | 否证条件 |
|---|---|---|---|
| B0 | 原 269 项回归 | 现有能力是否保留 | 任一既有测试失败 |
| S0 | Summary/Linear AR | 深度时序是否必要 | 简单候选效果等价且成本更低 |
| A0 | 完整时序候选集 | 新支路是否可运行 | 资格、训练或冻结推理不一致 |
| A1 | 移除 context support | 支持域是否贡献 | OOS/IN 风险无可复现差异 |
| M0 | 打乱 context 或重置时间 | 收益是否来自合法时序结构 | 破坏机制后不退化 |
| H0 | 弱故障/后程故障/工况漂移切片 | 是否改善目标失败 | 目标切片无改善 |
| R0 | 缺失、非有限、多速率、不规则采样 | 是否 fail-closed | 静默训练或跨 context 窗口 |
| V0 | 多随机种子和设备外验证 | 选择是否稳定 | 赢家随 seed/设备剧烈变化 |

合并门：DOR、M2N、FEC、ISGN 全量回归通过；新 CLI smoke 通过；源码无缓存/临时产物；现有符号/UDE/诊断模型测试保持通过。

## 6. 组件登记

机器可读登记位于 `docs/component_registry_external_integration.csv`。采用状态为 `ADOPTED_WITH_BOUNDS`，不登记为新基础算子。

## 7. 资源使用追踪

| Skill 资源 | 对本次工作的影响 |
|---|---|
| `README.md` | 采用问题先行、证据可审计的集成方式 |
| `core-principles.md` | 保留部署信息边界，明确支持域不增加事实证据 |
| `component-ontology.md` | 将更新区分为表示、序列状态、架构组合和工程治理 |
| `research-protocol.md` | 两个来源不明本地包只记 E1/E2 自报证据，不作创新首发声明 |
| `design-validation-protocol.md` | 建立 B0/S0/A0/A1/M0/H0/R0/V0 合并门 |
| `learning-protocol.md` | 时序与状态组件按 L2/L3 深度审计，普通 CLI/Schema 按 L1 接口审计 |
| `full-design-principles.md` | 以基线失败、信息合同、成本和停止条件约束候选引入 |
| `component-card.md` | 形成上述四类组件机制卡 |
| `comparison-matrix.md` | 形成三版本能力和边界比较 |
| `architecture-proposal.md` | 形成问题合同、假设、计算图与接口方案 |
| `experiment-plan.md` | 形成分层测试、机制破坏和压力测试计划 |
| `technical-report-scan.md` | 记录包哈希、来源限制、创新分类和证据等级 |
| `component-registry.csv` | 生成机器可读集成组件登记 |
| `neural_network_component_learning_log_v45.csv` | 仅用于确定审计深度，没有改写个人学习进度或伪造掌握结论 |

## 8. 结论边界

- 可以声称：当前工具新增了有边界的模型族资格、时序/波形竞争、独立 context 和工况支持，以及跨任务诊断仲裁。
- 测试后可以声称：这些能力与既有表格诊断、符号编译和 UDE 在当前回归集内兼容。
- 不得声称：已找到任意工业设备的全局最优模型，或包内卫星 replay 指标构成开发者盲测泛化证据。
