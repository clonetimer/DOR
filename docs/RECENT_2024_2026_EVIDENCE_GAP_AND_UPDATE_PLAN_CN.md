# DOR 近两年证据、需求差距与更新计划

核验日期：2026-08-13

严格检索窗口：2024-08-13 至 2026-08-13

目标需求：输入工业设备机理与数据后，工具能够设计并验证适合该设备的检测、定位、分类、辨识单阶段或多阶段故障诊断流水线。

## 1. 结论与边界

结论是：**DOR 仍未完全满足目标需求**。本轮更新后，它能够在已支持的数值表格/时序 IR、显式设备卡、显式监督与有限候选集合内，按 validation 证据、最差工况、缺失传感器压力和资源约束选择模型或保留等价集合；它不能接收任意原始工业数据和自然语言机理后自动生成全局最优架构，也不能把定位或参数估计自动升级为已证实根因。

“最合适”在 DOR 中被限定为：

> 在数据角色隔离、候选集合、随机种子、工况切片、传感器缺失压力、模型大小、结构复杂度代理与推理时延合同内，validation 效用稳定领先；证据不足时保留候选集合或返回 `UNSUPPORTED`。

该定义是有限、可审计的工程选择，不是全局神经架构搜索最优证明。

## 2. 检索方法和证据分层

只将官方数据页、DOI、IEEE Xplore、arXiv、PMLR、NeurIPS Proceedings、OpenReview 和官方挑战页作为技术事实来源。二手博客与聚合搜索结果不进入机器可读证据库。

- E0：预印本或仅作者自报，允许进入研究队列，不能成为默认能力依据。
- E1：同行评审论文、官方挑战报告或本地可复现实验事实。
- E2：跨来源复现或独立外部验证。
- E3：跨设备、跨场景、多次独立验证。

证据目录：

- 数据集：`benchmarks/public_datasets.json`，16 项，其中新增 9 项近两年记录；受限数据明确标为 `RESTRICTED`。
- 方法：`benchmarks/recent_methods_2024_2026.json`，12 项；每项含机制变化、创新类别、发表状态、证据等级、DOR 决策和风险。
- CLI：`dor dataset-catalog`、`dor method-catalog`、`dor benchmark-plan`。

## 3. 近两年公开工业数据集扫描

| 数据集 | 时间/来源 | 设备与压力 | 对 DOR 的直接含义 |
|---|---|---|---|
| DCASE 2025 Task 2 | 2025，官方挑战、Zenodo | 多类机器声、first-shot、source/target 域迁移、工厂噪声 | 暴露 raw audio 缺口；不得用数值时序 IR 冒充音频支持 |
| Industrial Robot Telemetry | 2025，Zenodo | 七轴机器人、运动/能耗、正常与主动改变运行 | 需要 episode/run 隔离和程序外推 |
| Industrial Screw Driving | 2025，Zenodo | 34,000+ 操作、自然磨损、六种受控场景 | 适合未来磨损漂移与离散工序验证 |
| PMSM inverter faults | 2024，Zenodo | 八个原始传感器、15 个派生量、九工况 | 必须分开原始/派生证据并做工况最差门禁 |
| Induction motor startup current | 2025，Zenodo/IEEE IECON | 四台电机、断条/端环、供电与启动时长变化 | 必须按完整启动和电机 specimen 切分 |
| XJTU-DV | 2025，Zenodo | 转子、泵、梁的动态视觉非接触观测 | 暴露 raw vision/event-camera 缺口 |
| PATH | 2024，arXiv/Zenodo/GitHub | 16 通道离散动力总成序列、污染训练、变量状态 | 增加污染训练、在线阈值和工况状态压力 |
| FactoryNet | 2026，arXiv/Hugging Face | 多机器人/CNC/拧紧、真实/仿真/反事实、S-E-F-C | 适合未来跨 embodiment 和 sim-to-real；当前预印本且许可未明确 |
| Multi-defect spherical roller bearing | 2025，Zenodo metadata | 单/双/多轴承故障、变速变载 | 文件受限；公开元数据页不等于公开可用数据集，benchmark 必须阻断 |

严格窗口外但仍保留在历史目录的 CWRU、Paderborn、NASA IMS/ADAPT、PRONOSTIA、MetroPT-3 与 UCI Hydraulic 用于长期可比基线，不计入“近两年新增”统计。

## 4. 近两年方法与评测扫描

| 工作 | 证据状态 | 机制/发现 | DOR 决策 |
|---|---|---|---|
| TSB-AD，NeurIPS 2024 | E1 | 统一比较 40 算法与大规模序列，指出数据/指标/协议缺陷；简单方法常有竞争力 | 采用评测原则，保留简单基线，后续补 VUS-PR/事件指标 |
| UniTS，NeurIPS 2024 | E1 | 任务 token 化与统一时序模型 | 仅作为未来可选候选，先过资源和跨域门禁 |
| PATH，2024 | E0 | 污染训练、离散序列、阈值敏感 | 采用 benchmark，不把模型结果当已验证方法 |
| DCASE 2025 Task 2 | E1 | 首样本机器声与域迁移 | 音频 adapter 后采用 benchmark |
| UniFault，2025 | E0 | 轴承数据统一化和跨域预训练 | 研究队列；需独立复现、specimen-held-out 和资源匹配 |
| Mantis，ICML 2025 workshop | E1 | 对比预训练 ViT、时序分类适配和校准 | 可选候选，不作为默认模型 |
| TSAD-C，UAI 2025 | E1 | 去污染器与时空图条件扩散 | 采用污染训练资格要求；复杂模型需资源匹配 |
| M2-STGN，IEEE TIM 2025 | E1 | 缺失数据下的图/掩码时序组合 | 采用缺失传感器需求，不直接照搬网络 |
| Industrial ASD，IEEE SENSORS 2025 | E1 | 工业现场噪声/混响中的 AE+GMM | 未来音频简单基线与 SNR 切片 |
| FactoryNet，2026 | E0 | 跨设备工业统一 schema 与 sim-to-real | 研究队列；等待评审、许可和独立验证 |
| TSFM zero-shot failure，2026 | E0 | 基础模型在 SWaT 的过度泛化与边界偏置 | 采用负向准入门：不能按名气默认启用 TSFM |
| Adaptive conformal TSFM，2026 | E0 | 漂移下加权自适应分位与误报 p 值 | 研究队列；不能替代独立 calibration/test |

## 5. 需求—能力差距矩阵

| 需求 | 更新前 | 本轮后 | 尚未关闭的差距 |
|---|---|---|---|
| 任意设备原始数据接入 | 数值表格、长时序、NPZ、多率 NPZ | 新增近期数据可消费/阻断目录 | 原始音频、动态视觉、通用 MAT/pickle/Parquet episode adapter 未实现 |
| 从“机理”形成模型 | 要求结构化设备卡，支持有限确定性机理特征 | 不变，增加错误宣称边界 | 不能可靠地从自然语言/PDF 自动抽取并验证机理；错误拓扑注入未联动选择 |
| 自动设计多阶段流水线 | 检测/定位/分类/辨识资格门与条件级联 | 保留 | 候选仍是有限成熟模型，不是开放式 NAS；上游错误传播仍需真实事件评测 |
| 选择“最适合”模型 | 聚合 validation 分数、实用差距、种子稳定性 | 增加最差工况、缺失传感器、字节数、复杂度代理、时延硬门禁 | 无全局最优保证；跨设备/长期漂移证据不足 |
| 缺失传感器 | 训练和推理直接失败 | 候选执行逐传感器中位数替代压力；TABULAR 推理允许声明组内回退并限制缺失组数 | raw audio/vision、长时序缺通道运行时适配未实现；中位数回退不是通用最优策略 |
| 训练集污染 | 假设可信正常数据 | PATH/TSAD-C 已进入证据和计划 | 自动去污染候选与污染率资格审计未实现 |
| 部署资源 | schema 声明但未执行 | 每候选实测 joblib 字节、结构复杂度代理、批量行时延并 fail closed | 目标硬件上的端到端内存、能耗、冷启动仍须现场验收 |
| 定位/根因/辨识语义 | 定位、分类、辨识均受监督与秩门控 | 不变 | 定位不等于根因；真实多故障根因证据和连续参数校准不足 |
| 公开复现 | 7 项历史目录、一个 UCI 原始适配器 | 16 项数据目录、12 项近期方法目录 | 新增数据仍多为 adapter `PLANNED`，不能宣称已跑出性能 |

## 6. 分解更新计划与实施状态

| 编号 | 优先级 | 更新点 | 验收条件 | 状态 |
|---|---:|---|---|---|
| G1 | P0 | 近两年数据/方法证据库 | 闭合时间窗口、主来源、许可/访问、机制变化、证据等级、风险可校验 | 已完成 |
| G2 | P0 | 工况切片效用 | manifest 声明切片；候选记录所有有效切片和最差分；可设权重与最低门槛 | 已完成 |
| G3 | P0 | 缺失传感器准入 | 显式 sensor groups、逐组压力分、最低门槛、推理回退和最大缺失组数 | TABULAR 软件闭环完成 |
| G4 | P0 | 部署资源硬门禁 | 模型字节、结构复杂度代理、时延逐候选记录并能淘汰 | 已完成软件闭环 |
| G5 | P0 | 能力/访问 fail-closed | raw audio/vision 和受限数据不得生成可执行 benchmark 结论 | 已完成目录门禁 |
| G6 | P1 | 污染训练候选 | PATH clean/contaminated 相同切分；去污染/稳健基线；污染率分层 | 待实现 |
| G7 | P1 | 原始音频/视觉 adapters | DCASE official split 与 XJTU-DV recording split；简单/深度模型资源匹配 | 待实现 |
| G8 | P1 | 基础模型候选准入 | UniTS/Mantis/UniFault 等以可选插件式候选进入；必须胜过简单基线且满足资源门 | 待实现，不默认启用 |
| G9 | P1 | 事件级现场指标 | 误报/小时、事件召回、检测延迟、VUS-PR、非 teacher-forced 级联 | 部分完成；现仅有 test-only 级联 |
| G10 | P2 | 生命周期闭环 | 漂移监测、再校准/重训触发、版本回滚、设备群外推 | 待实现 |

本轮按 G1 → G2/G3/G4 → G5 顺序实施。G6–G10 需要新增原始数据适配、依赖或真实部署证据，不能因路线图存在而标成已完成。

## 7. 组件机制卡

### G2：最差工况效用

- 输入/输出：候选在 validation 上的总体分数与 `slice_column=value` 分数 → 带最差切片惩罚的选择效用。
- 作用轴：工况/设备/负载等样本子群。
- 归纳偏置：部署安全取决于弱势工况，不应只优化总体平均。
- 表达边界：只覆盖 manifest 明示且样本数达到门槛的切片；未观测工况仍未知。
- 创新分类：评测与选择协议，不是新神经算子。

### G3：缺失传感器压力与回退

- 输入/输出：显式传感器组和训练中位数 → 每组缺失压力分、可选推理回退、缺失组审计列。
- 作用轴：特征/传感器。
- 归纳偏置：传感器不可用时用训练正常范围的中心值比静默 NaN 或任意零值更可审计。
- 表达边界：回退值不恢复缺失信息；只保证软件可执行，不保证诊断仍可靠。
- 失败模式：多个相关传感器同时丢失、漂移后中位数失效、机制特征对缺失源过敏。
- 创新分类：稳健性评测和部署策略。

### G4：资源门禁

- 输入/输出：拟合模型与 validation 批量 → joblib 字节、结构复杂度代理、每行推理时延和淘汰原因。
- 复杂度口径：线性/变换系数的数值元素数加树节点数；它是跨模型结构大小代理，不等同神经网络参数的统一理论量。
- 表达边界：开发机批量时延不等于目标 PLC/边缘设备端到端时延。
- 创新分类：系统约束与选择协议。

## 8. 公平比较矩阵

| 维度 | 强制一致项 | 允许变化项 | 输出 |
|---|---|---|---|
| 数据 | train/validation/calibration/test 角色；run/asset split；切片定义 | 候选特征视图 | hash、行数、泄漏守卫 |
| 训练 | 每一 rung 的 seed、训练行、监督目标 | 模型族内部拟合 | fit seconds |
| 效果 | 相同主指标、相同 validation | 候选预测 | aggregate、slice、sensor-dropout、selection score |
| 资源 | 相同 validation batch 和重复次数 | 模型结构 | bytes、parameter proxy、latency |
| 决策 | 相同实际差距和稳定性门槛 | 候选效用 | SELECTED/FIXED/ALTERNATIVES/UNSUPPORTED |

确认层实验仍须加入：B0 原始低复杂度基线、B1 数据/训练修正、S0 简单强基线、A0 完整方案、A1 去机理、C0 参数匹配、C1 时延匹配、M0 机制破坏、H0 假设切片、R0 缺失传感器/域迁移、V0 多种子。

## 9. 架构方案

```text
设备卡 + 数据清单 + 设计请求
             |
             v
      合同/监督/泄漏资格审查
             |
             v
  RAW / TEMPORAL / MECHANISM 特征视图
             |
             v
  有限候选：检测 -> 定位 -> 分类 -> 辨识
             |
             v
 validation 总体分 + 最差工况 + 缺失传感器
             |
             v
 字节/复杂度代理/时延硬门禁 + 多种子稳定性
             |
             v
 单一胜者、等价集合、仅组合或 UNSUPPORTED
             |
             v
 calibration 阈值/区间 -> test-only 非 teacher-forced 审计
```

G6–G10 作为独立扩展层，不会在尚未验证时进入默认候选 registry。

## 10. 实验与停止条件

1. 每个新增 adapter 必须校验官方文件 hash、许可、样本/通道数量和 split 单元。
2. 最差切片与缺失传感器分数低于请求门槛时，候选必须被淘汰而不是降级警告。
3. 资源超过任一硬门槛时，候选必须被淘汰；若全被淘汰，该阶段返回 `UNSUPPORTED`。
4. validation 无法稳定区分时返回 `ALTERNATIVES`；不允许用 test 选架构。
5. 音频/视觉没有原始 adapter 时，相关公开数据计划必须为 `BLOCKED`。
6. 预印本模型没有独立复现和匹配基线时，只能为 `RESEARCH_ONLY`。
7. 若连续三轮增加复杂度的收益低于实际差距、资源超标或只改善总体而恶化最差工况，则停止扩大模型。

## 11. 技术扫描结论

近期研究没有提供一个可直接替代 DOR、对任意工业设备都最优的通用模型。相反，公开证据支持以下工程结论：

- 数据与评测协议的可靠性经常比增加网络复杂度更关键；
- 基础模型值得作为候选，但零样本异常检测存在过度泛化风险；
- 缺失传感器、域迁移、污染训练和阈值选择是工业部署的一级问题；
- 音频、动态视觉、电流、SCADA、机器人 episode 需要不同输入 adapter 与 split 规则；
- “统一模型”不等于“统一证据”，不同阶段仍需要独立监督、校准和拒识。

因此本轮变化属于评测协议、系统约束、数据目录和成熟组件组合，没有“首次提出新神经组件”的声明。

## 12. 资源使用追踪

| 技能资源 | 对本轮工作的影响 |
|---|---|
| README / core-principles | 从任务失败与信息缺口定义收益，禁止复杂度先行 |
| component-ontology | 将变化归类为评测协议、数据表示、架构组合与系统约束 |
| research-protocol | 只用主来源，分开预印本、workshop、同行评审和官方挑战 |
| design-validation-protocol | 固化 B0–V0、公平比较、机制破坏和停止条件 |
| learning-protocol | 对选择器、缺失传感器与条件级联采用 L3 机制审计 |
| full-design-principles | 检查信息损失、资源复杂度、部署一致性与失败边界 |
| component-card template | 形成第7节组件机制卡 |
| comparison-matrix template | 形成第8节公平比较矩阵 |
| architecture-proposal template | 形成第1、5、9节方案与边界 |
| experiment-plan template | 形成第10节实验和停止条件 |
| technical-report-scan template | 形成第2、4、11节技术扫描 |
| component-registry template | 由每次运行的 `component_registry.csv` 和两个 benchmark JSON 提供结构化登记 |
| learning log v45 | 只决定分析深度，不作为方法有效性证据 |

## 13. 主来源索引

- DCASE 2025 Task 2：<https://dcase.community/challenge2025/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring>
- DCASE task paper：<https://arxiv.org/abs/2506.10097>
- Robot telemetry：<https://zenodo.org/records/17855686>
- Screw driving：<https://zenodo.org/records/15393134>
- PMSM inverter：<https://zenodo.org/records/14482932>
- Induction startup current：<https://zenodo.org/records/17048028>
- XJTU-DV：<https://zenodo.org/records/17447083>
- PATH：<https://arxiv.org/abs/2411.13951>
- FactoryNet：<https://arxiv.org/abs/2605.09081>
- Restricted spherical roller metadata：<https://zenodo.org/records/14856938>
- TSB-AD：<https://proceedings.neurips.cc/paper_files/paper/2024/hash/c3f3c690b7a99fba16d0efd35cb83b2c-Abstract-Datasets_and_Benchmarks_Track.html>
- UniTS：<https://proceedings.neurips.cc/paper_files/paper/2024/hash/fe248e22b241ae5a9adf11493c8c12bc-Abstract-Conference.html>
- UniFault：<https://arxiv.org/abs/2504.01373>
- Mantis：<https://openreview.net/forum?id=FDDkR53rim>
- TSAD-C：<https://proceedings.mlr.press/v286/ho25a.html>
- M2-STGN：<https://ieeexplore.ieee.org/document/11018134>
- IEEE industrial ASD：<https://ieeexplore.ieee.org/document/11330926>
- Zero-shot TSFM audit：<https://arxiv.org/abs/2607.12454>
- Adaptive conformal TSFM：<https://arxiv.org/abs/2604.20122>
