# DOR 公开数据集与原始文献差距审计

审计日期：2026-08-13。目标不是罗列模型，而是判断 DOR 能否在给定设备机理和数据后，选择可验证的检测、定位、分类、辨识流水线。机器可读事实源为 `benchmarks/public_datasets.json`；网页信息会变化，目录保存最后核验日期而不假装永久有效。

## 1. 结论

DOR v0.7.0.dev0 已经能生成有限候选集内、带拒答和审计记录的多阶段可运行流水线，但还不能声称满足“面向任意设备自动设计最合适诊断模型”的完整目标。它达到的是受合同约束的开发预览，主要差距是：真实公开数据闭环不足、时序候选仍依赖压缩特征、多故障定位不进入自动设计器、连续辨识没有独立不确定性校准、跨设备/跨工况外推没有默认门控。

## 2. 公开数据证据矩阵

| 数据集 | 可提供的证据 | 必须采用的划分单位 | 对 DOR 最有价值的用途 | 关键边界 |
|---|---|---|---|---|
| [UCI Hydraulic](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems) | 17 个传感器、2,205 个 60 秒周期；冷却器、阀、泵泄漏、蓄能器与稳定性目标 | 完整周期/工况块 | 多目标分类、检测及多标签定位入口 | 离散状态不能证明连续参数辨识；当前仓库只有派生特征视图 |
| [CWRU Bearing](https://engineering.case.edu/bearingdatacenter/welcome) | 正常与内圈/滚动体/外圈人工故障，多负载与故障尺寸 | 完整记录和试件 | 快速检查时序检测/分类候选 | 人工点蚀不能代表自然损伤；随机窗口会泄漏记录身份 |
| [Paderborn Bearing](https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/data-sets-and-download) | 人工与真实损伤，振动和电流，多工况 | 物理轴承试件 | 检验跨试件、跨工况和人工到真实损伤迁移 | CC BY-NC 4.0；必须单独报告迁移设置 |
| [NASA IMS](https://data.nasa.gov/dataset/ims-bearings) | 轴承全寿命振动序列 | 完整寿命实验 | 变化检测与退化辨识 | 终止故障记录不是稠密的故障起点/类别真值 |
| [FEMTO PRONOSTIA](https://data.phmsociety.org/nasa/) | 加速寿命振动/温度与 RUL 挑战数据 | 完整轴承寿命 | 时序退化与 RUL 扩展 | RUL 监督不等同于定位或故障类型监督 |
| [NASA ADAPT](https://data.nasa.gov/dataset/adapt-dataset) | 电压、电流、温度、开关、实验日志和系统图；受控故障注入 | 完整实验 run/场景 | 机理拓扑约束的检测—定位—分类—参数故障链 | 本仓库 ADAPT fixture 是合成合同样例，不是真实 benchmark |
| [UCI MetroPT-3](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset) | 真实地铁空气压缩机长时序与维护故障区间 | 时间顺序的维护 episode | 真实运维异常检测和时间迁移 | 单设备数据不能证明车队迁移；维护区间不是精确起点 |

选择逻辑：UCI Hydraulic 最适合先关闭多目标监督合同；ADAPT 最适合检验机理拓扑和根因安全规则；Paderborn 比 CWRU 更适合检验真实损伤迁移；IMS/FEMTO 主要支持退化/RUL，不能被用来填补故障分类真值；MetroPT-3 适合实际时间漂移和误报率评估。

## 3. 原始文献技术扫描

- NASA 的 [ADAPT benchmark 报告](https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20110011109.pdf) 比较同一测试台上的十种诊断算法，并同时描述故障、数据和度量。这支持 DOR 采用“共同执行合同 + 多种算法保留”的设计，而不是预设一个万能模型。
- [ADAPT 概率模型诊断案例](https://data.nasa.gov/dataset/probabilistic-model-based-diagnosis-for-electrical-power-systems)展示机理模型、贝叶斯网络和实时推理的组合价值。DOR 应将拓扑/物理规则作为有出处的约束或独立专家，不把它伪装成从数据发现的结构。
- [ROCKET](https://arxiv.org/abs/1910.13051)、[MiniRocket](https://arxiv.org/abs/2012.08791)表明随机卷积变换加线性分类器可兼顾时序精度与训练成本；[InceptionTime](https://arxiv.org/abs/1909.04939)提供可扩展深度卷积对照。三者应作为原始时序候选阶梯，而不是立即替换当前可审计统计基线。
- [有限假阳性的 conformal prediction sets](https://proceedings.mlr.press/v162/fisch22a.html)直接对应多标签定位的集合输出与假阳性预算。DOR 需要在独立 calibration split 上生成部件集合，不能用 softmax 最大类冒充并发定位。

未采用的捷径：不以论文中的平均榜单替代目标设备 validation；不把模型容量、训练技巧或数据增强登记成新机理组件；不把公开数据上的单次随机切分称为泛化证明。

## 4. 组件卡

| 组件 | 输入/输出 | 假设与失效模式 | 当前状态 |
|---|---|---|---|
| 数据资格门 | 数据清单、split、group → 阶段准入/拒绝 | 标签语义正确，group 能封锁同源样本；错误 group 会产生隐蔽泄漏 | 已实现 |
| 机理特征编译器 | 有出处的关系 + 传感器 → 确定性派生特征 | 错拓扑/错误单位会系统性误导；派生量不是新观测 | 已实现，缺单位量纲检查 |
| 时序表示 | 多通道序列 → 样本特征 | 统计/FFT 压缩损失相位和事件顺序 | 仅基线；原始时序候选待实现 |
| 阶段模型选择器 | 候选、预算、validation → 胜者/等价集/拒答 | 只在声明搜索空间内“最适合”；小验证集不稳定 | 已实现有限搜索 |
| 条件级联门 | 检测结果 → 后续阶段是否执行 | 上游漏检会遮蔽后续正确模型 | 已实现；需端到端非 teacher-forced 度量 |
| 多标签定位器 | 异常证据 → 部件候选集合 | 标签相关、并发故障和阈值校准 | 自动设计器未实现；旧 ontology localizer 独立存在 |
| 参数辨识器 | 特征 → 故障参数及区间 | 参数变化需可辨识；点预测置信度不等于覆盖率 | 点回归已实现，独立区间校准未实现 |
| 公开 benchmark 门 | 目录 + 请求阶段 → 可评估范围/禁止宣称 | 目录会过时，需要核验日期和官方来源 | 本轮已实现 |

## 5. 候选架构比较矩阵

| 方案 | 机理注入 | 原始时序 | 多故障 | 不确定性 | 计算/审计性 | 结论 |
|---|---:|---:|---:|---:|---|---|
| 当前统计/频带 + 传统模型 | 确定性特征 | 否 | 单标签 | 检测阈值/候选等价集 | 低成本、最易审计 | 保留为必选基线 |
| MiniRocket + 线性头 | 可与机理特征晚融合 | 是 | 可扩展多标签头 | 需外接 calibration | 中低成本、近确定性 | 下一时序默认候选 |
| InceptionTime 多任务头 | 可中/晚融合 | 是 | 支持共享编码和多头 | 需温度/conformal 校准 | GPU 成本高、消融复杂 | 数据规模达标后进入确认层 |
| 机理图 + 时序编码器 | 拓扑消息传递 | 是 | 天然部件级 | 需结构与预测双重校准 | 机制强但错图敏感 | ADAPT/多部件数据专项，不设为通用默认 |
| 单一端到端“大模型” | 难验证 | 是 | 理论可行 | 常未校准 | 搜索与审计成本最高 | 当前不准入默认候选集 |

## 6. 目标架构提案

```text
设备卡/机理图 ──单位与出处检查──┐
                                ├─ 数据 IR ─ 候选表示阶梯 ─┐
公开数据/用户数据 ─泄漏与任务门──┘                          │
                                                            ├─ 检测集合
             统计基线 → MiniRocket → 深度时序确认候选       ├─ 多标签定位集合
                                                            ├─ 故障分类集合
                   有限预算、多种子、独立 calibration/test ─┤
                                                            └─ 参数估计区间
                              ↓
             条件 DAG + OOD/拒答 + 端到端非 teacher-forced 审计
```

自动设计的目标函数必须是带约束的多目标选择：阶段性能、最坏工况性能、校准误差、延迟/内存、跨种子稳定性和机理压力测试共同决定 Pareto 可接受集。若多个候选落在 practical margin 内，返回 `ALTERNATIVES`，不得制造唯一最优。

## 7. 分期实验与验收计划

| 更新点 | 数据/对照 | 核心度量 | 通过门槛 | 失败处置 |
|---|---|---|---|---|
| U1 公开数据证据门 | 七个官方数据源目录；非法目录负例 | 合同完整性、阶段决策、出处 | 重复 ID/缺字段 fail closed；阶段边界可机器读取 | 阻断 benchmark |
| U2 UCI raw adapter | UCI 全 2,205 周期；当前派生特征作对照 | 行数/通道/标签/hash、group 泄漏 | 官方原始文件 hash 固定；周期不跨 split | 不发布精度 |
| U3 多标签定位 | UCI 四部件目标；单标签模型对照 | micro/macro F1、每部件召回、集合 FP | calibration 不参与训练；并发标签可表达 | 返回定位不支持 |
| U4 原始时序候选 | UCI/Paderborn；统计基线、MiniRocket、InceptionTime | macro F1、最坏工况、训练/推理成本 | 参数/预算匹配，多种子，试件隔离 | 保留统计基线 |
| U5 辨识区间 | 合成可辨识基准 + 有连续物理真值数据 | MAE、区间覆盖/宽度、秩 | 独立 calibration 达目标覆盖；秩不足拒答 | 仅点估计或拒答 |
| U6 端到端级联 | ADAPT/设备数据完整 run | 漏检、误报/小时、定位集合、分类、参数、端到端覆盖 | 非 teacher-forced；上游错误完整传播 | 禁止阶段分数替代端到端结论 |
| U7 机理压力测试 | 正确/移除/打乱/错误拓扑 | 性能变化、错误自信率 | 错机理不能比无机理更自信且无告警 | 机理分支降级/拒绝 |

本轮随后实施了 U2 的严格 ZIP/multirate adapter，并以官方 ZIP 的 PS1/FS1/TS1 三种采样率完成 2,205 周期代表性真实源冒烟；同时完成 U3、U5 软件闭环。U4 已增加 temporal-pyramid 前置候选，U6/U7 已增加可执行审计。三传感器冒烟不能外推成完整 17 传感器架构比较；MiniRocket/InceptionTime、事件延迟/误报率、错误拓扑准入联动仍未关闭，因此 U4/U6/U7 只能标记部分完成。

## 8. 资源使用登记

| 资源 | 本审计中的作用 |
|---|---|
| skill `README.md` / `SKILL.md` | 约束工作流、证据分层、输出结构与禁止的新颖性宣称 |
| `references/component-innovation-taxonomy.md` | 区分新算子、变体、组合、训练和系统优化 |
| `references/mechanism-analysis.md` | 编写组件卡的输入输出、假设、失效模式与可观测性 |
| `references/search-and-evidence.md` | 只用官方数据页、原论文/作者材料支撑技术事实 |
| `references/selection-guide.md` | 使用约束化多目标选择，而非单一榜单 |
| `references/evaluation-and-ablation.md` | 设计匹配预算、消融、压力测试与独立 calibration |
| `references/architecture-composition.md` | 形成条件 DAG、晚融合和安全降级方案 |
| 六个 templates | 分别落实为组件卡、比较矩阵、架构提案、实验计划、技术扫描和 registry 字段 |
| `learning_log.md` | 避免把确定性派生特征、新训练策略或容量增加误写为新传感证据/新组件 |

## 9. 不可宣称

- 不能称 DOR 已经找到任意设备的全局最优模型。
- 不能用同一数据集 validation 选模后的 test 表现替代跨设备泛化。
- 不能把受控人工故障、加速寿命故障和真实现场故障视为同一分布。
- 不能把 `PARTIAL`、`DERIVED_ONLY` 或合成 fixture 写成完整真实数据闭环。
- 截至本审计，没有足够证据提出新的神经网络算子；建议均为已知组件的受约束组合与验证路线。
