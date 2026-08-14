# DOR v1.0.2 Satellite R4 选择性吸收与架构审计

日期：2026-08-14

输入包：`DOR_Diagnostic_Stack_v1.0.2_Satellite_R4_20260814.zip`

SHA-256：`31521fcc3eee57cb7102b62bd865e5be8200b4599438a59f221c485b4694b9b7`

## 1. 技术报告扫描

| 对象 | 原包版本 | 原包声称 | 实际计算/合同变化 | 分类 | 本地证据 | 采用 |
|---|---:|---|---|---|---|---|
| DOR | 1.0.2 | R4 集成保护 | ISGN 观测不足时辨识拒答；预测时距由部署合同绑定 | 架构组合/安全合同 | 相关测试本地重跑 | 选择性采用 |
| ISGN | 2.3.0 | 时间基、时间充分性、Tabulated1D | 新增时间戳完整性与有限历史反事实预检；有界一维插值 | 数据合同、组件变体 | ISGN 109 项通过 | 采用并保留发现路由 |
| M2N | 2.16.1 | 波形表示竞争、递归时距选择 | 候选级波形配置与 H 步闭环验证风险 | 架构组合/训练评测 | M2N 95 项通过 | 采用并保留符号编译/UDE |
| FEC | 2.4.2 | 资产/舰队 Reference 治理 | 资产身份、工况身份、逐资产 sentinel 和晋级门 | 治理/系统安全 | FEC 118 项通过 | 采用并保留 mask 审计 |

原包 README、源码、Schema、测试与发布 JSON 属于上游一手材料，达到 E1；其自报测试属于官方/提供方实验声明。本文只把本地重新执行结果记为仓库内实验事实，不把它升级为跨团队独立复现，也不声称这些机制首次提出。

## 2. 问题合同

- 任务：从设备机理卡、数据合同和诊断目标设计检测—定位—分类—辨识有限流水线。
- 部署输入：只允许部署时可获得的传感器、工况、时间戳和预声明时距。
- 基线：DOR 0.8.0.dev0 / ISGN 2.2.0 / M2N 2.12.0+dor.1 / FEC 2.4.0。
- 失败假设：不规则采样可能被误当连续样本；一步验证可能错误选择长时距模型；跨资产参考库可能被无授权复用。
- 目标：上述三类错误必须 fail closed，同时不丢失自动设计、安全符号编译、UDE、模态校准和有限诊断候选能力。
- 资源边界：本轮验证为当前 CPU/GPU 开发环境的软件回归；没有形成新设备盲测性能证据。

## 3. 公平比较矩阵

| 维度 | 合并前 | R4 整包覆盖 | 本次选择性合并 |
|---|---|---|---|
| 自动诊断设计 | 有 | 会丢失 | 保留 |
| 时间基审计 | 无 | 有 | 有 |
| 观测时间充分性 | 无 | 有 | 有 |
| 有界经验本构 | 无 | 有 | 有 |
| 递归时距选择 | 无 | 有 | 有，且只接受 DOR 合同注入 |
| 安全模型加载 | `weights_only=True` | 部分路径回退为 `False` | 保留 `True` |
| 可执行 UDE | 有 | 会降为合同状态 | 保留 |
| 舰队 Reference | 无 | 有 | 有 |
| mask held-out 审计 | 有 | CLI 会丢失 | 保留 |
| 结论范围 | 有限候选最适合 | 有限候选最适合 | 有限候选、合同约束、可拒答 |

条件性结论：R4 对时间观测、预测时距和跨资产校准治理有明确增量；整包覆盖会造成现有功能和安全回退，因此不适合直接替换。

## 4. 组件机制卡

### 4.1 ISGN Observation Gate 2.3

- 一句话机制：先审计时间基，再比较即时观测与有限历史对导数的 held-out 可预测性；若历史显著更强，则拒绝“即时全状态充分”的辨识声明。
- 输入/输出：`X[N,T,Dx]`、`U[N,T,Du]`、可选 `sample_time_s[N,T]` → gate checks/status。
- 核心规则：`delta_t = diff(sample_time_s)`；时间充分性比较 `NRMSE_inst` 与 `min_lag NRMSE_history`，达到预设增益阈值时返回 `OBSERVATION_INSUFFICIENT`。
- 作用轴：轨迹和时间轴；状态为无持久状态的预检报告。
- 信息变化：不增加新观测；用同一数据构造反事实表示，可能揭示遗漏状态、延迟或传感器动态。
- 边界：历史增益不能指出缺失机理是什么；lag 系数不得解释成物理参数。
- 分类：数据/表示合同与组件变体，非新基础算子；证据 E1 + 本地测试事实。

### 4.2 Bounded Tabulated1D

- 一句话机制：在已声明的一维本构坐标和同一工况内，用线性/PCHIP 插值完成未知曲线，越界、重复输入多值和跨工况均拒绝。
- 输入/输出：`x[n], y[n]` → 有序 knots 与区间内预测。
- 公式：`y_hat = Interp({(x_i,y_i)}, x)`，仅当 `x in [min(x_i), max(x_i)]`。
- 作用轴：样本轴；无训练—推理状态差异，模型状态为 knots。
- 信息损失：压缩为结点；未观测区间、滞回和多值关系不可恢复。
- 分类：组件变体/表示方案；不是任意维预测后备模型。

### 4.3 M2N Horizon-qualified Temporal Selection

- 一句话机制：在显式声明的 H 步闭环时距上评价有限时序候选，并保留一步风险作为诊断信息。
- 输入/输出：独立上下文 `X[c,T,F]`、合同 `mode,H` → 候选风险、选择/组合及冻结模型。
- 计算：`x_hat[t+k+1]=f_theta(x_hat[t+k-L+1:t+k])`；选择风险使用 held-out H 步端点标准化 RMSE。
- 作用轴：上下文内时间；递归状态是最近 `L` 步预测窗口。
- 边界：原生闭环只允许 `input_columns == target_columns`，不得发明未来外生输入；不自动等同 RUL。
- 分类：评测/架构组合，不是新时序基础算子。

### 4.4 FEC Asset/Fleet Reference Governance

- 一句话机制：校准 Reference 绑定资产和外部工况身份，舰队扩展要求每个成员资产以独立可信正常 sentinel 通过支持与校准健康审计。
- 输入/输出：Reference bundle、资产身份、工况身份、sentinel → admitted/blocked/robustness decision。
- 作用轴：资产、Reference 构造变体；状态存在于候选、shadow、promotion lifecycle。
- 归纳偏置：默认资产不交换，只有显式证据才能扩大作用域。
- 失败模式：sentinel 健康假设错误、样本不足或有限 Reference 构造敏感；这些状态均不得静默晋级。
- 分类：治理和系统安全，不称为新神经网络组件。

## 5. 架构方案

```text
DeviceMechanismCard + DatasetManifest + Data + DiagnosticRequest
                │
                ├─ ISGN Observation Gate ── insufficient ──> UNIDENTIFIABLE
                │             │ sufficient
                │             v
                │       bounded mechanism evidence
                │
                ├─ DOR forecast binding ──> M2N finite candidates
                │                              │ H-step validation
                │                              v
                │                       selected/retained models
                │
                └─ declared asset scope ──> FEC Reference gate
                                               │ calibrated evidence
                                               v
                     DOR detection → localization → classification → identification
```

关键接口：

- `DatasetManifest.deployment.forecast_evaluation` 是预测时距唯一权威来源；
- `reference_scope.kind=FLEET_GROUP` 在简单 DOR score bridge 中直接阻断，必须走 FEC 逐资产 commissioning；
- ISGN normal/abnormal 任一侧观测不足，DOR differential identification 返回 `UNIDENTIFIABLE`；
- 自动设计候选仍是有限注册集合，validation 选优不构成全局 NAS 最优。

收益分解：新证据为可选真实时间戳和独立资产 sentinel；其余主要是表示、评测合同和治理偏置，不伪称创造了新物理信息。

## 6. 实验计划与结果

| ID | 唯一变化 | 回答问题 | 结果/门槛 |
|---|---|---|---|
| B0 | 合并前四套全量 | 当前能力 | 历史 348 passed |
| A0 | 完整选择性合并 | 总体兼容 | 423 passed |
| A1 | R4 专项测试 | 新增机制是否工作 | 相关专项通过 |
| S0 | 保留一步默认 | H 步机制是否可选 | 无声明时不启用递归 |
| M0 | 缺失/冲突时距 | 是否 fail closed | 抛出 `ValueError` |
| H0 | 时间断点/历史显著增益 | 预期切片是否拒答 | observation gate 测试通过 |
| R0 | 越界/多值 Tabulated1D | 是否拒绝外推/混合工况 | 测试通过 |
| C0/C1 | 参数/FLOPs 匹配 | 性能是否来自机制 | 本轮无新精度主张，留给设备数据实验 |
| V0 | 多种子与跨设备盲测 | 泛化是否稳定 | 未执行，不得声称 |

本地全量结果：DOR 101、ISGN 109、M2N 95、FEC 118，共 423 项。测试数量证明软件回归覆盖，不证明工业设备诊断精度。

## 7. 组件登记

机器可读登记见 `component_registry_dor_v102_r4.csv`。登记只记录达到 E1 的具体合同/组件，采用等级为本仓库内 A1，不外推行业采用程度。

## 8. 资源使用追踪

| Skill 资源 | 对本次工作的实际影响 |
|---|---|
| `README.md` | 确认问题优先、效果优先但成本显式计价的总原则 |
| `core-principles.md` | 区分新增时间戳证据与历史窗口确定性重编码 |
| `component-ontology.md` | 将四项更新分别归到数据合同、组件变体、架构组合和系统治理 |
| `research-protocol.md` | 以提供方源码/测试/发布 JSON 为一手来源，并禁止“首次提出”声明 |
| `design-validation-protocol.md` | 建立 baseline、机制破坏、压力测试和停止条件 |
| `learning-protocol.md` | 时间状态/路由按 L3 审计，Tabulated1D 按 L2，未伪造个人掌握记录 |
| `full-design-principles.md` | 固化输入、状态、作用轴、信息损失和部署一致性检查 |
| `component-card.md` | 形成第 4 节四张机制卡 |
| `comparison-matrix.md` | 形成第 3 节三方案公平比较 |
| `architecture-proposal.md` | 形成第 2、5 节问题合同与最终计算图 |
| `experiment-plan.md` | 形成第 6 节实验矩阵和未完成门 |
| `technical-report-scan.md` | 形成第 1 节逐项创新审计 |
| `component-registry.csv` | 形成机器可读组件登记文件 |
| `neural_network_component_learning_log_v45.csv` | 只用于选择 L2/L3 分析深度；不修改、不冒充学习完成 |

## 9. 不可宣称与停止条件

- 不得声称输入任意设备机理和数据即可得到全球最优模型；只能在已注册候选、数据质量和预算合同内选择最合适者或保留候选集。
- 不得把 423 个软件测试等同诊断准确率、飞行验证或跨设备泛化。
- 若低复杂度候选达到等价效果、递归时距收益低于波动、时间信息部署不可得、或舰队 sentinel 不充分，应停止增加复杂度或拒绝部署。
