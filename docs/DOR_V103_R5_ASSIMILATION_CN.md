# DOR v1.0.3 Satellite R5 选择性吸收与架构审计

日期：2026-08-18

输入包：`DOR_Diagnostic_Stack_v1.0.3_Satellite_R5_ActiveDiscovery_STABLE_20260817.zip`

SHA-256：`a3f529b81dfc2d2695d3d02a70f9e416b8ae0dc73fd04a1f5ef5baa84d4ba02b`

## 1. 技术报告扫描

| 对象 | R5 声称/变化 | 审计结论 | 分类 | 决策 |
|---|---|---|---|---|
| ISGN 2.4.0 | Active Mechanism Discovery：参数粒子、仿真似然、SMC、信息增益实验排序、M-open 扩展、可辨识性与稳定性判定 | 是本包唯一主要计算增量；有助于在有限机理假设空间内主动缩小歧义 | 架构组合、推断与实验设计 | 加固后吸收 |
| DOR 1.0.3 | Active Discovery advisory bridge | 上游桥接边界过宽：可转发任意实验字段，缺失隔离字段默认安全，后验和唯一性约束不足 | 系统安全接口 | 重写后吸收 |
| M2N 2.16.1 | 随包发布 | 无超越当前本地增强版的实质增量；整包覆盖会丢失符号编译/UDE | 既有组件 | 不覆盖 |
| FEC 2.4.2 | 随包发布 | 无超越当前本地增强版的实质增量；整包覆盖会丢失 mask 审计 | 既有组件 | 不覆盖 |
| `build/` 与重复产物 | 源码副本 | 发布冗余，会制造双源真相 | 构建产物 | 拒绝吸收 |

ZIP 共 1260 个条目，CRC、路径穿越与符号链接检查通过。R5 自报的 87/115/91/116 项测试和 68 episode 回放属于提供方声明；本仓库只把本地重跑记为软件回归事实，不升级为独立复现或工业诊断精度证据。

## 2. 问题合同

- 输入：结构化设备机理、可用实验/仿真合同、设备数据、任务与预算。
- 目标：在有限候选集合中设计检测—定位—分类—辨识流水线；当机理结构仍不充分时，提出最有信息价值的下一实验。
- 安全边界：隐藏真值、故障标签和真实结构不得参与实验排序；Active Discovery 只产生规划证据，不能直接升级为诊断、根因、机理认证或执行命令。
- 失败假设：近似似然会过度自信；假设空间可能不封闭；实验可能昂贵或危险；上游对象可能带 `payload/command/actuator`。
- 非目标：不承诺“任意设备全局最优模型”，不执行物理实验或执行器，不以单元测试替代设备盲测。

## 3. 公平比较矩阵

| 维度 | 合并前 R4 | R5 整包覆盖 | 本次 R5 选择性吸收 |
|---|---|---|---|
| 自动诊断设计 | 有 | 会丢失本地增强 | 保留 |
| 主动机理发现 | 无 | 有 | 有，并强化输入门 |
| 发现路由/Tabulated1D | 有 | 会回退 | 保留 |
| M2N 符号编译/UDE | 有 | 会回退 | 保留 |
| FEC mask held-out 审计 | 有 | 会回退 | 保留 |
| 真值隔离 | 多处显式 | 缺字段可被默认为安全 | 三字段必须逐项精确为 `false` |
| 后验合法性 | 不适用 | 未在 DOR 边界完整验证 | 非空、有限、非负、和为 1 |
| 唯一机理状态 | 不适用 | 可直接暴露 | 必须同时有稳定性和完整覆盖证明 |
| LIVE 建议 | 不适用 | 模块不执行，但桥可携带字段 | DOR 阻断 LIVE，剥离全部原始载荷 |
| 构建目录 | 规范 | 带重复 `build/` | 不采纳 |

条件性结论：R5 为“机理不确定时下一步测什么”补上了有价值的闭环，但没有把 DOR 变成全局自动建模器；其价值依赖高质量仿真银行、候选机理覆盖和外部安全实验流程。

## 4. 组件机制卡

### 4.1 ISGN Active Mechanism Discovery

- 一句话机制：用结构—连续参数联合粒子表示当前机理假设，以 repeat-aware 仿真样本构造稳健近似似然，按预期结构熵下降/成本选择下一实验。
- 输入/输出：参数粒子、仿真银行、观测、实验候选、预算 → 后验、实验排序、充分性/稳定性/可辨识性状态。
- 核心计算：`IG(e)=H(S|D)-E_y[H(S|D,y,e)]`；粒子权重由 Student-t 型仿真似然更新；实验按 `IG/cost` 排序。
- 作用轴与状态：结构、连续参数、实验和重复次数；状态为粒子权重及历史观测。
- 归纳偏置：有限候选、仿真分布可代表设备、结构信息优先于仅缩窄干扰参数。
- 表达边界：候选空间外机理不会凭空产生；M-open proposal 只有有界先验质量，不是证据。
- 失败模式：仿真—真实域差异、粒子覆盖不足、不可观测等价结构、单次异常观测与不安全实验。
- 本地加固：有限值/维度/成本/重复/模式/预算/阈值验证；无样本银行拒绝；proposal 质量不再静默裁剪。

### 4.2 DOR Active Discovery Advisory Firewall

- 一句话机制：把 ISGN 发现结果降格为不可执行规划建议，并在 DOR 边界验证隔离、概率和唯一性证据。
- 输入/输出：Active Discovery mapping → `DOR-ISGN-ACTIVE-DISCOVERY-ADVISORY/1.0`。
- 关键门：三个真值隔离字段必须显式为 `false`；后验和为 1；`UNIQUE_SUPPORTED` 需 `STABLE` 且覆盖完整；LIVE 一律阻断。
- 信息损失：有意删除 payload、command、actuator 和安全证据内容，只保留实验 ID、模式、重复、成本及安全资格布尔量。
- 结论边界：所有诊断、故障辨识、根因、机理认证和执行许可字段恒为 `false`。
- 分类：系统安全接口/证据治理，不是新的诊断模型。

## 5. 架构方案

```text
机理卡 + 数据 + 诊断请求 ──> DOR 有限候选自动设计 ──> 检测/定位/分类/辨识
        │
        └─ 机理结构仍有歧义
             └─> ISGN 参数粒子 + 仿真银行
                    ├─ SMC / predictive adequacy
                    ├─ evidence stability / M-open coverage
                    └─ expected information gain / cost
                              │
                              v
                   DOR Advisory Firewall
                    ├─ truth quarantine
                    ├─ posterior/uniqueness checks
                    ├─ strip execution payload
                    └─ ADVISORY_ONLY / BLOCKED
```

Active Discovery 不在自动设计主路径中隐式执行。外部实验系统若要采纳建议，仍需独立的安全审查、授权、执行和观测回传；本模块只排序，不执行。

## 6. 实验计划与结果

| ID | 唯一变化 | 问题 | 本地结果/停止门 |
|---|---|---|---|
| B0 | R4 基线 | 原能力是否稳定 | 历史 423 passed |
| A0 | Active Discovery 计算模块 | 信息增益、SMC、M-open 与可辨识性可运行吗 | ISGN 专项及全量通过 |
| A1 | DOR firewall | 隔离/概率/唯一性是否 fail closed | DOR 专项及全量通过 |
| S0 | LIVE + 原始载荷 | 是否可能越过 DOR | BLOCKED，载荷不出现在输出 |
| S1 | NaN/负值/不归一后验 | 是否静默接受 | BLOCKED/异常拒绝 |
| S2 | 缺失覆盖或稳定性 | 是否产生唯一结论 | BLOCKED |
| C0 | 四组件回归 | 是否破坏既有能力 | DOR 107 + ISGN 125 + M2N 95 + FEC 118 = **445 passed**；另 1 optional skip |
| V0 | 新设备多种子盲测 | 是否提升诊断性能 | 未执行，不得宣称 |
| V1 | 安全实验闭环 | 是否可用于真实设备自动试验 | 未执行，保持禁止执行 |

测试数量只证明当前软件合同与回归覆盖，不证明 Active Discovery 在某工业设备上获得最优诊断精度。

## 7. 组件登记

机器可读登记见 `component_registry_dor_v103_r5.csv`。采用级别只表示本仓库选择性集成，不代表行业采用度。

## 8. 资源使用追踪

| Skill 资源 | 实际影响 |
|---|---|
| `README.md` | 采用“问题/效果优先、成本显式计价、证据边界优先”的审计主线 |
| `core-principles.md` | 区分新观测、确定性重编码与候选先验；proposal 不作为证据 |
| `component-ontology.md` | 将 Active Discovery 归为推断/实验设计组合，将 bridge 归为系统安全接口 |
| `research-protocol.md` | 将用户 ZIP 视为一手提供方材料，并把自报实验与本地复现分开 |
| `design-validation-protocol.md` | 形成 B0/A0/A1/S0—S2/V0—V1 及停止条件 |
| `learning-protocol.md` | 按 L3 审计状态、路由和部署边界，不伪造个人学习记录 |
| `full-design-principles.md` | 补齐张量、状态、作用轴、信息损失、复杂度与部署一致性 |
| `component-card.md` | 形成第 4 节两张机制卡 |
| `comparison-matrix.md` | 形成第 3 节公平对照 |
| `architecture-proposal.md` | 形成第 2、5 节问题合同和计算图 |
| `experiment-plan.md` | 形成第 6 节消融/压力/未完成验证门 |
| `technical-report-scan.md` | 形成第 1 节逐项创新审计 |
| `component-registry.csv` | 形成机器可读组件登记 |
| `neural_network_component_learning_log_v45.csv` | 只用于确定分析深度；未修改该日志 |

## 9. 不可宣称与停止条件

- 不能声称给出任意设备数据和机理即可得到全球最优诊断模型；当前是有限候选、合同约束、证据可拒答的设计器。
- 不能把高后验质量单独解释为真实机理；必须检查预测充分性、稳定性、可辨识性和假设空间覆盖。
- 若没有可信仿真银行、候选机理、明确实验成本或安全资格，停止 Active Discovery 并返回阻断/不充分。
- 若候选在全部可准入实验下观测等价，报告 `UNIDENTIFIABLE`，不继续增加模型复杂度来制造确定性。
