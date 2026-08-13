# 参考文献吸收实施：阶段 1

日期：2026-08-13。状态：**阶段成果完成并通过四工具全量回归**。

## 1. 问题合同与成果

目标不是把论文名称加入候选列表，而是打通以下可验证链路：

```text
设备机理/ISGN候选表达式
  -> M2N安全语法与domain审计
  -> 冻结可微符号模块
  -> 已知向量场 + 可学习残差
  -> 固定步长RK4轨迹
  -> 下游诊断特征/模型
  -> FEC按真实模态mask独立校准与审计
  -> DOR资格判定或拒答
```

本阶段新增：

- ISGN：按导数质量、部分机理、问题类型和维数选择 SINDy、轨迹残差或 SymMatika 路线；隐式 SymMatika 明确标记为外部运行时缺口。
- M2N：安全表达式编译、严格/guarded domain、精确梯度、符号+神经残差组合、可微固定步长 RK4 UDE。
- FEC：独立测试集上的逐 mask 覆盖率、集合大小、singleton/empty 比例、top-class accuracy 和 unsupported 比例审计。
- DOR：在运行 FEC 前计算 `2^M-1` mask 样本预算；只按总样本足量不再视为 mask 资格充分。
- 研究目录：登记 SymMatika、PHM Part II、Neural Compiler、Conformal Fusion 和 PFM；PFM 明确暂缓。

信息边界：本轮没有生成新的工业事实；符号表达式、确定性特征和仿真数据都不能替代真实设备观测。两个航天器数据集仍是合成证据，且既有正常校准轨迹仅为 3/5 条，不能升级为部署就绪结论。

## 2. 技术报告扫描

| 方法 | 日期/来源 | 实际计算变化 | 最近邻 | 分类 | 本阶段采用 |
|---|---|---|---|---|---|
| SymMatika | 2025-08-18, arXiv:2507.03110 | GP operator feedback + motif reuse + implicit metric | GP/PySR/AI Feynman | 搜索算法 | ISGN 路由与外部候选门；未冒充原生运行时 |
| PHM Part II | 2025-09-25, arXiv:2509.21207 | PHM learning/observational bias taxonomy | physics-informed ML | 综述/分类体系 | DOR 资格原则 |
| Neural Compiler | 2026-05-21, arXiv:2605.22498 | 受限程序转冻结可微模块 | 手写 PyTorch/SymPy/UDE | 系统与架构组合 | 采用安全表达式子集 |
| Conformal Fusion | 2026-08-07, arXiv:2608.07183 | vacuous missing evidence + mask Mondrian calibration | modality dropout/CP-MDA | 架构与校准组合 | 复用已有算法并新增部署审计 |
| PFM | 2026-07-03, arXiv:2607.03524 | 在感知特征空间监督 flow matching | flow matching/distillation | 训练目标 | 与工业诊断主线不匹配，暂缓 |

上述近期工作均按作者公开预印本处理。论文自报实验是 E2 官方实验，不是 DOR 工业独立复现；本轮没有“首次提出”或“新基础算子”声明。

## 3. 组件机制卡

### 3.1 M2N Safe Symbolic Compiler

- 一句话机制：把 allowlist 表达式 AST 降为 PyTorch 运算，同时拒绝属性访问、索引、任意函数和控制流。
- 输入输出：输入字典中的张量可为标量、`[B]` 或可广播形状；输出保持广播结果形状和输入 dtype/device。
- 最小例：`y=k*x+sin(u)`，`x=[1,3]`、`u=[0,π/2]`、`k=2`，得到 `[2,7]`。
- 梯度：`∂sum(y)/∂k=sum(x)=4`，`∂y/∂x=k`；测试已验证。
- domain：strict 模式对零除、非正 log、负 sqrt fail closed；guarded 模式显式改变边界语义并写入 contract。
- 信息操作：只重编码已声明方程，不增加观测；结构冻结，只有显式参数可训练。
- 复杂度：前向为 AST 运算数 `O(N_ops)`；解释式 Python dispatch 不宣称优于手写/编译 kernel。
- 失败：错误机理会被精确执行；周期参数、奇点、幂定义域和广播误用仍可能导致优化失败。
- 最近邻差异：相比手写 PyTorch，收益是系统生成和合同，不是单方程精度；相比论文 Neural Compiler，本实现只覆盖小型表达式语言，无循环、矩阵程序或通用递归。

### 3.2 M2N Fixed-step Hybrid UDE

- 一句话机制：`dx/dt=f_known(t,x,u)+f_neural(x,u)`，用可微固定步长 RK4 积分。
- shape：`x0[...,D]`、`times[T]`、可选 `controls[T,...,U]`，输出 `[T,...,D]`。
- 状态：沿时间递归；控制量在每个积分区间左保持。
- 梯度：梯度穿过所有 RK4 状态和已知/残差路径；长轨迹内存随 `T` 增长，未实现 adjoint。
- 归纳偏置：可信项硬编码，只让神经分支补未知项。
- 失败：刚性系统、自适应步长需求、长轨迹梯度和 known/residual 功能重叠。
- 证据：简谐振子 100 个步长达到 `2e-5` 终点误差测试；这只是数值单元测试，不是工业效果。

### 3.3 ISGN Discovery Router

- 一句话机制：用问题语义和观测资格选择发现方法，不用模型声望替代数据条件。
- 规则：可靠导数+稀疏库→PySINDy/native sparse；导数不可靠+部分机理→trajectory residual；grammar uncovered→physics reducer/SymMatika；无导数也无先验→BLOCKED。
- 失败：路由正确不等于候选方程正确；库遗漏、状态不可观测和噪声仍会失败。
- 最近邻差异：SINDy在固定库中稀疏拟合；SymMatika搜索表达式树；轨迹残差通过求解器避免直接导数训练。

### 3.4 FEC Mask Calibration Audit

- 一句话机制：只在独立测试标签上，按真实 presence mask 分别测量经验覆盖和集合效率。
- 输入：`score__<class>`、`mask__<modality>`、`true_label`；输出 JSON 审计合同。
- 非梯度组件：conformal 阈值和审计不参与模型反向传播。
- 失败：每个 mask 的 exchangeability 破坏时，经验通过不能建立未来覆盖；mask 数随模态数指数增长。
- 反事实：未见 mask 必须计为 unsupported，不能借用完整模态阈值。

## 4. 公平比较矩阵

| 维度 | 手写已知方程 | 安全符号编译 | 纯神经动力学 | Hybrid UDE |
|---|---|---|---|---|
| 已知结构 | 硬 | 硬 | 无 | 硬核心+学习残差 |
| 单方程数值优势 | 基线 | 无 | 不适用 | 不适用 |
| 系统生成 | 低 | 高 | 中 | 高 |
| 梯度 | 精确 autograd | 精确 autograd | autograd | 经求解器 autograd |
| 未知项 | 手工修改 | 需另接残差 | 全部学习 | 局部学习 |
| 主要风险 | 人工错误 | 语言/domain边界 | 外推和数据量 | credit assignment/solver稳定性 |
| 当前成熟度 | A4 | DOR内 A2 | 成熟方法 | DOR内 A2固定步长 |

公平边界：尚未在等训练预算、等参数量和等硬件下比较故障诊断效果，所以不能声称 Hybrid UDE 优于纯神经或手写方案。

## 5. 架构方案

可证伪假设：当设备存在可信但不完整的连续时间机理时，冻结已知项并仅学习缺失项，应在机理适用域外推和参数效率上优于等预算纯 MLP；若打乱已知项后效果不下降，或等参数 MLP 达到同等外推，则降低机理解释强度。

| 方案 | 机制 | 复杂度 | 用途 | 停止条件 |
|---|---|---:|---|---|
| S0 | 手写已知模型+零残差 | 低 | 数值基线 | 已知模型明显错误 |
| B0 | 纯 MLP/现有表格模型 | 中 | 数据驱动基线 | — |
| A0 | 安全编译+局部残差+RK4 | 中 | 不完整连续动力学 | 刚性/延迟/credit assignment 不合格 |
| A1 | 自适应/adjoint UDE | 高 | 长轨迹或刚性问题 | 当前阶段未实现 |

接口保持为 `ISGN candidate -> M2N compilation contract -> model/score pack -> FEC reference -> DOR decision`。M2N 的编译精确性字段固定与物理真实性分离。

## 6. 机制验证实验计划

| ID | 实验 | 唯一变化 | 回答问题 | 状态/否证条件 |
|---|---|---|---|---|
| B0 | 纯数据模型 | 无已知项 | 当前水平 | 待设备任务 |
| S0 | 手写方程 | 无编译器 | 编译是否只提供工程收益 | 待延迟测试 |
| A0 | 编译+UDE | 完整方案 | 是否总体有效 | 数值单测通过 |
| A1 | 去掉神经残差 | 无未知项 | 残差是否必要 | 待真实任务 |
| C0 | 等参数 MLP | 参数匹配 | 是否只是容量 | 待真实任务 |
| C1 | 等延迟 MLP | 延迟匹配 | 是否只是计算 | 待目标硬件 |
| M0 | 打乱/符号翻转已知项 | 破坏机理 | 是否因正确机理改善 | 待真实任务 |
| H0 | 机理适用域内/外切片 | 数据切片 | 是否改善预期切片 | 待真实任务 |
| R0 | 奇点、刚性、长步长、缺失mask | 压力条件 | 何时失效 | domain/mask单测已覆盖部分 |
| V0 | 多种子 | 初始化 | 结果是否稳定 | 待训练实验 |

通过门槛需由具体设备请求声明。默认不得用 test 选择模型；FEC 需独立 calibration/test，逐 mask 至少达到配置的样本门槛。

## 7. 全量验证结果

| 工具 | 结果 |
|---|---:|
| DOR | 55 passed |
| M2N | 40 passed |
| FEC | 75 passed |
| ISGN | 99 passed |
| 合计 | **269 passed** |

新增 13 项测试均包含失败边界：代码注入、非法 domain、UDE shape/梯度、导数路线误用、隐式运行时过度声明、未见 mask 和 mask 样本组合爆炸。

## 8. 组件登记与结论边界

机器可读登记见 `validation/reference_assimilation_stage1/component_registry.csv`，验证摘要见 `validation/reference_assimilation_stage1/stage_result.json`。

- 已证明：上述接口和数学行为由单元测试支持，四工具旧测试未回归。
- 实验支持：仅限单元/合成数值行为，不是工业准确率改善。
- 仍是假设：UDE 和符号编译能改善两个航天器数据集或真实设备诊断。
- 不得声称：自动发现唯一真实机理、全局最优架构、任意缺失模态覆盖、硬件/飞行部署就绪。

## 9. 资源使用追踪

| Skill资源 | 对实施的影响 |
|---|---|
| README | 从“符号到运行时”和“mask到审计”的失败缺口出发 |
| core-principles | 区分新观测、确定性编译、表示和系统收益 |
| component-ontology | 编译器归为系统/架构组合，UDE归为连续状态架构，审计归为评测层 |
| research-protocol | 只采用本地原始论文并登记日期、来源、最近邻和证据等级 |
| design-validation-protocol | 固化 B0/S0/A0/A1/C0/C1/M0/H0/R0/V0 |
| learning-protocol | 对编译、UDE、发现路由和多模态校准执行 L3 机制分析 |
| full-design-principles | 检查 shape、mask、状态、梯度、信息损失、复杂度和部署差异 |
| component-card模板 | 形成第3节四张机制卡 |
| comparison-matrix模板 | 形成第4节公平比较 |
| architecture-proposal模板 | 形成第1、5节架构和信息合同 |
| experiment-plan模板 | 形成第6节实验与否证条件 |
| technical-report-scan模板 | 形成第2节近期工作扫描 |
| component-registry模板 | 形成机器可读登记表 |
| learning log v45 | 只用于确定 L3 分析深度，不作为组件有效性证据 |
