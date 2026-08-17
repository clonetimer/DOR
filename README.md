# DOR Diagnostic Stack v1.0.3+timer.1 — Satellite R5-derived Integration

> 本版本在 R4 能力上选择性吸收 R5 Active Mechanism Discovery：用结构—参数粒子和信息增益排序减少机理歧义，并以 DOR advisory firewall 阻断真值、LIVE 执行和原始载荷越界。自动设计、安全符号编译、UDE、模态掩码审计均保留；仍不能据此宣称任意设备全局最优。

源码组件：DOR `1.0.3+timer.1` / ISGN `2.4.0+timer.1` / M2N `2.16.1+timer.1` / FEC `2.4.2+timer.1`。

2026-08-18 fresh full regression：DOR 107 + ISGN 125 + M2N 95 + FEC 118 = **445 passed**，另 1 个可选依赖测试跳过。吸收决策和边界见 `docs/DOR_V103_R5_ASSIMILATION_CN.md`，实施台账见 `docs/IMPLEMENTATION_LEDGER_DOR_V103_R5_CN.md`。

Active Discovery 输出必须经过不可执行的 DOR 防火墙：

```bash
dor active-discovery-advisory --input isgn_active_discovery.json --output advisory.json
```

该入口不会执行实验；缺失真值隔离声明、非法后验、无稳定性/覆盖证明的唯一机理状态或任何 LIVE 建议都会返回 `BLOCKED`。

## 自动诊断模型设计预览

```text
结构化设备机理卡 + 表格/多通道时序数据 + 四阶段监督合同
                         ↓
数据/泄漏/可辨识性资格审查 + 机理特征编译
                         ↓
检测 → 定位 → 分类 → 辨识 条件 DAG 与有限候选竞争
                         ↓
可运行模型包 + 阶段拒答状态 + 机制审计报告
```

```bash
dor auto-design --device-card device_card.json --manifest dataset_manifest.json \
  --data data.csv --request auto_design_request.json --out design_output
dor auto-predict --bundle design_output/pipeline_bundle --data deployment.csv --output predictions.csv
```

当前“最合适”严格表示声明预算、候选集合与 validation 合同内的稳定最佳或等价集合，不表示全局 NAS 最优。缺少定位、分类或辨识监督时相应阶段返回 `UNSUPPORTED`，不会隐式生成伪标签。详见 `docs/AUTO_DIAGNOSTIC_ARCHITECT_CN.md` 和实施台账 `docs/IMPLEMENTATION_LEDGER_AUTO_DESIGN_CN.md`。

R4 的递归预测必须显式绑定部署时距：

```bash
dor forecast-bind --manifest dataset_manifest.json --device-card device_card.json --output bound_device_card.json
m2n temporal-design --spec bound_device_card.json --data sequence.csv --out temporal_output
```

若未声明时距、声明冲突、需要未来外生输入、时间基不完整或跨资产 Reference 未经 FEC commissioning，相关路径拒答而不是猜测。

2026-08-13 更新后，候选选择还可执行最差工况、缺失传感器、模型字节、结构复杂度代理与开发机推理时延门禁。近两年数据/论文证据、明确差距和 G1—G10 计划见 `docs/RECENT_2024_2026_EVIDENCE_GAP_AND_UPDATE_PLAN_CN.md`；原始音频/视觉、污染训练、基础模型和生命周期闭环仍属于未完成项。

公开 benchmark 在运行前可先审查其证据范围与泄漏约束：

```bash
dor dataset-catalog
dor benchmark-plan --dataset-id uci-hydraulic \
  --stages DETECTION LOCALIZATION CLASSIFICATION IDENTIFICATION

dor uci-hydraulic-prepare --archive condition-monitoring-hydraulic.zip \
  --out prepared_uci_hydraulic

dor spacecraft-prepare --dataset-root whole_spacecraft_phm_dataset_v6_0 \
  --out prepared_spacecraft --feature-profile causal_raw
```

目录会明确区分“数据存在该监督”和“当前 DOR 已能安全消费该监督”。研究证据、差距与 U1—U7 更新路线见 `docs/PUBLIC_DATASET_AND_LITERATURE_GAP_AUDIT_CN.md`。

航天器适配器逐样本读取文件树，禁止把注入参数、事件真值或标签路径用作特征，并按 `trajectory_pair_id` 整组切分。`causal_raw` 是去除状态/有效位、创新量和残差的严格视图；`deployable_full` 只适合作为对照。ADCS v5.2.2 与 PHM v6.0 的全量结果目前是 `CONDITIONALLY_QUALIFIED`，不是飞行部署结论，详见 `docs/SPACECRAFT_DATASET_FULL_TEST_CN.md`。

---

# DOR Diagnostic Stack v0.6.0 — Propagation / Root-Cause Development Prototype

**当前源码组件基线：DOR 1.0.3+timer.1 / ISGN 2.4.0+timer.1 / M2N 2.16.1+timer.1 / FEC 2.4.2+timer.1**

新增入口：`m2n model-plan`、`m2n temporal-design`、`m2n predict-temporal`、`dor identify-plan`、`dor identify`、`dor multi-task-arbitrate`、`dor vertical-validate` 和 `dor freeze-gate`。设计与证据边界见 `docs/EXTERNAL_TOOLKIT_INTEGRATION_DESIGN_CN.md`。

参考文献吸收阶段 1 已在当前开发树加入安全符号编译、fixed-step UDE、ISGN 发现路由、FEC mask held-out audit 和 DOR mask 样本预算门。实现边界、269 项全量回归和遗留实验见 `docs/REFERENCE_ASSIMILATION_STAGE1_CN.md`；这些开发能力尚未升级上述发布基线版本号。

> 生命周期：DOR 仍处于开发收敛期。v0.6 已形成可执行的传播/根因推理原型，但**本版本不宣布 v0.6 开发节点完成**。完整 NASA ADAPT telemetry、实验日志、方向拓扑与独立 fault truth 尚未进入可重复本地 benchmark，因此当前根因结果只证明合同、安全规则与源约束 fixture 可运行，不证明真实 ADAPT 根因诊断准确率。

## 已完成的前置节点

- v0.3：Control Plane / DatasetManifest / TaskQualification / DiagnosticPlan；
- v0.4：Context Hierarchy / Reference Applicability；
- v0.5：Fault Ontology / Mechanism-to-Component Map / Fault Localization v1。

## v0.6 本轮新增

```text
DOR-LOCALIZATION-RESULT/0.5
+ directional topology with provenance
+ VALID timestamped evidence
+ optional mechanism-component binding
        ↓
Propagation eligibility
        ↓
minimal concurrent root-set search
        ↓
SUPPORTED / ALTERNATIVES / UNKNOWN
        ↓
DOR-ROOT-CAUSE-RESULT/0.6
```

### 根因声明必须同时满足

1. fault-specific local support 有时间戳；
2. 至少有一个非局部 downstream anomaly 被方向 topology path 解释；
3. propagation path 每条边均有 provenance；
4. downstream timing 与 root support 时间相容；
5. 配置要求的 anomaly evidence coverage 被满足；
6. 若存在多个同等最小 root sets，则保留 `ALTERNATIVES`；
7. 并发独立根因允许形成 root set，不强制单根因。

## 明确禁止的捷径

- earliest anomaly ≠ root cause；
- upstream topology node ≠ root cause；
- localization result ≠ root cause；
- unprovenanced adjacency 不得支持 propagation；
- 没有时间信息不得把 localization 升级成 root-cause claim；
- Test fault log 不得参与 reasoner rule 选择。

## 新 CLI

```bash
dor root-cause \
  --ontology fault_ontology.json \
  --localization localization.json \
  --evidence evidence.csv \
  --output root_cause.json
```

可选参数：

```text
--mechanism-map
--max-root-set-size
--coverage-required
--alternative-margin
--temporal-tolerance
--max-path-depth
```

v0.3~v0.5 的 CLI 全部保留。

## 当前验证边界

- DOR v0.6 DEV PROTO：34 tests passed；
- 其中 root-cause 专项安全测试：10 passed；
- NASA ADAPT：仅有 source-constrained synthetic timing fixture；
- 该 fixture 的 topology 关系来自公开 NASA 文献，但时间戳/evidence 是为了软件合同验证而构造；
- **不能把 fixture 的 `SUPPORTED` 解释为真实 ADAPT benchmark accuracy。**

## v0.6 尚未关闭的门槛

- 完整 ADAPT `dataset_text.zip` 未进入本地运行环境；
- EPS diagram 尚未转换为完整 provenance-bound topology graph；
- Experiment Log 尚未形成 time-aligned independent truth manifest；
- 还没有真实 fault-injection run 上的 propagation/root-cause post-hoc benchmark；
- 因此尚不进入 v0.7 Identification orchestration 的正式开发闭环。

详见：

- `docs/DOR_V0.6_ROOT_CAUSE_PROTO_CN.md`
- `docs/USER_GUIDE_CN.md`
- `docs/DEVELOPMENT_UPDATE_CN.md`
- `docs/VALIDATION_STATUS_CN.md`
