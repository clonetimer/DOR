# DOR v0.5 Fault Semantics / Mechanism-to-Component / Localization 开发说明

版本：0.5.0 DEV RC1  
阶段：DOR Core Development  
组件：DOR 0.5.0 / ISGN 2.2.0 / M2N 2.3.1 / FEC 2.4.0

## 1. 为什么 v0.5 必须出现

DOR 0.3 解决了控制平面，DOR 0.4 解决了 reference/context 的资格问题。但到 v0.4 为止，DOR 仍主要知道：

- 当前任务是否有资格；
- 哪个模型/证据可用；
- 什么时候必须 ABSTAIN。

它还没有一个正式的诊断语义层去回答：

> 当前 evidence 到底支持“哪个部件、哪种 fault hypothesis”？

v0.5 的目标因此被限定为 **Fault Semantics + Conservative Localization**，而不是提前做 Root Cause。

## 2. Frozen contracts

本版本新增：

- `DOR-FAULT-ONTOLOGY/0.5`
- `DOR-MECHANISM-COMPONENT-MAP/0.5`
- `DOR-LOCALIZATION-RESULT/0.5`
- `DOR-LOCALIZATION-BATCH/0.5`

### 2.1 Fault Ontology

Fault Ontology 至少描述：

- component；
- sensor；
- command；
- component topology；
- fault mode；
- fault target component；
- expected evidence；
- contradiction evidence；
- provenance；
- 可选 `exclusion_group`。

Ontology 的语义是“**允许被检验的 hypothesis space**”，不是“系统中这些 fault 一定存在”。

### 2.2 Mechanism-to-Component Map

用于把 ISGN/M2N 的 mechanism-level evidence 对齐到：

```text
mechanism_id
  ↕
component_ids
sensor_ids
provenance
```

Mechanism binding 必须有 provenance。v0.5 不允许仅凭变量名/attention/feature importance 自动生成物理绑定。

## 3. Evidence contract

v0.5 localizer 接收 evidence table。最小字段：

```text
evidence_id
scope_type
scope_id
state
strength
validity
```

可选：

```text
value
timestamp
provenance
case_id
```

`validity` 非有效状态的 evidence 不进入 scoring。

### 3.1 CLASS evidence

已有 FEC class evidence 可以用：

```text
state = CLASS
value = <fault/condition class>
```

但 class evidence 仍必须与 ontology 的 `expected_value` 精确对齐，不能把 arbitrary classifier output 偷换成 root-cause probability。

## 4. Localization v1 算法

v0.5 采用可审计 hypothesis scoring，而不是端到端 root-cause neural network。

每个 fault hypothesis 的 score 来自：

1. 明确 expected evidence 的匹配；
2. 矛盾 evidence 的惩罚；
3. 仅作为有限辅助的 generic component/mechanism scope evidence。

默认 policy：

```text
min_support = 0.45
retain_margin = 0.08
max_contradiction_fraction = 0.35
candidate_set_cap = 32
```

这些是 v0.5 开发默认值，不是跨设备固定的物理常数。

### 4.1 Generic anomaly 不能识别 fault mode

例如：

```text
pump = ABNORMAL
```

最多说明 pump scope 值得关注，不能直接得出：

```text
pump_internal_leakage_severe
```

因此 generic scope bonus 被限制，默认不足以单独让具体 fault mode 通过 `min_support`。

### 4.2 并发 fault 与 alternatives 分离

同一个 `exclusion_group` 里的 modes 互斥，例如：

```text
pump leakage weak
pump leakage severe
```

如果二者 evidence 无法区分，返回 `ALTERNATIVES`。

不同 exclusion groups 可以并发，例如：

```text
pump leakage
+
accumulator pressure loss
```

v0.5 会返回一个多 fault `supported_fault_set`，不会因为存在多个 fault 就错误标记为 alternatives。

## 5. 输出语义

### SUPPORTED

被保留的每个互斥 fault group 都能被当前 evidence 唯一解析；不同 groups 可以构成并发 fault set。

### ALTERNATIVES

至少一个互斥 group 内仍有多个近似等价 hypotheses，保留 `candidate_fault_sets`。

### UNKNOWN

有效 evidence 存在，但不足以支持 ontology 中的具体 fault hypothesis。

### ABSTAIN

例如：

- 没有有效 evidence；
- candidate-set combinatorics 超过安全上限；
- 其他合同/有效性条件不满足。

## 6. NASA ADAPT fixture 的角色

NASA ADAPT 是 NASA Ames 的 aerospace-representative Electrical Power System testbed。官方数据页说明该 EPS 支持可重复受控 fault insertion，并记录 voltage/current/temperature/switch 等传感器；公开数据有 128 sensors、2 Hz sampling，并同时存在 physical/software faults。

v0.5 当前发布包中加入的是 **published-source fixture**，只转写公开论文中能够明确支持的少量语义：

- Relay stuck-open / stuck-closed；
- relay command + feedback 的诊断关系；
- E261 voltage sensor stuck 示例；
- ST516 speed-sensor offset/stuck 示例。

这套 fixture 用于验证：

```text
ontology schema
mechanism/component binding
sensor-vs-component fault semantics
command/feedback evidence
mutually-exclusive fault modes
```

**它不是完整 ADAPT Dataset 的替代品。** 当前环境已确认 NASA 官方 `dataset_text.zip / Experiment_Log.xls / AdaptHTMLdiagrams.zip` 资源存在；`Experiment_Log.xls` 可访问，但完整 ZIP 未被纳入本发布包。v0.6 前必须以完整 ADAPT telemetry/topology 进行真实回归，或明确保持 DEFERRED。

## 7. Hydraulic 真实数据 handoff 验证

本轮没有重新训练 Hydraulic 模型，也没有读取 Test truth 来做 reasoning。

流程是：

```text
既有 Pump FEC/DOR calibrated component-class result
+
既有 Accumulator FEC/DOR calibrated component-class result
        ↓
只保留两条链都已 SUPPORTED 的共同 sample_id
        ↓
生成 component CLASS evidence
        ↓
DOR v0.5 ontology/localizer
        ↓
localization_batch.json
        ↓
冻结结果后，才读取 truth_posthoc.csv
```

共同 Test samples 为 40；其中在两条输入链上均满足 `SUPPORTED` 的 qualified samples 为 29。

v0.5 localization 对这 29 个 cases 全部返回 `SUPPORTED`；post-hoc exact fault-set match 为 29/29。

正确解释：

> 这证明 DOR 能在真实设备数据链上把**已校准、component-scoped class evidence**保真转换为 fault ontology / multi-fault localization semantics。

错误解释：

> “DOR 已经从原始 Hydraulic sensor data 100% 完成 fault isolation。”

后者并未被本实验支持。

## 8. 本版本明确不做什么

v0.5 不实现：

- downstream propagation path；
- root-cause inference；
- earliest alarm = root cause；
- abnormal residual magnitude = root cause probability；
- fault identification / parameter drift inversion；
- fault recovery / maintenance decision。

对应输出固定：

```text
root_cause_claimed = false
propagation_inferred = false
```

## 9. DOR 0.6 入口条件

v0.6 只在以下条件成立时继续：

1. topology edge 的方向和语义有来源；
2. fault injection 的 component/type/time 能够作为独立 truth；
3. sensor/command timing 可用；
4. root cause 与 propagated symptoms 有可检验区别；
5. multiple concurrent faults 不被单 root-cause 假设强行压缩。

NASA ADAPT 是 v0.6 的第一优先数据；在完整数据不能进入本地 runtime 前，0.6 只能进行合同和单元级开发，不能宣布真实闭环。
