# DOR v0.6 Propagation / Root-Cause Reasoner 开发原型

## 1. 状态

**STARTED / EXECUTABLE PROTOTYPE / NOT CLOSED**

v0.5 已经能够把资格化 evidence 映射为 fault/component hypotheses，但明确不声明 root cause。v0.6 的目标是把“故障候选”和“真实根因候选”分开，并给传播声明加入可审计门槛。

## 2. 问题定义

输入：

```text
Fault Ontology 0.5
+ Localization Result 0.5
+ VALID timestamped evidence
+ directional topology with provenance
+ optional Mechanism-to-Component Map
```

输出：

```text
DOR-ROOT-CAUSE-RESULT/0.6
```

主要状态：

- `SUPPORTED`：唯一最小 root-cause set 获得足够支持；
- `ALTERNATIVES`：多个最小 root-cause sets 在当前证据下不可区分；
- `UNKNOWN`：localization 存在，但没有资格升级为根因结论；
- `ABSTAIN`：合同保留该状态，当前原型主要通过 UNKNOWN 表达信息不足。

## 3. v0.6 推理规则

### 3.1 根因不能由最早报警直接决定

最早 anomaly 可能是：

- 下游快速响应；
- 传感器故障；
- 不同并发故障；
- acquisition/timing bias。

因此 `earliest_anomaly_is_not_root_cause_rule=true` 被固化到输出安全字段。

### 3.2 topology 只提供传播可达性，不提供根因事实

只有以下关系可进入当前物理/信号传播图：

```text
POWER_FLOW / FLUID_FLOW / MECHANICAL /
THERMAL / CONTROL / SIGNAL
```

`FEEDBACK / OBSERVES / CONTAINS` 当前不作为物理传播边。

每条被使用的传播边必须有 provenance；没有 provenance 的路径只能作为审计信息，不能支撑 root-cause claim。

### 3.3 需要 fault-specific 本地时间支持

一个 fault hypothesis 如果只有 topology upstream 位置，而没有带时间戳的 fault-specific matched evidence，不得进入 root-cause eligible 集合。

### 3.4 下游证据必须满足时间相容性

若下游 evidence 时间早于 root local support（超过允许 tolerance），该 evidence 不得由该 root 解释，而记录为 temporal contradiction。

这不会自动把更晚出现的 upstream fault 判死刑；未被解释的 anomaly 可以要求第二个独立 root，或者使最终结果保持 UNKNOWN。

### 3.5 并发根因是一等结果

root-cause search 对不同 `exclusion_group` 的 faults 允许组合。默认最多搜索 3 个并发 root，并优先最小 cardinality root set。

因此：

```text
多个异常分支
≠ 必须单一根因
```

### 3.6 证据覆盖门槛

默认：

```text
coverage_required = 1.0
```

即 root set 必须解释全部可解释、VALID 的 anomaly evidence。该参数可调，但降低它属于显式 policy 变化，不能根据 held-out truth 调参。

## 4. 当前 NASA ADAPT fixture 的证据等级

`examples/nasa_adapt_v06_source_constrained_fixture/` 使用公开 NASA 资料支持的“E261 是 EY260 下游电压”关系构造了一个软件合同 fixture。

其中：

- 设备/变量关系：source-constrained；
- fault semantic：source-constrained；
- 时间戳与 evidence values：synthetic；
- 用途：测试 provenance + timing + propagation contract；
- 禁止用途：宣称真实 ADAPT root-cause accuracy。

当前 fixture 可以得到：

```text
localization  = SUPPORTED(EY260_failed_open)
root cause    = SUPPORTED([EY260_failed_open])
propagation   = EY260 -> E261_node
```

这只代表原型代码在受控合同例子中正确执行。

## 5. 当前单元/安全测试

v0.6 新增 10 个核心测试：

1. provenance + timing 支持唯一 upstream root；
2. 早于 upstream fault 的 downstream anomaly 不被错误归因，并允许两根因集合；
3. 无 provenance topology 不能产生 propagation claim；
4. 无 timestamp 保持 UNKNOWN；
5. 两个独立根因可形成 concurrent root set；
6. 等价 root sets 输出 ALTERNATIVES；
7. localization alone 不会被改名为 root cause；
8. FEEDBACK edge 不被偷当作物理传播路径；
9. sensor-health fault 不被静默重命名为 observed physical component fault；
10. 存在无法解释的 VALID anomaly 时，不强行输出单根因。

## 6. 为什么本轮不关闭 v0.6

NASA 官方 ADAPT 数据具备真实 EPS 硬件、受控故障注入、128 传感器与实验日志，是合适的 v0.6 目标。但当前运行环境只能取得部分公开资源；完整 telemetry ZIP 和 EPS diagram ZIP 尚未进入本地可重复执行链。

因此当前能够证明的是：

> Root-cause v0.6 的合同、保守推理规则与软件实现已经启动并通过单元级安全测试。

尚不能证明：

> DOR 已经在真实 ADAPT runs 上可靠区分 root cause 与 propagation symptom。

## 7. v0.6 Closure Gate

只有以下全部满足后，才把 v0.6 从 DEV PROTO 升为 DEV RC：

- 完整 ADAPT telemetry locally acquired + hash-bound；
- EPS topology 转换为 provenance-bound graph；
- Experiment Log 独立形成 fault-event truth manifest；
- evidence generation 不读取 truth；
- fault onset / mode / command context 与 telemetry 对齐；
- 单故障 + 多故障场景均进入 benchmark；
- root-cause reasoning 完成后才做 post-hoc truth audit；
- 至少验证 sensor fault / component fault / command fault 的区分；
- ambiguous cases 保持 ALTERNATIVES/UNKNOWN，而不是用 truth 修规则。
