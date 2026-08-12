# DOR v0.6 开发更新说明

## 1. 版本状态

- DOR：0.5.0 DEV RC1 → **0.6.0 DEV PROTO**
- ISGN：2.2.0（Core 未修改）
- M2N：2.3.1（Core 未修改）
- FEC：2.4.0（Core 未修改）

v0.5 已正式完成 Fault Semantics / Localization 开发节点；v0.6 已开始 Propagation / Root-Cause Reasoner，但本包不把 v0.6 标记为完成。

## 2. v0.5 封口确认

发布硬化阶段发现并修复了一个合同一致性问题：部分 `UNKNOWN/ABSTAIN` 分支没有保留完整 policy/safety 字段，可能导致 runtime output 与 JSON Schema 不一致。修复后 DOR v0.5 fresh regression 为 **24 passed**。

因此最终 v0.5 DEV RC1 是修复后的发布包，而不是更早的 22-test 草案。

## 3. v0.6 新增 Root-Cause contract

新增：

```text
DOR-ROOT-CAUSE-RESULT/0.6
```

以及：

```bash
dor root-cause
```

原型从 v0.5 localization hypotheses 出发，结合：

- timestamped fault-specific local evidence；
- provenance-bound directed topology；
- temporal compatibility；
- evidence coverage；
- concurrent root set search；
- equivalent-set preservation。

## 4. 安全规则

明确固化：

- earliest anomaly ≠ root cause；
- topology upstream ≠ root cause；
- localization ≠ root cause；
- unprovenanced path 不可用于传播声明；
- 无 timestamp 不升级根因结论；
- multiple independent roots 允许；
- Test truth 不参与 reasoning。

## 5. NASA ADAPT 的当前用法

NASA ADAPT 是后续正式 root-cause benchmark 的优先目标。当前包内只有：

- v0.5 published-source semantic fixture；
- v0.6 source-constrained synthetic timing fixture。

当前环境已经取得官方 Experiment Log 文件，但完整 telemetry ZIP / diagram ZIP 因运行环境下载限制尚未进入本地包。因此 v0.6 不做真实 ADAPT accuracy 声明。

## 6. 本轮验证

```text
DOR v0.6 full tests      34 passed
v0.6 root-cause tests     10 passed
```

其余三个引擎本轮没有修改 Core；0.5 发布阶段的 fresh results 保持为：

```text
M2N 2.3.1  36 passed
FEC 2.4.0  73 passed
ISGN 2.2.0 doctor PASS
```

这些数字的 fresh 时间点和含义在 `VALIDATION_STATUS_CN.md` 中区分记录。

## 7. 为什么没有继续跳到 v0.7

v0.7 是 Fault Identification orchestration，需要异常窗口回灌 ISGN、normal-vs-fault mechanism comparison 和参数/结构变化辨识。

如果 v0.6 尚未在真实 fault-injection data 上把 localization 与 root cause 分开，就直接进入 v0.7，会把 propagation/root-cause 缺口隐藏在后续机理辨识结果中。因此当前开发纪律是：

> **先完成 v0.6 的真实 ADAPT Closure Gate，再正式推进 v0.7。**
