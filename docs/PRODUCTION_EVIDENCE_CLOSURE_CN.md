# DOR 1.0.4 Production Evidence Closure Phase 1

## 1. 本阶段目标

本阶段不继续扩张诊断功能，而是解决一个更基础的问题：**软件合同通过、平均 benchmark 指标较高，并不等于生产落地有效。**

DOR 1.0.4 新增 `DOR-PRODUCTION-EVIDENCE-GATE/1.0`，把证据明确分成三层：

1. `ASSISTED_POC`：允许工程师辅助使用和诊断 ML qualification；
2. `ASSISTED_PRODUCTION`：要求 prospective gate、真实 distribution shift、missing-sensor、新 session/asset、部署运维指标；
3. `SAFETY_CRITICAL_AUTONOMY`：额外要求 hazard analysis、closed-loop safety 和真实 fault-injection。

当前结论：

```text
ASSISTED_POC              SUPPORTED
ASSISTED_PRODUCTION       BLOCKED
SAFETY_CRITICAL_AUTONOMY  BLOCKED
```

这不是降级，而是把“能做什么”和“还不能承诺什么”变成机器可验证的发布合同。

## 2. OPSSAT 通道级 remediation

旧 release 的 supported test rows：523 / 529。

旧单一链结果：

- precision = 0.6638
- recall = 0.7000
- F1 = 0.6814
- balanced accuracy = 0.8028
- FPR = 0.0944
- TP/FP/FN/TN = 77 / 39 / 33 / 374

1.0.4 新增 `scripts/run_opssat_channelwise_benchmark.py`。它不把所有 telemetry channel 强行交给同一个异常评分器，而是：

```text
train-normal -> 每通道拟合有限候选
validation   -> 每通道选择候选
cal-normal   -> conformal-style p-value / threshold calibration
test         -> 仅最终报告
```

候选集包括标准化能量、多个 PCA reconstruction、KNN distance、Ledoit-Wolf Mahalanobis 和 Isolation Forest。稀疏 calibration channel 仍 fail-closed。

新的 remediation 结果：

- supported = 523 / 529 = 0.9887
- precision = 0.7615
- recall = 0.9000
- F1 = **0.8250**
- balanced accuracy = **0.9125**
- FPR = **0.0751**
- TP/FP/FN/TN = 99 / 31 / 11 / 382

相对旧 release：

- F1 +0.1436
- balanced accuracy +0.1097
- false negatives 33 -> 11
- false positives 39 -> 31

最弱的有异常 test channel `CADC0892` 现在：

- recall = 0.8571
- F1 = 0.7059
- FPR = 0.0870

这验证了“channel heterogeneity 需要 context-specific model selection”这一工程判断。

### 2.1 不能过度解释

这轮 remediation **不是新的盲测**。在引入新的 channel-wise candidate set 之前，历史 OPSSAT test 结果已经被项目看到。因此：

- 当前改进可以证明 remediation 方向有效；
- 当前改进不能被包装为独立的 prospective qualification；
- 下一份独立 session/asset 数据必须使用已经冻结的 `release/PROSPECTIVE_ACCEPTANCE_POLICY_NEXT.json`。

## 3. Hydraulic 独立工程压力测试

新增 `scripts/run_hydraulic_practical_stress.py`，重新执行：

- split-group overlap audit；
- exact feature duplicate audit；
- train 拟合 + validation model selection；
- nuisance-only baseline；
- RAW vs RAW+MECHANISM 同模型对照；
- synthetic feature-noise stress；
- synthetic missing-sensor stress。

结果：

- validation 选择：HistGradientBoosting；
- test accuracy = 0.9708；
- test macro-F1 = **0.9706**；
- nuisance-only baseline accuracy = 0.25，macro-F1 = 0.1462；
- RAW+MECHANISM 相对 RAW 的同模型 macro-F1 增量 = **+0.00845**，仍低于 request 中 0.02 practical margin。

人工 Gaussian feature stress：

| 噪声尺度（train std 比例） | macro-F1 |
|---:|---:|
| 0.01 | 0.9498 |
| 0.05 | 0.8066 |
| 0.10 | 0.6405 |
| 0.20 | 0.5451 |

人工 missing-sensor stress 中，当前最敏感的是 `FS2` 整组缺失，macro-F1 降到约 **0.8518**。

这些结果只说明模型对人工扰动的脆弱性，不等价于真实 distribution shift / sensor outage qualification。因此 production gate 仍为 BLOCKED。

## 4. 发布与回归修复

### 4.1 `verify_all_tests.py` clean-source bug

原脚本执行 DOR tests 时只加入 `src/`，但 DOR integration tests 直接 import `isgn.contracts`。在未安装四组件 wheel 的 clean source 环境中会 collection error。

1.0.4 已改为：

- DOR suite 同时加入 DOR / ISGN / M2N / FEC source roots；
- 各 engine suite 继续使用自己的 source root；
- 不依赖 ambient site-packages 才能收集 DOR integration tests。

当前环境 fresh 结果：

- DOR: **111 passed**
- M2N: **95 passed**
- FEC: **118 passed**
- ISGN: **125 tests collected**；monolithic full run 超过当前执行窗口，因此本阶段不制造新的 “125 passed” 声明。

OPSSAT / hydraulic 当前 assisted POC route 不 admit ISGN，所以这不阻断当前 POC 结论；但它明确阻断把本轮证据外推到 Active Discovery / root-cause production。

### 4.2 release hygiene

原 ZIP 带 `.pytest_cache` / `__pycache__`，与 `verify_stack.py` 的 release-hygiene 规则矛盾。1.0.4 打包流程会在最终 verification 前移除这些缓存，并保留自动检查。

## 5. 仍未关闭的 production gates

当前明确 BLOCKED：

- `PROSPECTIVE_ACCEPTANCE_POLICY`：政策已冻结，但尚未在新的独立数据上执行；
- `DISTRIBUTION_SHIFT`：只有人工 stress，没有真实 drift/session shift；
- `MISSING_SENSOR`：只有人工 mask，没有 operational availability benchmark；
- `NEW_SESSION_OR_ASSET`：没有冻结后的新资产/新 session；
- `DEPLOYMENT_OPERATIONS`：没有 false alarms/hour、detection delay、CPU/RAM、持续 latency、operator workload；
- `HAZARD_ANALYSIS`；
- `CLOSED_LOOP_SAFETY`；
- `REAL_FAULT_INJECTION`：NASA ADAPT 仍是 source-constrained fixture，不是真实 telemetry root-cause benchmark。

## 6. 下一阶段唯一主线

下一阶段应是 **Production Evidence Closure Phase 2**，而不是继续堆诊断 stage：

1. 用冻结政策跑新的 OPSSAT-like mission slice / telemetry period；
2. 使用完整 UCI Hydraulic 2,205-cycle raw archive，按 session/group 做 prospective split；
3. 完成真实 sensor drift / outage / recalibration protocol；
4. 获取并转换 NASA ADAPT real fault-injection telemetry + Experiment Log；
5. 加入部署指标与 operator workload；
6. 上述证据完成后再重新评估 `ASSISTED_PRODUCTION`。
