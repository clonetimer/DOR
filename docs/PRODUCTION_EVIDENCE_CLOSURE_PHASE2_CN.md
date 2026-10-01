# DOR 1.0.5 Production Evidence Closure Phase 2

## 1. 本阶段目标

Phase 2 不扩展诊断阶段，而是继续关闭“高分是否能转化为生产证据”的漏洞。新增三类可执行合同：

1. **Prospective benchmark qualification**：先验证是不是冻结后的新 session/new asset，再比较冻结阈值；
2. **Deployment operations evidence**：区分连续运行暴露与 benchmark selected segments，禁止把后者的 rate 写成生产 false alarms/hour；
3. **Public source intake**：对外部官方 archive 做本地 hash、文件名、大小（若已知）和 ZIP member 约束，避免“来源 URL 写对了但本地 bytes 不可追溯”。

本阶段仍然保持：ASSISTED_POC 可支持；ASSISTED_PRODUCTION 与 SAFETY_CRITICAL_AUTONOMY 阻断。

## 2. Prospective 证据门

新增 CLI：

```bash
dor prospective-evaluate \
  --policy release/PROSPECTIVE_ACCEPTANCE_POLICY_NEXT.json \
  --benchmark <new_benchmark.json> \
  --output prospective_result.json
```

生产证据 credit 需要同时满足：

- benchmark 绑定**精确冻结 policy SHA-256**；
- candidate set、hyperparameters、acceptance thresholds 在 test 打开前冻结；
- test labels 不进入 candidate selection / calibration；
- 数据是 independent holdout；
- 数据来自新的 session/asset/mission slice；
- 当前模型设计前不存在该 test 的历史结果知识；
- 最终冻结指标全部达标。

### 当前 OPSSAT 控制实验

当前 channel-wise remediation 的冻结指标：**16/16 通过**，但 evaluator 返回：

```text
status = INELIGIBLE
production_gate_credit = false
```

原因不是性能差，而是：历史 test 已经被看过，而且它不是 policy freeze 之后的新 mission slice。因此当前结果只能继续作为 development remediation evidence。

## 3. 运行时告警负担：selected-segment proxy

新增 CLI：

```bash
dor deployment-ops-evaluate \
  --spec validation/production_evidence/phase2/opssat_selected_segment_ops_spec.json \
  --predictions validation/production_evidence/phase2/opssat_predictions_with_duration.csv \
  --output ops_result.json
```

当前 OPSSAT feature 中 `duration` 是 segment duration（秒）。将 supported nominal selected segments 的 duration 作为**仅供 benchmark 解释的分母**，当前 remediation 得到：

- supported nominal selected-segment exposure：约 **28.3575 h**；
- false-positive segments：**31**；
- proxy：约 **1.093 FP / selected nominal segment-hour**。

但 OPSSAT benchmark 是领域专家选择出的 telemetry segments，而不是连续 mission timeline，所以 evaluator 返回：

```text
status = PROXY_ONLY
production_gate_credit = false
```

这条值**禁止**写成“卫星生产误报率 1.093/h”。真正的 DEPLOYMENT_OPERATIONS gate 仍要求连续观察窗口、已知数据缺口、event onset、检测延迟、CPU/RAM/latency/cold-start 和 operator workload。

## 4. 官方数据 intake

新增：

```bash
dor source-intake-check \
  --spec release/source_intake_specs/UCI_HYDRAULIC_447.json \
  --artifact <official_archive> \
  --output intake.json
```

当前已固化：

- UCI Hydraulic 官方 ZIP URL + 18 个必须唯一出现的文件（`profile.txt` + 17 sensors）；
- NASA ADAPT `dataset_text.zip` / `Experiment_Log.xls` / `AdaptHTMLdiagrams.zip` 官方 URL；
- NASA legacy resource listing 中可核对的文件大小。

本地 intake PASS 只证明“本地 bytes 与声明约束一致”；若没有 CI/browser 的下载记录，它不会伪造“已经证明 bytes 一定来自该 URL”。UCI adapter 在 intake 后还会继续验证 `profile=(2205,5)` 和每路传感器矩阵宽度。

## 5. 仍然阻断的生产证据

- 新 session / 新 asset / 新 mission slice 的 prospective benchmark；
- 真实 distribution shift：sensor calibration drift、maintenance shift、跨设备/跨 session；
- 有明确 availability semantics 的 operational missing-sensor benchmark；
- continuous false alarms/hour/day 与 event detection delay；
- sustained latency、CPU/RAM、cold start；
- NASA ADAPT 真实 fault-injection telemetry + Experiment Log + topology replay；
- hazard analysis 与 closed-loop safety。

## 6. 证据解释纪律

1. metric PASS ≠ prospective eligible；
2. selected-segment rate ≠ operational rate；
3. local archive hash ≠ download-channel provenance；
4. fixture/root-cause contract PASS ≠ real fault-injection accuracy；
5. 当前 1.0.5 的主要进展是让这些边界由程序阻断，而不是靠 README 提醒。
