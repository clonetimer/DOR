# DOR 四工具合体故障诊断工具包 — v1.0.5+timer.1 Production Evidence Closure Phase 2 使用指南

源码版本：DOR 1.0.5+timer.1 / ISGN 2.4.0+timer.1 / M2N 2.16.1+timer.1 / FEC 2.4.2+timer.1

## 生产证据门禁

在把 benchmark 结果用于落地决策前，先运行：

```bash
dor production-evidence-gate \
  --evidence release/PRODUCTION_EVIDENCE_INPUT.json \
  --output production_evidence_status.json
```

当前发布基线是 `ASSISTED_POC=SUPPORTED`、`ASSISTED_PRODUCTION=BLOCKED`、`SAFETY_CRITICAL_AUTONOMY=BLOCKED`。该门禁不会用单测或总体平均分替代跨工况、跨资产、真实漂移、部署负载或安全证据。详见 `PRODUCTION_EVIDENCE_CLOSURE_CN.md`。

Phase 2 在此基础上增加：

```bash
# 新 session/new asset 打开 test 前，policy 必须已冻结并 hash 绑定
dor prospective-evaluate --policy release/PROSPECTIVE_ACCEPTANCE_POLICY_NEXT.json \
  --benchmark prospective_benchmark.json --output prospective_result.json

# selected benchmark segments 只会给 PROXY_ONLY；连续运行才可能取得 production credit
dor deployment-ops-evaluate --spec deployment_ops.json \
  --predictions predictions.csv --output ops_result.json

# 外部官方数据先做本地 bytes/hash/结构 intake
dor source-intake-check --spec release/source_intake_specs/UCI_HYDRAULIC_447.json \
  --artifact condition+monitoring+of+hydraulic+systems.zip --output intake.json
```

当前 OPSSAT control intentionally 返回 `INELIGIBLE`，即使 16/16 冻结 metric checks 通过；这是因为历史 test 已知，不能重新解释为 prospective benchmark。详见 `PRODUCTION_EVIDENCE_CLOSURE_PHASE2_CN.md`。

## R5 Active Discovery 建议入口

当已有机理候选仍无法区分时，可由 ISGN 用仿真银行和实验候选生成结构后验、可辨识性与下一实验排序，再交给 DOR 做边界审查：

```bash
dor active-discovery-advisory \
  --input isgn_active_discovery.json \
  --output active_discovery_advisory.json
```

输入必须显式包含 `truth_quarantine`，其中 `truth_channels_allowed`、`held_out_truth_loaded`、`true_structure_loaded` 均精确为 `false`；`structure_posterior` 必须为有限、非负且总和为 1 的映射。若声称 `UNIQUE_SUPPORTED`，还必须有 `evidence_stability.status=STABLE` 和 `hypothesis_space_coverage_complete=true`。

该命令只输出 `ADVISORY_ONLY` 或 `BLOCKED`，永不允许实验执行、诊断、根因或机理认证。输入中的 `payload`、`command`、`actuator` 与安全证据内容不会被转发；真实设备实验必须由外部安全授权流程处理。

> 本版保留表格诊断、符号编译、UDE 和自动设计，新增 R4 时间观测门、递归时距资格和资产/舰队 Reference 治理。它只在显式候选、数据与预算合同内选择最合适模型，不保证任意设备的全局最优模型。

## 1. 安装

推荐使用 `scripts/install.sh` / `scripts/install.ps1` 从当前源码安装四个组件。`wheels/` 是本地构建产物，不作为 Git 源码交付的唯一安装路径。

验证：

```bash
dor stack-info
isgn doctor
m2n doctor
fec --help
```

### 模型族与时序设计

```bash
m2n model-plan --spec device_card.json --data sequence.csv --out model_plan.json
m2n temporal-design --spec device_card.json --data sequence.csv --out temporal_output
m2n predict-temporal --design-dir temporal_output --data heldout.csv --out predictions.csv
```

需要递归时距时先由 DOR 绑定：

```bash
dor forecast-bind --manifest dataset_manifest.json --device-card device_card.json --output bound_device_card.json
```

`deployment.forecast_evaluation` 可声明 `ONE_STEP/1` 或 `RECURSIVE_ROLLOUT/H`。M2N 不得自行猜测 H，也不得为闭环预测发明未来外生输入。

`DeviceCard` 必须显式声明 independent context、局部时间、数据 split 和 `modeling_contract`。工况支持标志只表示是否位于训练经验支持域，不是故障概率。

### 辨识与跨任务仲裁

```bash
dor identify-plan --ontology fault_ontology.json --localization localization.json --output identification_request.json
dor identify --ontology fault_ontology.json --localization localization.json --evidence evidence.json --output identification.json
dor multi-task-arbitrate --detection detection.json --localization localization.json --identification identification.json --output diagnosis.json
```

## 2. 原 v0.4 主链仍然可用

```bash
dor manifest-check --manifest dataset_manifest.json --data features.csv
dor reference-plan --manifest dataset_manifest.json --request diagnostic_request.json --data features.csv --out reference_plan
dor plan --manifest dataset_manifest.json --request diagnostic_request.json --data features.csv --out plan
dor run --manifest dataset_manifest.json --request diagnostic_request.json --data features.csv --out run
```

## 3. v0.5：建立 Fault Ontology

最小结构示例：

```json
{
  "contract": "DOR-FAULT-ONTOLOGY/0.5",
  "ontology_id": "example",
  "domain": "industrial-device",
  "components": [
    {"component_id":"pump","component_type":"hydraulic_pump","kind":"PHYSICAL"}
  ],
  "sensors": [],
  "commands": [],
  "topology_edges": [],
  "faults": [
    {
      "fault_id":"pump_leakage_weak",
      "target_component_id":"pump",
      "fault_mode":"internal_leakage_weak",
      "category":"PARAMETRIC",
      "exclusion_group":"pump_condition",
      "expected_evidence":[
        {"scope_type":"COMPONENT","scope_id":"pump","expected_state":"CLASS","expected_value":"1"}
      ]
    }
  ],
  "provenance":{"sources":["device manual / public dataset documentation / certified engineering source"]}
}
```

注意：Ontology 不是用来“给标签起名字”的装饰文件。没有来源的 component/fault/evidence 关系不能被当作物理事实。

## 4. 可选：Mechanism-to-Component Map

当 evidence 来自 ISGN/M2N mechanism slots 时，用：

```json
{
  "contract":"DOR-MECHANISM-COMPONENT-MAP/0.5",
  "map_id":"example-map",
  "bindings":[
    {
      "mechanism_id":"pump_flow_balance",
      "component_ids":["pump"],
      "sensor_ids":[],
      "provenance":"certified mechanism bundle / engineering equation"
    }
  ]
}
```

先检查：

```bash
dor ontology-check \
  --ontology fault_ontology.json \
  --mechanism-map mechanism_component_map.json
```

## 5. Evidence CSV

最小字段：

```csv
evidence_id,scope_type,scope_id,state,strength,validity
E1,COMPONENT,pump,ABNORMAL,0.92,VALID
```

已有分类链可以输出：

```csv
evidence_id,scope_type,scope_id,state,value,strength,validity
E2,COMPONENT,pump,CLASS,2,1.0,VALID
```

### 重要限制

只有：

```text
pump, ABNORMAL
```

并不足以证明：

```text
pump leakage severe
```

具体 fault mode 需要 ontology 中声明的 specific evidence signature。

## 6. 单 case localization

```bash
dor localize \
  --ontology fault_ontology.json \
  --mechanism-map mechanism_component_map.json \
  --evidence evidence.csv \
  --output localization.json
```

输出：

- `fault_hypotheses`
- `component_candidates`
- `supported_fault_set`
- `candidate_fault_sets`
- `matched_evidence`
- `contradicting_evidence`
- `UNKNOWN/ALTERNATIVES/ABSTAIN` 原因

## 7. Batch localization

Evidence 中增加 `case_id`：

```bash
dor localize-batch \
  --ontology fault_ontology.json \
  --evidence evidence.csv \
  --case-column case_id \
  --output localization_batch.json
```

v0.5 支持并发 fault。若两个 faults 不属于同一 `exclusion_group`，它们可以同时进入 `supported_fault_set`。

## 8. NASA ADAPT 示例

`examples/nasa_adapt_published_fixture/` 是**source-derived contract fixture**，不是完整 telemetry 数据集。

可运行：

```bash
dor ontology-check \
  --ontology examples/nasa_adapt_published_fixture/fault_ontology.json \
  --mechanism-map examples/nasa_adapt_published_fixture/mechanism_component_map.json

dor localize \
  --ontology examples/nasa_adapt_published_fixture/fault_ontology.json \
  --mechanism-map examples/nasa_adapt_published_fixture/mechanism_component_map.json \
  --evidence examples/nasa_adapt_published_fixture/evidence_relay_stuck_open.csv \
  --output relay_localization.json
```

它只验证公开来源可支撑的 Relay/sensor fault semantics；不要把它的结果解释为完整 ADAPT benchmark 成绩。

## 9. Hydraulic 真实数据 handoff 示例

`examples/hydraulic_localization_real/` 展示：

```text
FEC/DOR calibrated component class
→ DOR v0.5 ontology
→ multi-component localization
```

`truth_posthoc.csv` 与 `evidence.csv` 分离。reasoning 不读取 Test truth。

## 10. v0.5 不允许的输出

即使 localization 为 SUPPORTED，也不能写：

- “这就是 root cause”；
- “其他 component anomaly 一定是该 fault 的传播后果”；
- “fault probability = localizer score”。

v0.5 score 是 hypothesis-support score，不是校准概率。

## 11. 当前下一步

DOR 0.6 将开发 Propagation / Root-cause Reasoner v1。完整 NASA ADAPT topology/fault-injection telemetry 是优先验证材料；没有进入本地回归前只允许做合同级开发。

---

# 附录：v0.6 Root-Cause DEV PROTO 使用方式

> 本节只适用于开发验证；v0.6 尚未完成真实 ADAPT closure。

## 1. 先完成 v0.5 localization

```bash
dor localize \
  --ontology fault_ontology.json \
  --evidence evidence.csv \
  --output localization.json
```

## 2. 再执行 root-cause prototype

```bash
dor root-cause \
  --ontology fault_ontology.json \
  --localization localization.json \
  --evidence evidence.csv \
  --output root_cause.json
```

如果有 mechanism map：

```bash
dor root-cause \
  --ontology fault_ontology.json \
  --mechanism-map mechanism_component_map.json \
  --localization localization.json \
  --evidence evidence.csv \
  --output root_cause.json
```

## 3. 正确理解输出

- `SUPPORTED`：当前配置下存在唯一最小 root-cause set；
- `ALTERNATIVES`：多个最小 root sets 不可区分；
- `UNKNOWN`：有 localization，但证据不足以升级成 root cause；
- `root_cause_claimed=false` 时禁止把 fault localization 当根因结论。

默认 `coverage_required=1.0`。生产研究中不要为了降低 UNKNOWN 率而根据 Test truth 调低该参数。
