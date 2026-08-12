# DOR 四工具合体故障诊断工具包 — v0.5 使用指南

版本：DOR 0.5.0 / ISGN 2.2.0 / M2N 2.3.1 / FEC 2.4.0

> v0.5 是开发候选。Detection / Classification 的运行主链继承 v0.4；v0.5 新增的是 Fault Semantics / Localization。Root Cause / Propagation / Identification 尚未声明完成。

## 1. 安装

推荐从 `wheels/` 安装四个 wheel，或使用 `scripts/install.sh` / `scripts/install.ps1`。

验证：

```bash
dor stack-info
isgn doctor
m2n doctor
fec --help
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
