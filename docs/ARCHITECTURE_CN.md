# DOR Diagnostic Stack v0.5 架构

## 1. 四工具边界

```text
                 DOR Control + Diagnostic Semantics
        ┌────────────────────────────────────────────┐
        │ Dataset / Task / Context Qualification     │
        │ Fault Ontology / Component Map             │
        │ Evidence → Fault Hypothesis / Localization │
        │ Arbiter / Provenance / Abstention           │
        └───────────────┬────────────────────────────┘
                        │
       ┌────────────────┼────────────────┐
       ▼                ▼                ▼
     ISGN              M2N              FEC
 mechanism          model family      evidence
 discovery /        selection /       calibration /
 certification      evidence schema   lifecycle
```

DOR 不是第四种模型。它决定当前任务和 evidence **允许支持什么诊断语义**。

## 2. v0.5 在既有主链中的位置

```text
DatasetManifest 0.4
        ↓
TaskQualification / EngineAdmission
        ↓
ISGN / M2N
        ↓
FEC calibrated evidence
        ↓
ReferenceApplicability 0.4
        ↓
Fault Ontology 0.5
+ Mechanism-to-Component Map 0.5
        ↓
Localization 0.5
        ↓
SUPPORTED / ALTERNATIVES / UNKNOWN / ABSTAIN
```

## 3. 责任分离

### ISGN
回答 What mechanism?，产生 certified mechanism/parameter/residual hooks；没有资格时可被 DOR 正式 SKIP/BLOCKED。

### M2N
回答 What qualified model?，在 finite model portfolio 中训练/验证并输出 SELECTED/AMBIGUOUS/PORTFOLIO/UNSUPPORTED。

### FEC
回答 What evidence is valid/calibrated?，不决定“物理上哪个 component 是 root cause”。

### DOR v0.5
新增回答：

> “当前有效 evidence 与声明 ontology 相比，支持哪些 component/fault hypotheses？”

## 4. v0.5 与 v0.6 的硬边界

v0.5 只做 localization/hypothesis support；即使 topology 已进入 Fault Ontology，也**不沿 topology 推断因果传播**。

v0.6 才允许研究：

```text
local evidence
+ topology
+ temporal precedence
+ mechanism compatibility
+ contradictions
→ root cause vs downstream symptom
```

因此 v0.5 输出安全字段固定为：

```text
root_cause_claimed=false
propagation_inferred=false
```


## v0.6 Propagation / Root-Cause Prototype

v0.6 在 v0.5 localization 之后增加独立 root-cause 层。该层要求 timestamped fault-specific support、provenance-bound directional topology、temporal compatibility 与 evidence coverage；它不会把 topology upstream、earliest anomaly 或 localization top-1 直接解释为 root cause。当前为 DEV PROTO，真实 ADAPT closure 尚未完成。
