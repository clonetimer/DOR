# FEC Scientific & Engineering Specification v2.2

## 1. 定位

FEC 是工业 PHM 的 Evidence 中间层：

```text
物理系统 / 传感器 / Virtual / Simulation / Learned Provider
→ Contact + Provenance / Modality Contract
→ Evidence + Validity State
→ Local/Hierarchical Coordination
→ Reference Compatibility + Production Calibration
→ PHM downstream
```

它回答“证据能否发言、相对正常有多异常、来自哪里、依赖谁、当前参考是否仍有资格解释”，而不自动回答最终根因、RUL 或维修策略。

## 2. Frozen semantic state

Frozen Core 继续保持：`G / E / A / C_presence / topology`。Control Plane 继续保持：`S_ctx / S_ref / U / D_cal`。

### G — Presence / Eligibility
Contact 是否存在且可计算。`cannot compute != zero evidence`。

### E — normal-relative Evidence
默认 Core 使用 finite-sample empirical p-value，并取 `E=-log(p)`。

### A — Applicability
Contact 存在，但当前方法假设、采样、任务/激励是否允许其可靠发言。

### C_presence
证据面实际存在比例；不能被 `C_effective` 替代。

### S_ctx / S_ref / U / D_cal
分别描述 Operating Context support、Reference compatibility、normal write permission、reference health dependency set。

## 3. v2.1 新增：Evidence provenance 不等于 Evidence vote

FEC v2.1 把 Evidence 来源作为 sidecar contract：

- `SourceKind`: observed / virtual / simulated / augmented / learned；
- `SourceDomain`: real / simulation / hybrid；
- asset / acquisition / physical object / modality；
- provider name/version/fingerprint；
- parent source ids / lineage；
- optional `PhysicsKnowledgeProfile`。

`PhysicsKnowledgeProfile` 记录 provider 中物理知识如何进入模型：

- observational bias；
- inductive bias；
- learning bias；
- hard / soft / none constraint。

这些字段说明 Contact Provider 的来源和假设，**不额外产生 anomaly Evidence，也不能作为额外投票**。

## 4. v2.1 新增：Modality Availability

FEC 区分：

```text
OBSERVED
DERIVED
SIMULATED
MISSING
```

因此“Contact 仍可计算”不自动意味着原始物理 modality 仍然 observed。例如 raw pressure 缺失但 virtual pressure 可生成时，可以有 `G=1`，同时 modality state 为 `DERIVED`。

## 5. v2.1 新增：Reference Compatibility Contract

Commissioned Reference Bundle 可绑定：

```text
modality_signature
+ provider_fingerprints
+ source_domain
```

只要已声明维度发生变化：

```text
old provider → new provider
OBSERVED → DERIVED
REAL → SIMULATION
```

旧 Reference 必须返回 `reference_unsupported`，不能先计算 Evidence 再选择“最正常”的 reference。

## 6. Evidence Lineage

共享上游数据源会被 `audit_evidence_lineage()` 标记。例如：

```text
raw pressure ─→ virtual flow
      └──────→ learned residual
```

这提示三者并非独立信息源，但 **shared lineage != semantic equivalence**。Lineage Audit 永不自动修改 FEC equivalence id。

## 7. Missing-modality class evidence（Experimental）

当且仅当 Adapter/下游 Provider 提供 genuine non-negative class evidence 时，v2.1 提供实验路径：

```text
per-modality ClassEvidence
→ missing modality = vacuous evidence
→ optional Dempster-style fusion
→ modality-mask Mondrian conformal calibration
→ class prediction set
```

约束：

- scalar FEC anomaly `E` 禁止转换成 Dirichlet class evidence；
- calibration 必须按实际 modality mask 分层；
- sparse / unseen mask 返回 unsupported，不自动借用邻近 mask；
- 只保证/评估 mask group 条件，不宣称 instance-conditional validity；
- 该能力在 v2.1 仍为 Experimental Diagnostic Adapter。

## 8. Stable production lifecycle

```text
trusted X_leaf + trusted X_root
+ D_cal
+ modality/provider reference contract
→ build Reference Bundle
→ raw disjoint trusted sentinel under same contract
→ health/support audit
→ promote
→ durable persist
→ restore after restart
→ compatibility check before Evidence use
→ evaluate / audit / refresh / rollback
```

## 9. 不允许的状态坍缩

不得把以下状态压成一个 confidence：

```text
G, E, A, C_presence, S_ctx, S_ref, U, D_cal,
modality status, provenance/lineage, provider compatibility
```

## 10. Core 变更原则

Frozen Core SHA-256 仍必须为：

```text
b31a166a385a4287cdac17fc05888b52270a329ec2b6d72b1d1384ad25ad206e
```

v2.1 全部新增能力位于 Adapter / Extension / Control Plane。

## 10. v2.2：Lineage Availability Resolution

### 10.1 应用发现的问题

v2.1 已有 provenance graph 与 modality state，但二者之间没有强制求值关系。真实 ZeMA 液压数据包含由物理传感器计算的 virtual sensors；当上游物理传感器缺失时，virtual signal 的历史/缓存数值仍可能留在数据流或文件中。仅靠“值是否存在”会错误保持 `G=1`。

### 10.2 正式规则

`resolve_evidence_availability()` 在 Contact/Reference 选择之前执行：

```text
OBSERVED source
    requires runtime modality = OBSERVED

DERIVED/SIMULATED source
    requires every declared parent available

unresolved parent
    -> MISSING unless externally declared available

lineage cycle
    -> fail closed
```

因此：

```text
cannot regenerate from valid upstream sources
!=
stale value exists in cache/file
```

后者不得使 source 获得运行资格。

### 10.3 Explicit provider selection

Resolver 不自动选择替代 provider。Adapter/Policy 必须显式声明：

```text
modality -> selected source_id
```

这保持了 FEC 已有原则：不能通过“哪个 reference / provider 让当前样本更 normal”来做路由。

### 10.4 Reference Compatibility

由 resolver 构造 compatibility 时，必须绑定：

- effective modality signature；
- selected provider fingerprint；
- selected source 的 transitive upstream provider fingerprints；
- source domain。

Observed→Derived、任意上游 provider 更新或 source-domain 变化均可使旧 Reference 变为 `REFERENCE_UNSUPPORTED`。

### 10.5 不属于本能力

v2.2 不提供自动 imputation、自动 provider quality estimation、自动 failover、自动 dependency discovery 或万能 missing-modality calibration。
