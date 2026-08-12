# DOR v0.4 Context Hierarchy / Reference Applicability 开发说明

版本：0.4.0 DEV RC1  
开发目标：闭合 DOR 计划树中的 **D2 Evidence / Context Qualification**，不扩展故障定位、根因或健康管理。

## 1. 本轮核心问题

DOR v0.3 采用严格 exact-context 规则：

```text
exact context 有足量独立 calibration reference -> FEC 可运行
exact context 稀疏/未见 -> DOR ABSTAIN
```

该规则安全，但真实数据暴露两个问题：

1. Hydraulic Accumulator 中 240 个 test cycle 只有 80 个拥有 `(cooler,pump)` exact reference；
2. OPSSAT-AD 中 9 个 telemetry channel 有 2 个 channel 的 normal calibration reference 极少。

v0.4 要解决的不是“如何减少 ABSTAIN”，而是：

> **什么时候一个更粗的 context 有资格作为 reference ancestor；什么时候只能输出 POWER_LIMITED / CONTEXT_UNSUPPORTED。**

## 2. 新核心合同

新增：

- `DOR-DATASET-MANIFEST/0.4`
- `DOR-REFERENCE-APPLICABILITY/0.4`
- `DOR-DIAGNOSTIC-PLAN/0.4`
- `DOR-DIAGNOSIS-BUNDLE/0.4`

### 2.1 Context hierarchy

示例：

```json
{
  "context": {
    "policy": "HIERARCHICAL",
    "fallback_policy": "DECLARED_ANCESTOR_ONLY",
    "levels": [
      {
        "level_id": "exact",
        "columns": ["cooler", "pump"],
        "min_group_size": 20,
        "fallback_allowed": false,
        "sharing_basis": "exact declared nuisance operating condition"
      },
      {
        "level_id": "pump_regime",
        "columns": ["pump"],
        "min_group_size": 40,
        "fallback_allowed": true,
        "sharing_basis": "explicitly declared domain/policy hypothesis"
      }
    ]
  }
}
```

约束：

1. hierarchy 必须从细到粗，后一级 columns 必须是前一级的真子集；
2. DOR 不自动创建 global fallback；
3. coarser fallback 必须显式 `fallback_allowed=true`；
4. coarser fallback 必须提供非空 `sharing_basis`；
5. `policy=EXACT` 时禁止任何 fallback；
6. 当前唯一允许的 fallback policy 是 `DECLARED_ANCESTOR_ONLY`。

## 3. Reference applicability 状态

每个 test row 在 FEC 运行前由 DOR 给出：

| 状态 | 含义 |
|---|---|
| `EXACT_SUPPORTED` | finest context 的独立 calibration reference 达到该 level 的最小样本数 |
| `HIERARCHY_SUPPORTED` | exact 不足，但一个预先声明、允许共享且样本充足的 ancestor context 可用 |
| `POWER_LIMITED` | exact/允许 ancestor 在 calibration 中存在，但 reference 数量不足 |
| `CONTEXT_UNSUPPORTED` | 没有 exact reference，也没有任何预先声明的可用 ancestor |

这些状态只使用：

- DatasetManifest；
- Train/Validation/Calibration/Test 的角色分配；
- context values；
- calibration reference count。

**不读取 test label，也不读取 test score。**

## 4. DOR 与 FEC 的边界

DOR v0.4 不复制 FEC calibration 算法。

执行方式为：

```text
DOR 声明 hierarchy
       ↓
每个 declared level
       ↓
FEC 在该 level 上独立建立 exact reference
       ↓
FEC 对所有 test rows 产生该 level 的 calibration 输出
       ↓
DOR 根据 ReferenceApplicability
为每一行选择“最细的已准入 level”
       ↓
DOR arbitration
```

因此：

- FEC 仍只看到一个明确 context key；
- DOR 不修改 p-value / conformal threshold；
- 同一 calibration sample 可分别进入不同 level 的独立 FEC reference；
- DOR 只是选择预先批准的 reference level。

## 5. 安全边界

### 5.1 hierarchy admission 不是新的 coverage theorem

`HIERARCHY_SUPPORTED` 只说明：

> 当前 row 有一个被 Manifest 明确允许的 ancestor reference，而且该 ancestor 的 reference 数量满足门槛。

它**不证明** child context 与 ancestor 内其它 child contexts 可交换。

因此 coarser-reference 的 coverage 只能在对应 ancestor scope 下解释，不能自动宣称每个 child context 都满足同样 coverage。

### 5.2 禁止 Test 驱动 pooling

禁止：

```text
看 Test 哪些 context coverage 不好
→ 合并 context
→ 重新 calibration
→ 报告新的 Test 性能
```

这会把测试集变成 calibration-policy selection 数据。

v0.4 的 hierarchy 必须在 held-out Test audit 之前冻结。

## 6. Hydraulic Accumulator 真实验证

预先声明：

```text
exact      = (cooler,pump)
pump_regime = (pump)
```

其中 `pump_regime` 被明确标记为 **DEVELOPMENT_HYPOTHESIS**：仅用于验证 DOR hierarchy 运行能力，不作为“cooler 状态物理等价”的事实声明。

ReferenceApplicability：

- Test rows：240
- `EXACT_SUPPORTED`：80
- `HIERARCHY_SUPPORTED`：160
- `POWER_LIMITED`：0
- `CONTEXT_UNSUPPORTED`：0

在 v0.4 运行后：

- `SUPPORTED`：234 / 240
- `ABSTAIN`：6 / 240
- empirical set coverage（all test rows）：0.95833
- empirical set coverage（supported rows）：0.98291
- supported singleton accuracy：0.98291

但 post-hoc audit 同时发现：

- exact-supported rows empirical coverage：0.9875
- hierarchy-supported rows empirical coverage：0.94375
- `(cooler=100,pump=0)` child group empirical coverage：0.85

这个结果非常关键：

> hierarchy 显著降低了拒答，但 **coarse-group support 不能被偷换成 child-group coverage guarantee**。

因此该 hydraulic parent fallback **仍是开发假设，不应直接升级成生产默认策略**。

## 7. OPSSAT-AD 真实卫星验证

OPSSAT 当前不声明任何跨 telemetry channel 的 ancestor：

```text
policy = EXACT
level  = channel_exact
```

ReferenceApplicability：

- Test rows：529
- `EXACT_SUPPORTED`：523
- `POWER_LIMITED`：6
- `HIERARCHY_SUPPORTED`：0
- `CONTEXT_UNSUPPORTED`：0

DOR 输出：

- `SUPPORTED`：523
- `ABSTAIN`：6
- supported-row abnormal F1：0.68142
- supported-row balanced accuracy：0.80278

这 6 个 row 仍然拒答，证明 v0.4 **没有为了降低 ABSTAIN 而自动跨 channel pooling**。

## 8. 本轮结论

DOR 0.4 已经闭合：

```text
Context Hierarchy Schema
→ Reference Applicability
→ Sample-Power Gate
→ FEC per-level exact calibration
→ DOR per-row reference-level selection
→ POWER_LIMITED / CONTEXT_UNSUPPORTED fail-closed
```

但尚未闭合：

1. 如何证明或认证一个 ancestor sharing assumption；
2. 如何在不污染 calibration/test 的情况下做 reference-sharing validity audit；
3. 如何建立设备语义驱动的正式 context ontology。

这些问题不应继续塞进 v0.4。下一计划节点仍应是 **DOR 0.5 Fault Ontology + Mechanism-to-Component Map + Localization**；NASA ADAPT 在该节点加入。
