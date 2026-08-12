# 新设备接入与 Production Admission 指南 v2.2

## Step 1 — 设备边界

列出物理部件、可观测量、任务/激励、外生 Operating Context、可由谁证明健康，以及每类数据属于 REAL / SIMULATION / HYBRID 哪个 domain。

## Step 2 — Source / Modality Inventory

对每个 raw/virtual/simulated/learned source 声明：

- source id；
- modality；
- physical object；
- asset / acquisition id；
- provider name/version/fingerprint；
- parents / lineage；
- OBSERVED / DERIVED / SIMULATED / MISSING；
- optional PhysicsKnowledgeProfile。

先运行 lineage audit。共享来源只触发 review，不自动 equivalence。

## Step 3 — Contact Pack

每个 Contact 至少声明 key、物理语义、topology path、G、A、方法假设、期望方向/响应、calibration normal 条件。fault label 不得进入 Contact。

## Step 4 — Context Contract

Context 优先来自外部运行条件/任务 setpoint；动态过程可用 phase。不要从 fault-responsive variable 反选 Context。

## Step 5 — Reference Compatibility Contract

对 Production Reference 声明其真正依赖的：

```text
modality signature
provider fingerprint set
source domain
```

provider/feature pipeline 更新后重新计算 fingerprint。OBSERVED→DERIVED、REAL→SIMULATION 等变化不能偷用旧 Reference。

## Step 6 — Reference 角色

至少准备：leaf normal、root normal、trusted sentinel、held-out normal、fault/degradation evaluation。优先物理 run 独立；同轨迹切窗只能 exploratory。

## Step 7 — D_cal

声明 reference 依赖的 health claims。只有外部可信 provenance 覆盖 dependency set 才允许 stage。

## Step 8 — 先做 normal validity

```text
sentinel validity
→ held-out normal
→ fault/degradation
```

不要先看 fault 结果再反选 split/window/alpha。

## Step 9 — Missing modality

scalar FEC 主链先做 ModalityAvailability + Reference Compatibility：

- raw missing + virtual replacement 可以 `G=1`；
- 但 modality state 必须是 DERIVED；
- 如果 Reference 只 commission 在 OBSERVED，返回 `REFERENCE_UNSUPPORTED`。

只有 genuine class evidence 才考虑 experimental mask-conditioned class conformal。

## Step 10 — 弱故障

按 Contact mechanism → excitation → directionality → sensor observability 检查；仍无 Evidence 时标记 not identifiable，不放宽 FPR 门槛。

## Step 11 — 生产状态

promotion 后立即持久化 state；restart 时先做 digest + compatibility 检查，再允许 Evidence 进入 Policy。


## v2.2：Reference 选择前先解析 runtime lineage availability

只要 Adapter 包含 virtual / learned / simulated source，运行时不要单独手写 `ModalityAvailability` 后直接选 Reference。建议顺序：

```text
physical modality state
        ↓
resolve_evidence_availability(provenance graph)
        ↓
available / blocked sources
        ↓
显式选择 observed 或 failover provider
        ↓
resolved ReferenceCompatibilityContract
        ↓
Reference Bundle compatibility gate
```

原则：

- 上游 source 缺失必须向所有声明依赖它的 derived source 级联；
- stale/cache/file 中仍有数值不能恢复 source eligibility；
- unresolved external parent 默认 fail closed；
- failover provider 必须由 Adapter/调用方显式选择，FEC 不按 p-value 自动挑选；
- observed→derived、provider version、transitive upstream provider 或 source-domain 改变时，旧 Reference 默认不兼容。

参见 `LINEAGE_AVAILABILITY_GUIDE.md`。
