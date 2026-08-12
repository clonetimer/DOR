# v2.4.0

- Added context-conditioned scalar anomaly-score calibration bridge.
- Added context-conditioned class-score conformal bridge.
- Added fail-closed support for sparse/unseen operating contexts.
- Frozen Core unchanged.

## 2.3.0 - Diagnostic class-score bridge

- Added persisted `FEC-CLASS-CONFORMAL-REFERENCE/1.0`.
- Added `fec class-calibrate` and `fec class-infer`.
- Directly consumes selected-classifier probability score packs such as M2N v2.3.
- Preserves independent calibration, mask support and abstention; Frozen Core unchanged.

# Changelog

## v2.2.0 — Application-Driven Lineage Availability Update

- 新增 `SourceAvailabilityState` / `EvidenceAvailabilityResolution`；
- 新增 `resolve_evidence_availability()`；
- 上游 observed modality 缺失会级联阻断依赖它的 virtual/learned source；
- unresolved parent 默认 fail closed；lineage cycle fail closed；
- replacement provider 必须显式选择；
- compatibility 可由 resolved lineage 生成并包含 transitive provider fingerprints；
- 新增 ZeMA/UCI 真实液压应用验证；
- Frozen Core 未修改。

## v2.1.0 — Evidence Provenance & Missing-Modality Update

应用驱动更新；Frozen Core 不变。

Stable Control Plane：

- 新增 `EvidenceProvenance`、`ProviderIdentity`、`PhysicsKnowledgeProfile`；
- 新增 `audit_evidence_lineage()`：发现 shared upstream source，但从不自动赋 equivalence；
- 新增 `ModalityAvailability`：OBSERVED / DERIVED / SIMULATED / MISSING；
- 新增 deterministic provider fingerprint；
- `ReferenceBundleMeta` 可绑定 `ReferenceCompatibilityContract`；
- provider version/fingerprint、modality signature、source domain 不匹配时返回 `reference_unsupported`；
- candidate sentinel 也必须满足相同 compatibility contract；
- compatibility metadata 进入 durable Reference Bundle store；
- CLI 支持 `reference_contract` 配置。

Experimental Diagnostic Adapter：

- 新增 `fuse_class_evidence_by_mask()`；
- absent modality 贡献 vacuous class evidence；
- 新增 `MaskConditionalClassConformal`；
- sparse/unseen mask 显式 unsupported；
- scalar FEC `E` → Dirichlet shortcut 继续硬阻断。

Validation：

- controlled Function Kill Test 验证 provider/mask/domain mismatch 拒绝、lineage 重叠审计、mask conditional vs single marginal calibration；
- 新增 v2.1 regression/contract tests。

## v2.0.0 — Development Baseline

- 将 FEC 整理为结构化工具包；
- durable Reference Bundle store；
- unified CLI；
- leaf+root production lifecycle 与 restart/rollback；
- 文档合并到最新规范。

## v1.6 — Reference Bundle

- SKAB REAL-R1 暴露 leaf/reference lifecycle 不一致；
- 引入原子 Leaf + Root Reference Bundle。

## v1.5 — Production Calibration Control Plane

- stage / validate / promote / rollback；sentinel audit；U/D_cal；support/ESS/health state。

## v1.3–v1.4 — Prior-Art Assimilation / Function Gap

- conditional / weighted / localized conformal；safe admission；reference cleaning；class evidence/opinion fusion。

## v1.0 — Frozen Scientific Core

- 冻结 `G / E / A / C_presence / topology / S_ctx / S_ref / U / D_cal` 的科学边界。
