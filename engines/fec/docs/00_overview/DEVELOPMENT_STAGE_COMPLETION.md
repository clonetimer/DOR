# FEC 开发阶段完成与“应用驱动更新阶段”判定

## 1. 判定结论

从 **v2.0.0** 开始，可以把 FEC 的**通用工具包开发阶段视为基本完成**，并进入应用驱动更新阶段。

这里的“开发完成”限定为：FEC 作为 Evidence Substrate 的通用闭环功能已经具备；不表示所有设备 Contact、所有 drift、所有下游诊断任务都已经解决。

## 2. 开发阶段完成判据

一个新设备在不修改 Frozen Core 的前提下，应能完成：

| 基础能力 | v2.0 状态 | 证据 |
|---|---|---|
| Contact / G / A / topology 配置 | 完成 | Core + templates + demo config |
| finite-sample leaf Evidence | 完成 | Frozen Core |
| equivalence / local hierarchical coordination | 完成 | Frozen Core tests |
| root calibration | 完成 | empirical/conformal backends |
| Context/reference validity separation | 完成 | S_ctx / S_ref contracts |
| U / D_cal admission | 完成 | Production manager / bundle manager |
| independent sentinel validation | 完成 | v1.5/v1.6 manager + tests |
| anti-conservative block | 完成 | calibration audit + real negative cases |
| conservative / power-limited state | 完成 | calibration audit + bundle policy |
| leaf+root atomic refresh | 完成 | v1.6 Reference Bundle |
| promotion / rollback | 完成 | v1.5/v1.6 + tests |
| **process restart persistence** | **v2.0 完成** | JSON store roundtrip/rollback tests |
| **最新生产 CLI 与 Bundle 同步** | **v2.0 完成** | end-to-end CLI tests |
| Adapter / Contact Pack 模板 | 完成 | `templates/` |
| 不支持状态显式返回 | 完成 | unsupported/block contracts |
| 真实设备 run-disjoint admission | 已有证据 | SKAB REAL-R1 |

发布复验：`49 passed`。

## 3. 为什么说现在已经进入“应用过程中的更新阶段”

关键证明不是版本号，而是过去几轮功能变化的来源已经从“补最基本主链”转向“真实应用暴露边界后再修正”：

### 证据 A：TEP normal drift → v1.5 Production Calibration

早期静态 calibration 在 TEP chronological normal drift 下失效。尝试 time weighting/localized support 后并不能无条件恢复合法校准，因此 FEC 没有继续增加一个万能公式，而是建立 `UNVERIFIED/HEALTHY/BLOCKED`、sentinel、promotion/rollback 的控制面。

这是典型的应用场景暴露未考虑情况后更新工具包。

### 证据 B：SKAB REAL-R1 → v1.6 Reference Bundle

真实 SKAB 应用表明：root reference 已有生命周期，但 leaf reference 可能仍旧；leaf Evidence 一旦饱和，单独刷新 root 无法恢复排序信息。于是 v1.6 把 leaf+root 变成原子 Reference Bundle。

这个需求不是最初设计清单凭空添加，而是来自真实设备应用。

### 证据 C：v1.6 可运行但进程重启丢状态 → v2.0 Persistence

当 Reference Bundle 真正作为工具使用时，发现 active/retired 只在内存中，重启后不能恢复，也无法保证 rollback。这是部署使用暴露的基础工程缺口。v2.0 补齐 durable store，并验证 restart 等价。

### 证据 D：库升级到 Bundle，但入口仍是 v1.5 root-only → v2.0 CLI 同步

这是工具化应用过程中暴露的“库能力与用户入口不同步”问题。v2.0 统一生产入口后，主链才真正闭环。

## 4. 从 v2.0 开始，什么叫“更新”

以下属于更新阶段：

- 新设备暴露新的 Contact/Excitation 模式；
- 某类 Context 需要新的合法 calibration backend；
- 新的 Reference lifecycle 边界在真实运行中被证明；
- 新数据 provenance/角色隔离规则需要加强；
- 新的 downstream 需要结构化 Evidence 输出接口；
- 性能、兼容性、可操作性问题在实际部署中出现。

## 5. 什么仍然不属于“开发未完成”

以下不是 FEC 工具包必须内建的通用功能：

- 每一种设备的机理特征；
- universal root cause；
- RUL；
- novel-fault recognition；
- 所有 drift 的自动修复；
- 所有 prior-art backend 的持久化；
- 设备数据库、UI、云服务。

这些要么属于设备 Adapter，要么属于下游 PHM，要么属于后续可选 Extension。

## 6. 阶段门槛

v2.0 后若要修改 Frozen Core，至少需要：

1. 在多个独立设备/数据域出现同一种核心语义失败；
2. 证明无法通过 Adapter / Extension / Control Plane 修复；
3. 有反事实测试说明 Core 变更不会破坏既有合同；
4. 重新执行全部回归和真实设备基线。

## 7. v2.1 对“应用驱动更新阶段”的进一步证明

v2.1 没有重新展开通用 Core，而是从应用/文献对照中识别三个已经存在于实际部署语义中的缺口：

1. virtual/simulated/learned Contact 的来源没有结构化 lineage，容易被误当独立信息源；
2. provider 更新或 observed→derived modality 改变后，旧 Reference Bundle 没有硬兼容 gate；
3. genuine class-evidence 路径在 modality 缺失时需要按 mask 校准，不能继续使用单一 marginal threshold。

解决方式全部位于 Extension / Control Plane，并通过 Function Kill Test 证明新增合同产生实际不同的拒绝/校准行为；Frozen Core SHA 未变化。这符合“应用暴露通用边界→小范围更新工具包”的更新阶段定义。


## 8. v2.2：真实应用再次证明“开发基线完成 ≠ 停止更新”

ZeMA/UCI 液压试验台的 missing-TS4 攻击没有要求修改 Frozen Core，而是暴露出 v2.1 Control Plane 的执行缺口：lineage 与 modality state 已能记录，却没有把 upstream loss 自动传播到 downstream virtual/learned sources。

v2.2 因此只增加 `resolve_evidence_availability()`、explicit failover selection 与 resolved compatibility。该演化符合既定节奏：

```text
稳定开发基线
→ 真实应用
→ 可复现的通用缺口
→ 最小 Extension/Control-Plane 更新
→ 回归验证
```

这比继续预想所有未来场景更符合 FEC 的更新阶段定义。


## 9. v2.3：闭合 DOR/M2N 的 class-score 运行接口

在 fault-diagnostic 应用链中，FEC v2.1 已有 mask-conditioned class conformal 算法扩展，但缺少“外部 selected classifier score pack → durable calibration reference → prediction-set CSV”的直接运行入口。M2N v2.3 补上模型训练/选择后，这一缺口成为可复现的接口问题。

v2.3 因此仅新增 `class_score_bridge` 与 CLI，不修改 Frozen Core。它要求独立 calibration split、明确 class frame 和 modality mask，并对 unsupported mask abstain。这仍符合 application-driven update：**下游实际链路暴露接口缺口 → 最小 Extension/Control-Plane 更新**。
