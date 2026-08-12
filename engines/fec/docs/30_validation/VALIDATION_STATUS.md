# FEC v2.3 Validation Status

## 1. 本发布包验证范围

最终回归：`71 passed`。

v2.1 新增验证覆盖：

- v2.0 全部既有回归；
- provider fingerprint deterministic/change-sensitive；
- provenance JSON roundtrip；
- lineage shared-ancestor audit；
- cyclic lineage metadata rejection signal；
- OBSERVED / DERIVED / SIMULATED / MISSING distinction；
- Reference Bundle provider/modality/source-domain compatibility；
- candidate sentinel compatibility gate；
- compatibility metadata persistence/restart；
- missing modality vacuous class evidence；
- mask-conditioned class conformal；
- unseen/sparse mask explicit unsupported；
- scalar anomaly `E` → class evidence hard stop。

Controlled Function Kill Test：`validation/validate_v21_assimilation.py`。

该 harness 不是工业故障检测 benchmark，只验证新增合同是否具有独立功能。

## 2. v2.1 Function Kill Test 结果

受控场景：

- commissioned runtime：`valid`；
- provider version change：`reference_unsupported`；
- modality `OBSERVED → DERIVED`：`reference_unsupported`；
- source domain `REAL → SIMULATION`：`reference_unsupported`；
- lineage audit：3 组 shared-upstream overlap，automatic equivalence = 0；
- mask-conditioned class coverage：full ≈ 0.902，partial ≈ 0.902，gap = 0；
- single marginal threshold：full = 1.000，partial ≈ 0.802，gap ≈ 0.198；
- scalar-E → Dirichlet shortcut：BLOCKED。

以上结果只说明：在该受控分布中，mask-conditioned calibration 修正了由结构性 missingness 造成的 group gap，并证明 FEC 的拒绝/lineage 合同生效；不等于真实 PHM 数据上的通用性能保证。

## 3. Frozen Core

SHA-256 仍为：

```text
b31a166a385a4287cdac17fc05888b52270a329ec2b6d72b1d1384ad25ad206e
```

## 4. 历史多域验证索引

外部数据不随工具包分发。

| 域/数据集 | 类型 | FEC 验证作用 | 当前边界 |
|---|---|---|---|
| UCI/ZeMA Hydraulic | 真实工业台架 | branch capacity、A、Coverage、弱 valve 边界 | trusted whole-system normal 太少 |
| Tennessee Eastman Process | 过程工业基准 | topology、chronological drift、localized support、conflict | 无可信外生 shift descriptor 时可 UNSUPPORTED |
| NASA C-MAPSS FD002/FD004 | 退化基准 | context-conditioned calibration、contamination、group audit | 不作为 universal RUL claim |
| EHSS | 电液伺服仿真 | excitation applicability、phase calibration、fault/model mismatch | 仿真不等于真实设备性能 |
| EUV target instance | 多子系统仿真 | Frozen Core 跨设备、local vs global | local branch 不等于 root cause |
| Uniwear | 真实刀具磨损 | 数据桥/production-role adequacy 负验证 | 不满足严格 admission |
| SKAB water loop | 真实水循环台 | run-disjoint admission、leaf drift、directional Contact | 部分 Contact power-limited；跨阀被 gate 阻断 |

## 4. v2.2 Application-Driven Validation — ZeMA Hydraulic Sensor Assurance

真实外部数据：**Condition monitoring of hydraulic systems (ZeMA/UCI)**。原始数据不随 Toolkit 分发。

本轮用途不是整机 Production Admission，而是 provenance / virtual sensor / missing modality 攻击。关键事实与当前运行证据：

- 严格“cooler=100, valve=100, pump leakage=0, accumulator=130, stable=0”仅 10 cycles，因此不宣称整机 Production Admission；
- 数据集 CP virtual sensor 可由 `FS2 × (TS3-TS4)` 的单系数物理代理在独立 held-out cycles 上重现：RMSE ≈ 0.0081 kW，corr ≈ 0.9993；
- 反演 `TS4 <- CP, TS3, FS2` 可达到 RMSE ≈ 0.030 °C，但 lineage 显示 CP 本身依赖 TS4，因此 TS4 缺失时该“高精度 failover”被正确 BLOCK；
- 不使用 CP/CE/TS4 作为运行输入的 learned TS4 provider 在 held-out cycles 上 RMSE ≈ 0.263 °C、corr ≈ 0.99885，并可在 TS4 缺失时解析为 `DERIVED`；
- sensor-consistency Reference：sentinel FPR ≈ 3.65%，held-out FPR ≈ 3.04%；
- paired counterfactual：+1 °C TS4 offset 检出 ≈ 99.3%，0→2 °C/cycle drift 检出 100%；弱 offset/drift 并未被人为调阈值强行闭环；
- TS4 missing 时，dataset CP/CE 与 circular `TS4-from-CP` 均级联 unavailable；显式 independent TS4 provider 与新的 CP provider 可恢复为 DERIVED；旧 observed Reference 返回 `reference_unsupported`。

结论：v2.1 的 provenance metadata 有价值，但缺少 lineage→runtime availability 的执行语义。该缺口促成 v2.2。


## 5. v2.3 Class-score Bridge Validation

新增 `tests/test_class_score_bridge_v23.py` 两项合同测试，当前总回归为 `71 passed`。

Cross-tool isolated-wheel smoke 使用独立 train/validation/calibration/test split：M2N v2.3 选择 `mechanism__logistic_regression`，FEC v2.3 在 150 calibration rows 上持久化 `FEC-CLASS-CONFORMAL-REFERENCE/1.0`，并对 150 test rows 输出 prediction sets。该 smoke 中 calibration-supported fraction=1.0，empirical test coverage=0.94，mean set size≈0.9467。该数值仅用于接口闭环，不是生产性能或额外理论保证。
