# OPSSAT-AD 模型状态说明

## 一句话

**已经有了一个经过 M2N 重新选择、FEC 按 channel 独立校准、DOR 可拒答的卫星异常检测模型链；还没有完成具体故障类型、故障部件和物理根因诊断。**

## 当前模型

```text
18 official segment features
    -> M2N: PCA reconstruction
    -> anomaly_score
    -> FEC: exact-channel trusted-normal empirical p-value
    -> DOR: NORMAL / ABNORMAL / ABSTAIN
```

## Context-balanced P1

为继续闭合“跨遥测通道的可信校准”，本轮没有降低 `min_group_size`，也没有用 pooled/nearest-channel reference。相反，在**任何模型选择之前**重新冻结数据角色：只要某 channel 的 official-train normal 总量允许，就预留至少 20 个独立 calibration segment，然后从头重新运行 M2N。

结果：

- selected：`raw__pca_reconstruction`；
- validation ROC-AUC：`0.880087`；
- 对 runner-up 的 minimum practical margin：`0.104205`；
- FEC exact-channel support：`523/529 = 98.87%` official-test segment；
- DOR abstain：`6/529 = 1.13%`；
- supported-row F1：`0.6814`；
- supported-row balanced accuracy：`0.8028`；
- supported test-normal weighted empirical FPR：`0.09443`（alpha=0.10）。

仍不支持的两个 channel：

- `CADC0886`：整个 official-train pool 只有 5 个 normal segment；
- `CADC0890`：整个 official-train pool 只有 3 个 normal segment。

因此它们不是“切分得不好”而是当前公开数据本身不足以满足 `min_group_size=20`。

## 为什么仍称为 P0/P1 anomaly detector

它已经满足：

- 模型不是拍脑袋指定，而是有限 family validation selection；
- Train / Validation / Calibration / Test 角色在模型选择前冻结；
- calibration/test 不参与模型选择；
- calibration 按 telemetry channel 做资格审查；
- sparse channel 不借 pooled reference；
- DOR 会拒答。

但尚不满足：

- fault taxonomy；
- component localization；
- propagation/root cause；
- raw time-series representation competition；
- 对 `CADC0886/CADC0890` 的足量 independent normal reference。

因此，**“OPSSAT-AD 的卫星异常检测模型链已经设计好并可运行”是成立的；“卫星完整故障诊断模型已经设计完成”不成立。**


## DOR v0.4 回归状态

DOR 0.4 引入 Context Hierarchy / Reference Applicability 后，OPSSAT-AD **继续保持 strict exact-channel policy**：523/529 official-test rows 为 `EXACT_SUPPORTED`，6/529 为 `POWER_LIMITED`，`HIERARCHY_SUPPORTED=0`。这说明 v0.4 没有为了减少拒答而自动建立跨 telemetry channel 的共享 reference。
