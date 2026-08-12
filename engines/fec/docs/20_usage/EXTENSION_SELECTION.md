# Extension Selection Guide

## 默认

如果同一 Context 下 normal reference 近似 exchangeable，优先使用：

- Core empirical leaf Evidence；
- Empirical root backend。

## Mondrian / conditional

适合离散且由外部声明的 Context group。每组必须有足够独立 normal reference。

## Localized conformal

适合连续 Context，但必须检查 local support / ESS。漂亮的 FPR 不能替代 support qualification。

## Weighted conformal

只在 covariate shift descriptor 和 density ratio/weight model 可审计时使用。故障响应变量不能为了降低误报而自动变成 shift descriptor。

## Reference cleaning

只在 reference contamination 有外部确认时使用。`U/D_cal` 仍是第一道预防机制。

## Opinion fusion / DBF / DS

只有 Adapter 输出真正非负 class evidence、共享相同 class frame、带 provenance 时启用。scalar anomaly `E` 不得硬转 opinion。

## Evidence provenance / reference compatibility（v2.1 Stable）

真实/virtual/simulated/learned Provider 均建议启用。它不改变 `E`，只负责 lineage、provider identity、modality state 和 Reference compatibility。provider/modality/domain 变化时应重新 admission，而不是选择“最正常”的旧 reference。


## Lineage availability resolution（v2.2 Stable）

只要 provenance graph 含 virtual / learned / simulated source，就应在 Reference 选择前解析 runtime availability。上游缺失、未解析 parent 或 lineage cycle 默认 fail closed；stale/cache value 不提供 eligibility。resolver 不负责自动选择 failover provider，也不证明 derived provider 的数值精度。

## Mask-conditioned class conformal（v2.1 Experimental）

仅用于 genuine class evidence 且 test-time modality 可缺失的分类场景。每个 mask 需要足量 labeled calibration；sparse/unseen mask 默认 abstain。它不是 scalar FEC Evidence 的默认融合器。


## Class-score conformal bridge（v2.3 Stable）

适用于外部/上游模型已经生成**共享 class frame 的概率 score**，且 FEC 拥有独立 labeled calibration split 的故障分类路径。默认要求：

- 每行 `score__<class>` 非负且和为 1；FEC 不静默归一化任意 logit；
- calibration score pack 带 `true_label`；
- 若存在 `mask__<modality>`，按 mask 分组校准；未见/稀疏 mask 明确 abstain；
- calibration split 不得兼任上游 model-family selection validation；
- output 是 prediction set + calibration support，不是 root cause。

该 bridge 可直接消费 M2N v2.3 的 `M2N-DIAGNOSTIC-SCORE-PACK/1.0`。
