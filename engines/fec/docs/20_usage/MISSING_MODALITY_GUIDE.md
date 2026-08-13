# Missing Modality Guide v2.2

## 0. v2.2 前置：先解析 lineage availability

`ModalityAvailability` 不应再与 provenance graph 人工分离维护。若存在 virtual/learned sources，先调用 `resolve_evidence_availability()`；上游缺失造成的 downstream source BLOCK 必须先于 Reference 选择。详见 `LINEAGE_AVAILABILITY_GUIDE.md`。

## 1. Scalar FEC 主链

缺失原始 modality 时先声明 `ModalityAvailability`。若 virtual/simulated provider 能生成 Contact，可出现：

```text
raw modality = MISSING
replacement modality = DERIVED / SIMULATED
Contact G = 1
```

但这不能沿用只针对 OBSERVED modality commissioning 的 Reference Bundle；compatibility gate 应返回 `reference_unsupported`。

## 2. Class-evidence 实验路径

仅当有 genuine non-negative class evidence：

```python
fused = fuse_class_evidence_by_mask(batches, masks)
cal = MaskConditionalClassConformal(alpha=0.1, min_group_size=100)
cal.fit(fused.probabilities, labels, masks, vacuity=fused.combined_vacuity)
sets = cal.prediction_sets(test_prob, test_masks, vacuity=test_vacuity)
```

缺失 source 被当作 vacuous evidence；其 stale array value 被忽略。

### Held-out mask audit

完成独立 calibration 后，可在另一个带 `true_label` 的 held-out score pack 上逐 presence mask 审计覆盖与集合效率：

```bash
fec mask-calibration-audit \
  --scores heldout_scores_with_true_label.csv \
  --reference class_reference.json \
  --output mask_calibration_audit.json \
  --min-test-group-size 20
```

报告同时给出 empirical coverage、coverage shortfall、mean/singleton/empty set size、top-class accuracy 和 unsupported rows。它是 operational audit，不创建 instance-conditional 保证；每个 mask 内仍要求 exchangeability。

## 3. 资格条件

- 每个 mask 必须有足够 labeled calibration；
- calibration/test 在该 mask 内仍需 exchangeable；
- mask group 稀疏时返回 unsupported；
- 不允许为了补样本自动把物理含义不同的 missing-pattern 合并；
- 不宣称该模块解决 acquisition drift；必要时另行验证 weighted/localized 方法。
