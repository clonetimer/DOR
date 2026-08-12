# FEC v2.3 Class Score Bridge

## 1. 目的

把上游 selected diagnostic classifier 的类别概率，转换成**独立校准的 prediction set**，同时保留 modality-mask support/refusal。它位于 FEC Extension/Control Plane，不进入 Frozen Core。

## 2. 输入

Calibration CSV：

- `score__<class>`：至少两列，非负、每行和为 1；
- `true_label`：独立 calibration label；
- 可选 `mask__<modality>`：1=present，0=absent。没有 mask 列时按单一完整 source 处理。

M2N v2.3 唯一 `SELECTED/FIXED` classifier 可直接产生该格式。

## 3. 校准

```bash
fec class-calibrate \
  --calibration-scores m2n_design/scores/calibration_scores.csv \
  --score-contract m2n_design/score_contract.json \
  --reference fec_class_reference.json \
  --alpha 0.10 \
  --min-group-size 20
```

内部调用现有 `MaskConditionalClassConformal`。每个支持 mask 持久化：`n_reference / threshold / alpha`。若没有任何 mask 达到 `min_group_size`，拒绝生成空 reference。

## 4. 推理

```bash
fec class-infer \
  --scores test_scores.csv \
  --reference fec_class_reference.json \
  --output prediction_sets.csv
```

输出包含：

- `top_class / top_score`：仅作为模型原始摘要；
- `prediction_set / prediction_set_size`；
- `calibration_supported`；
- `calibration_reason`；
- `prediction_set_status = CLASS_SET_SUPPORTED | ABSTAIN`。

DOR 才负责把 prediction set 对齐 Fault Ontology，并与 localization/identification/root-cause evidence 仲裁。

## 5. 禁止捷径

- scalar anomaly `E` 不能改名成 class probability；
- 同一 validation split 不能同时做上游模型选择和 FEC calibration；
- unseen mask 不借用“最接近/最正常”的其他 mask threshold；
- top-1 score 不等于最终诊断 confidence；
- prediction set 不等于物理根因。
