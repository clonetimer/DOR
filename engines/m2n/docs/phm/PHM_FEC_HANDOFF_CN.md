# M2N PHM Evidence 与 FEC Handoff

M2N 的职责到以下量为止：

- mechanism residual；
- finite-model residual；
- model disagreement；
- normal support / OOD。

M2N 不直接输出 fault probability、fault class 或 RUL。

`m2n evidence` 使用冻结设计产生 sample-level evidence；`m2n fec-export` 将 residual/disagreement 做窗口 RMS，support/OOD 保留为 reference-compatibility context。

`m2n fec-calibrate` 要求显式 trusted healthy claim；`m2n fec-infer` 调用外部 FEC Core。FEC 输出不得反向改变已经冻结的 M2N architecture。

## 时间基准注意

`m2n evidence` 优先使用 `time_s`，其次 `time`。如果输入没有时间列，只能退化为样本序号。此时 `m2n fec-export --window-s` 的数值实际上对应“样本序号基准”，不能被解释成真实秒数。正式 PHM/FEC 使用应在原始数据中提供真实时间列/采样周期。


## v2.3 故障分类 score handoff

`m2n diagnostic-design` 在唯一分类模型被 `SELECTED/FIXED` 后，可从独立 `calibration` split 输出 `scores/calibration_scores.csv`：

- `score__<class>`：按冻结 `class_frame` 排列的 selected-model predictive probability；
- `true_label`：仅 calibration score pack 必需；
- `score_contract.json`：声明来源、特征视图、prohibited claims 与 FEC/DOR 消费语义。

这些 score **不是 scalar anomaly E，也不是 DOR 根因概率**。FEC v2.3 的 `class-calibrate` / `class-infer` 可把它们变成 mask-conditioned prediction set；DOR 再做 class-frame/ontology 对齐和最终仲裁。
