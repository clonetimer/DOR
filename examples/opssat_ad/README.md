# OPSSAT-AD context-balanced P1 example

这是当前集成包的**真实卫星异常检测**基线证据，不是 fault-class/root-cause 示例。

主链：

```text
18 official segment features
  -> role freeze before model selection
  -> M2N SELECTED raw__pca_reconstruction
  -> FEC exact-channel calibration (alpha=0.10, min_group_size=20)
  -> DOR NORMAL / ABNORMAL / ABSTAIN
```

关键结果见 `calibration_summary.json`：

- 7/9 channel calibration-supported；
- official test 523/529 segment supported；
- CADC0886 / CADC0890 因 official-train normal 总量只有 5 / 3，继续 fail closed；
- test 未参与模型选择或 calibration reference 拟合。

`opssat_p1_features.csv` 是从 OPSSAT-AD `dataset.csv` 派生的 18-feature role-frozen 表，不包含原始 `segments.csv` 的 30 万余 telemetry points。
