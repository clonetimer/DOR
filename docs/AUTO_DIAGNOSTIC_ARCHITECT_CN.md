# DOR Auto Diagnostic Architect 使用说明

## 1. 作用

该通道接收结构化设备机理卡、数据清单、实际数据和设计请求，自动完成数据资格审查、机理特征编译、检测/定位/分类/辨识阶段准入、有限候选训练比较和可运行模型包生成。

它不承诺全局 NAS 最优，也不会在缺少标签时生成伪定位或伪辨识结论。

## 2. 设计

```bash
dor auto-design \
  --device-card examples/auto_design_servo/device_card.json \
  --manifest examples/auto_design_servo/dataset_manifest.json \
  --data examples/auto_design_servo/data.csv \
  --request examples/auto_design_servo/auto_design_request.json \
  --out /tmp/dor-auto-servo
```

重复相同设计可加入 `--resume`。请求、设备卡、数据或 ontology 哈希变化时会拒绝复用。

## 3. 推理

```bash
dor auto-predict \
  --bundle /tmp/dor-auto-servo/pipeline_bundle \
  --data examples/auto_design_servo/data.csv \
  --output /tmp/dor-auto-servo/predictions.csv
```

若某阶段没有稳定唯一胜者，输出会保留等价候选的独立列，不把平均结果伪装为唯一诊断。

## 4. 数据形式

- `TABULAR`：在 `features.raw_columns` 显式列出数值特征。
- `TIMESERIES_LONG`：每行一个时间点，声明 group、time 和 channel 列；按 group 压缩为样本。
- `TIMESERIES_NPZ`：`X` 为 `[sample,time,channel]`，其他 split/label 数组通过 `metadata_keys` 映射。

所有训练特征必须有限；首版不隐式插补。run/asset 标识应进入 `leakage_guard.group_columns`。

## 5. 安全状态

- `QUALIFIED`：阶段监督和验证合同可运行。
- `PORTFOLIO_ONLY`：检测可训练，但没有故障标签证明某个 detector 最优。
- `ALTERNATIVES`：多个候选在 practical margin 内不可区分。
- `UNSUPPORTED`：标签、类别覆盖、数据质量或可辨识性不足。

详细机制、比较、实验和不可宣称内容见每次输出的 `audit_report.md`。
