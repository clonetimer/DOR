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
- `MULTIRATE_NPZ`：各传感器保留独立 `[sample,time]` 数组与采样率，不用伪插值制造统一采样率；UCI Hydraulic 官方 ZIP 可由 `dor uci-hydraulic-prepare` 转换。

请求中设置 `"representations": {"temporal_pyramid": true}` 后，固定统计/频带、顺序敏感 temporal-pyramid 和完整组合会作为不同视图进入同一验证预算。它仍不是 MiniRocket/InceptionTime，也不能恢复被窗口化丢失的精确事件语义。

所有训练特征必须有限；首版不隐式插补。run/asset 标识应进入 `leakage_guard.group_columns`。

## 5. 安全状态

- `QUALIFIED`：阶段监督和验证合同可运行。
- `PORTFOLIO_ONLY`：检测可训练，但没有故障标签证明某个 detector 最优。
- `ALTERNATIVES`：多个候选在 practical margin 内不可区分。
- `UNSUPPORTED`：标签、类别覆盖、数据质量或可辨识性不足。

定位可声明单一 `label_column`，也可声明多个二值 `target_columns`。后者使用 One-vs-Rest 并发部件模型，在独立 calibration split 上按部件设置经验假阳性阈值，推理输出 prediction set。辨识若存在 calibration split，会同时输出点估计和绝对误差分位上下界；没有独立 calibration 时不生成区间。

每次设计另生成 `end_to_end_test_audit.json` 和 `mechanism_stress_audit.json`。前者只在选模结束后读取 test 并应用实际检测门控；后者打乱有出处的机理特征，结果不反向参与本次选模。

详细机制、比较、实验和不可宣称内容见每次输出的 `audit_report.md`。
