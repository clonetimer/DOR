# FEC v2.4.0 Release Notes

## 定位

Application-driven calibration update。Frozen Core 未修改。

## 新增

- `anomaly-calibrate` / `anomaly-infer`：对上游 selected detector 的 scalar anomaly score 按外部声明 context 建立 trusted-normal reference；稀疏/未见 context fail closed。
- `class-context-calibrate` / `class-context-infer`：在 v2.3 class-score bridge 之上增加 operating-context key，并可与 modality mask 组成复合 calibration stratum。
- 新 contract：`FEC-ANOMALY-CONTEXT-REFERENCE/1.0`。
- 新 contract：`FEC-CONTEXT-CLASS-CONFORMAL-REFERENCE/1.0`。

## 触发证据

- Hydraulic Accumulator：全局 class calibration 在某 held-out nuisance group 上经验 coverage 下降到 0.85。
- OPSSAT-AD：不同 telemetry channel 的 calibration reference 数量和 score distribution 显著不同；exact channel calibration 对稀疏 channel 必须拒答，而不能无条件 pooled fallback。

## 不包含

- 不自动发现 context；
- 不自动 pooling / nearest-context fallback；
- 不声称 conditional coverage 在任意 non-exchangeable 环境下自动成立；
- 不修改 FEC Frozen Core。
