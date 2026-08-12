# Context-conditioned Score Calibration — FEC v2.4

## 1. 为什么新增

真实设备验证暴露了两个不同但同构的问题：

1. **液压 Accumulator**：全局 class conformal 的总体经验 coverage 尚可，但某些 held-out operating group 明显下降；
2. **OPSSAT-AD**：PCA anomaly score 的不同 telemetry channel 具有不同参考分布，且有些 channel 的独立 normal calibration 数量不足。

因此 v2.4 不修改 Frozen Core，而是在 Extension/Control Plane 增加两个外部 score bridge。

## 2. scalar anomaly score

```bash
fec anomaly-calibrate \
  --calibration-scores calibration_scores.csv \
  --reference anomaly_ref.json \
  --context-columns channel \
  --alpha 0.10 \
  --min-group-size 20

fec anomaly-infer \
  --scores test_scores.csv \
  --reference anomaly_ref.json \
  --output anomaly_decisions.csv
```

要求：calibration score pack 必须来自独立 trusted-normal calibration split。每个 context 使用自己的经验上尾 p-value。未见或样本不足 context 输出 `CONTEXT_UNSUPPORTED`，不会自动借用最相近/最宽松 reference。

## 3. fault class score

```bash
fec class-context-calibrate \
  --calibration-scores calibration_scores.csv \
  --reference class_context_ref.json \
  --context-columns cooler,pump \
  --alpha 0.10 \
  --min-group-size 20

fec class-context-infer \
  --scores test_scores.csv \
  --reference class_context_ref.json \
  --output class_context_sets.csv
```

`score__<class>` 仍必须是共享 class frame 上的 genuine probability。若同时存在 `mask__<modality>`，实际 calibration key 是 `(declared context, modality mask)`。

## 4. 安全边界

- Context 必须由 Adapter / DOR 明确声明，FEC 不从 test labels 自动寻找“最好分组”。
- 未见 context 默认 fail closed。
- 更细 context 通常提高语义一致性，但会降低 reference coverage；这是明确的 power/support trade-off，不应通过偷偷 pooling 消除。
- empirical test FPR/coverage 是审计结果，不是新的理论保证。
- scalar anomaly p-value 不等于故障类别概率；class prediction set 不等于物理 root cause。
