# Reference Lifecycle v2.2

## 1. Commissioning

```text
trusted leaf normal run(s)
+
trusted root normal run(s)
+
D_cal claims
+
ReferenceCompatibilityContract
→ stage bundle
```

leaf/root `data_id` 不得重叠。

## 2. Compatibility Contract

可声明：

- modality signature：OBSERVED / DERIVED / SIMULATED / MISSING；
- provider fingerprints；
- source domain：REAL / SIMULATION / HYBRID。

这些是硬兼容条件。已声明条件不匹配时：

```text
REFERENCE_UNSUPPORTED
```

不允许借用“看起来更正常”的旧 reference。

## 3. Shadow validation

raw trusted sentinel 必须与 leaf/root 数据独立；若 Bundle 有 compatibility contract，sentinel 也必须声明相同 contract，否则 candidate 不得 promotion。

## 4. Promotion states

- `healthy`：可 promotion；
- `conservative`：validity-safe 但可能 power-limited；
- `anti_conservative`：BLOCK；
- support 不足或 reference compatibility 不满足：UNSUPPORTED/BLOCK。

## 5. Durable persistence

active/retired bundle 使用 canonical JSON + SHA-256 + atomic replace。v2.1 additionally persists the optional Reference Compatibility Contract。shadow/rejected candidate 仍不持久化。

## 6. Rollback

rollback 同时恢复 leaf + root reference 及其 compatibility metadata。
