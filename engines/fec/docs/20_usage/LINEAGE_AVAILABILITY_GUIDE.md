# Lineage Availability Guide v2.2

## 1. 为什么 v2.1 仍然不够

v2.1 已经能记录：

- observed / virtual / learned / simulated provenance；
- parents / shared lineage；
- modality state；
- provider fingerprint；
- Reference Compatibility Contract。

但 provenance graph 与 `ModalityAvailability` 仍需 Adapter 人工保持一致。实际 ZeMA 液压数据暴露出一个危险配置：

```text
TS4 = MISSING
CP  = DERIVED
```

在数据文件里 `CP.txt` 仍然存在，但 ZeMA 的 CP virtual sensor 依赖 TS3、TS4、FS2。若真实 TS4 传感器失效，历史/缓存中的 CP 数值不能继续被当成当前可生成 Evidence。

## 2. v2.2 resolver

```python
from fec.extensions import resolve_evidence_availability

resolution = resolve_evidence_availability(
    provenance_graph,
    {
        "TS3": "observed",
        "TS4": "missing",
        "FS2": "observed",
    },
    available_external_sources={"external:ambient_temperature"},
)
```

规则：

1. `OBSERVED` source 必须有对应 `OBSERVED` modality；
2. virtual/learned/augmented/simulated source 的**全部声明 parent**必须可用；
3. 未解析 parent 默认 fail closed，除非显式列入 `available_external_sources`；
4. lineage 有 cycle 时整体 fail closed；
5. stale/precomputed file value 不覆盖 lineage availability；
6. resolver **不自动选择 failover provider**。

## 3. 显式 failover

如果 `sensor:TS4` 消失，而存在：

```text
virtual:TS4:model
parents = TS1, TS2, TS3, FS2
```

且这些 parent 当前可用，则该 source 可解析为：

```text
TS4 = DERIVED
```

调用者仍必须显式选择它：

```python
availability = resolution.materialize_modality_availability({
    "TS4": "virtual:TS4:model",
    "TS3": "sensor:TS3",
    "FS2": "sensor:FS2",
})
```

FEC 不会因为哪个 provider 让 p-value 最大，就自动切换到哪个 provider。

## 4. 与 Reference Compatibility 的连接

```python
runtime_contract = resolution.reference_compatibility(
    selected_sources,
    source_domain="real",
)
```

该方法把：

- effective modality signature；
- selected provider；
- selected source 的 transitive upstream provider fingerprints；

一起绑定到 Reference Compatibility。

因此 observed→derived failover、上游 provider 更新或 source-domain 变化，都要求重新匹配/准入 reference。

## 5. 不是 imputation algorithm

v2.2 不负责：

- 自动学习 virtual sensor；
- 自动判断某个 provider 的预测质量；
- 自动选择替代传感器；
- 用 stale derived value 续命；
- 自动合并 missing-mask calibration group。

virtual/soft sensor 的准确性、适用域和 D_cal 仍由 Device Adapter 与独立验证负责。
