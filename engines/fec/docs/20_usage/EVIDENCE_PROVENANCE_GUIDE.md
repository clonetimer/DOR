# Evidence Provenance / Lineage 使用指南

## 1. 什么时候必须声明 provenance

真实设备 Adapter 中，以下来源至少应声明：

- raw sensor；
- virtual/soft sensor；
- simulator output；
- data augmentation output；
- neural/model-derived residual or feature。

## 2. Provider fingerprint

推荐按 provider 名称、版本、影响输出的配置和可选模型 artifact SHA-256 计算：

```python
from fec.extensions import ProviderIdentity

provider = ProviderIdentity.build(
    "virtual_pressure",
    "1.4",
    configuration={"inputs": ["flow", "spool"], "normalization": "rev3"},
    artifact_sha256="...64 hex...",
)
```

只要这些内容变化，fingerprint 就应变化。若 Reference Bundle 绑定旧 fingerprint，新 provider 必须重新 admission。

## 3. Lineage

`parents` 写直接上游 source id。`audit_evidence_lineage()` 会展开共享祖先并报告潜在重复信息源。

禁止：

```text
shared source → 自动 equivalence
correlation → 自动 equivalence
```

只有设备语义能够定义 equivalence。

## 4. PhysicsKnowledgeProfile

仅用于 learned/model-derived provider：

- observational bias：真实/仿真/增强/virtual sensing 数据来源；
- inductive bias：architecture 中的 topology/symmetry/conservation 等；
- learning bias：physics loss、monotonicity constraint 等；
- constraint type：hard / soft / none。

这些元数据不产生额外 `E`。
