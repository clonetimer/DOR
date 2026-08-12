# FEC v2.2 Architecture

## 1. 四层结构

### Device / Provider Layer

负责：原始传感器、virtual sensing、simulation、learned provider、Contact semantics、G/A、Context、excitation、D_cal claims，以及 v2.1 provenance / modality state 与 v2.2 runtime lineage availability。

### Frozen Evidence Layer

`fec/core/fec_core.py` 负责 empirical leaf Evidence、topology、equivalence collapse、local coordination、C_presence/C_effective/A propagation 和 topology signature。

### Production Control Plane

负责 conformal backend、Reference Bundle、U/D_cal、sentinel audit、promotion/rollback、durable state，以及 v2.1 Reference Compatibility Contract；v2.2 在 Reference 选择前先由 resolved lineage 生成 runtime compatibility。

### Optional Diagnostic Layer

class evidence / opinion fusion / mask-conditioned class conformal 等，不属于 scalar-E 稳定主链。

## 2. Stable path

```text
raw / virtual / simulated / learned source
  ↓
EvidenceProvenance + ModalityAvailability
  ↓ Device Adapter
Contact matrix + Context + G/A + lineage
  ↓ leaf reference
leaf Evidence E
  ↓ Frozen Core topology
root Evidence
  ↓ Reference compatibility gate
  ↓ root backend
p_root + support
  ↓ calibration/control-plane state
Policy Evidence State
```

Reference compatibility gate 必须先于不匹配 reference 的实际 Evidence 使用。

## 3. Provenance 与 topology 是两套不同语义

- topology：Contact 属于哪个物理局部、如何协调；
- lineage：Contact 的数据/模型从哪里来、是否共享上游源。

共享 lineage 只触发审计，不能自动推导 equivalence。

## 4. PhysicsKnowledgeProfile

observational / inductive / learning bias 用于描述 learned/model-derived provider 如何吸收物理知识。它们不是三份 runtime Evidence；尤其 physics loss 不应被重复计票。

## 5. Optional missing-modality class path

`MaskConditionalClassConformal` 是 Experimental：仅接受 genuine class evidence；缺失 modality 作为 vacuous source；mask 未校准时拒答。它不替代 `G/E/A` 和 root calibration。
