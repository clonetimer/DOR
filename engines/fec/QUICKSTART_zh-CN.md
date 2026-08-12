# FEC Toolkit v2.2 使用指南

## 1. 安装与自检

建议 Python 3.10+。

```bash
python -m pip install -e .
python -m pip install -r requirements-dev.txt
# 离线环境已有 setuptools 时：python -m pip install --no-build-isolation -e .
python -m pytest -q
python validation/validate_v21_assimilation.py
python validation/validate_v22_lineage_availability.py
```

发布基线应通过全部测试、v2.1 Function Kill Test 与 v2.2 lineage-availability validation。

## 2. 最小非生产运行

```bash
python -m fec.cli run \
  --config configs/demo_machine.json \
  --normal-leaf examples/data/normal_leaf.csv \
  --normal-root examples/data/normal_root.csv \
  --input examples/data/inference.csv \
  --output fec_output.csv
```

## 3. v2.1 Reference Contract

配置可声明：

```json
"reference_contract": {
  "modalities": {
    "temperature": "observed",
    "vibration": "observed"
  },
  "provider_fingerprints": ["sha256:..."],
  "source_domain": "real"
}
```

只要 runtime 的已声明 modality/provider/domain 不一致，Production 结果必须为 `reference_unsupported`，不会借用旧 reference。

provider fingerprint 可由：

```python
from fec.extensions import ProviderIdentity
p = ProviderIdentity.build(
    "my_contact_pack", "1.0",
    configuration={"feature_schema": "rev2"},
)
print(p.fingerprint)
```

## 4. Production Commissioning

leaf/root/sentinel 必须分配不同物理 `data_id`；trusted claims 覆盖 `D_cal`；sentinel 同时必须符合 Reference Contract。

```bash
python -m fec.cli production \
  --config configs/demo_machine.json \
  --state runtime/reference_state.json \
  --key default \
  --input examples/data/inference.csv \
  --output runtime/output.csv \
  --trusted-claims demo_machine:healthy \
  --commission-leaf examples/data/normal_leaf.csv \
  --commission-root examples/data/normal_root.csv \
  --commission-sentinel examples/data/trusted_sentinel.csv \
  --bundle-id demo-bundle-v1 \
  --leaf-data-id leaf-run-001 \
  --root-data-id root-run-002 \
  --sentinel-data-id sentinel-run-003
```

## 5. 重启恢复

```bash
python -m fec.cli production \
  --config configs/demo_machine.json \
  --state runtime/reference_state.json \
  --key default \
  --input examples/data/inference.csv \
  --output runtime/output_after_restart.csv
```

compatibility metadata 与 leaf/root reference 一起持久化。

## 6. Evidence Provenance

真实 Adapter 应为 observed / virtual / simulated / learned source 提供：

- source id；
- modality / physical object / asset / acquisition id；
- provider identity/fingerprint；
- parents；
- source domain；
- optional PhysicsKnowledgeProfile。

详见 `docs/20_usage/EVIDENCE_PROVENANCE_GUIDE.md`。

## 7. Missing Modality

scalar FEC 主链首先使用 `ModalityAvailability` 和 Reference Compatibility。只有 genuine class evidence 才使用 experimental `MaskConditionalClassConformal`。详见 `docs/20_usage/MISSING_MODALITY_GUIDE.md`。

## 8. 接入新设备

1. 明确设备/物理对象、可观测量、Context、任务与 excitation；
2. 建 Device Adapter / Contact Pack；
3. 声明每个 source 的 provenance 和 modality state；
4. 审计 lineage，防止 derived source 被误当独立信息源；
5. 声明 G/A/topology/equivalence；
6. 建 run-disjoint leaf/root/sentinel/held-out；
7. 声明 D_cal + Reference Compatibility Contract；
8. 先审计 normal validity，再看 fault/degradation；
9. 不满足资格时返回 BLOCK/UNSUPPORTED。

## 9. v2.2 Lineage Availability

当物理传感器缺失时，不要仅手工修改 modality mask；先解析 provenance DAG：

```python
from fec.extensions import resolve_evidence_availability

resolution = resolve_evidence_availability(
    provenance_graph,
    {"TS3": "observed", "TS4": "missing", "FS2": "observed"},
)

# 若 CP 依赖 TS4，它必须被级联阻断。
assert not resolution.state("virtual:CP").available
```

若存在合法替代 provider，显式选择后再生成新的 runtime compatibility contract。完整说明见 `docs/20_usage/LINEAGE_AVAILABILITY_GUIDE.md`。
