
> **v2.4 application update:** adds exact context-conditioned bridges for external scalar anomaly scores and class probabilities. Sparse/unseen contexts fail closed; Frozen Core remains unchanged. See `docs/20_usage/CONTEXT_SCORE_CALIBRATION_v2.4.md`.
# FEC Toolkit v2.4

**FEC — Fractal Evidence Clamping / 分形证据夹持**

FEC 是面向工业设备故障诊断与健康管理（PHM）的 **Evidence Substrate / Evidence Lifecycle Protocol**。它不替代故障分类器、RUL 模型或因果诊断器，而是把设备机理、异质观测、运行上下文、Evidence 来源和正常参考组织成可校准、可局部协调、可审计、可拒答的结构化 Evidence。

## 当前阶段

v2.0 是通用开发阶段完成基线；v2.1 增加 Evidence provenance / provider / missing-modality reference 合同；v2.2 是第一次由真实设备应用直接触发的通用更新，新增 fail-closed lineage availability resolution。**v2.3 继续保持 Frozen Core 不变，只在 Extension/Control Plane 闭合 selected diagnostic class scores → independent calibration → prediction set/ABSTAIN 的运行接口。**

## Stable Production Profile

```text
Device / Mechanism / Context
        ↓
Device Adapter + Contact Pack
        ↓
Evidence Provenance / Lineage
Modality Availability / Provider Fingerprint
        ↓
G / A + Contact Evidence
        ↓
Frozen FEC Core
Topology / Equivalence / Local Coordination
        ↓
Leaf + Root Reference Bundle
+ Reference Compatibility Contract
        ↓
S_ctx / S_ref / U / D_cal
        ↓
Shadow Sentinel Validation
        ↓
Promote / Persist / Restore / Rollback
        ↓
Policy Evidence State
        ↓
PHM downstream
```

v2.1–v2.3 稳定 Control Plane：

- `EvidenceProvenance` / lineage audit；
- `ProviderIdentity` / deterministic provider fingerprint；
- `ModalityAvailability`：OBSERVED / DERIVED / SIMULATED / MISSING；
- `ReferenceCompatibilityContract`：modality signature + provider fingerprint + source domain；
- compatibility metadata 跨 restart persistence；
- `resolve_evidence_availability()`：上游缺失向 virtual/learned lineage 传播；
- failover provider 必须显式选择，且 transitive provider fingerprint 进入 Reference Compatibility。

Stable v2.3 class-score bridge：

- `MaskConditionalClassConformal`：只用于 genuine class evidence；
- `class-calibrate`：在独立 labelled calibration split 上持久化 mask-conditioned class conformal reference；
- `class-infer`：输出 prediction set 或 row-level `ABSTAIN`；
- missing modality 作为 modality mask 进入条件校准，不把 modality vacuity 偷换成 fault root cause。

## 目录

```text
FEC_Toolkit_v2.3/
├── fec/
│   ├── core/                       # Frozen Core
│   ├── extensions/                 # calibration / provenance / bundle / optional extensions
│   ├── runtime.py
│   └── cli.py
├── configs/
├── templates/
├── examples/
├── validation/                     # controlled Function Kill Tests
├── tests/
├── docs/
│   ├── 00_overview/
│   ├── 10_architecture/
│   ├── 20_usage/
│   ├── 30_validation/
│   ├── 40_research/
│   └── 90_history/
├── release/
├── QUICKSTART_zh-CN.md
├── CHANGELOG.md
└── pyproject.toml
```

## 快速开始

```bash
python -m pip install -e .
# 离线环境已有 setuptools 时：python -m pip install --no-build-isolation -e .
python -m pytest -q
python validation/validate_v21_assimilation.py
python validation/validate_v22_lineage_availability.py
```

最小运行和 Production Commissioning 见 `QUICKSTART_zh-CN.md`。

## Frozen Core

`fec/core/fec_core.py` SHA-256 必须为：

```text
b31a166a385a4287cdac17fc05888b52270a329ec2b6d72b1d1384ad25ad206e
```

## 不应宣称

FEC 当前不宣称 universal anomaly detector、universal root-cause classifier、novel-fault recognizer、few-shot superior model、automatic safe online learner，或任意 missingness/non-exchangeability 的万能 conformal 修复器。


## v2.3 class-score conformal bridge

FEC v2.3 新增 Extension/Control-Plane 级 `class-calibrate` / `class-infer`，用于接收**独立模型产生的 genuine class probability score pack**，在独立 labeled calibration split 上拟合 mask-conditioned conformal prediction set，并对 unseen/sparse modality mask 显式 abstain。

```bash
fec class-calibrate --calibration-scores calibration_scores.csv --reference class_ref.json --score-contract score_contract.json
fec class-infer --scores test_scores.csv --reference class_ref.json --output prediction_sets.csv
```

该桥接不修改 Frozen Core，不把 scalar anomaly `E` 转成 class belief，也不做 DOR root-cause/final diagnosis。
