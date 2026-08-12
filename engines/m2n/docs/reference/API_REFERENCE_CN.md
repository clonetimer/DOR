# M2N Practitioner 常用 API 速查

## Knowledge / Planning

### `KnowledgeLevel`
`K0 | K1 | K2 | K3`

### `KnowledgeSource`
`DECLARED_ENGINEERING | COMMISSIONING_EVIDENCE | FINITE_ENGINEERING_HYPOTHESIS | BLINDED_FOR_VALIDATION | SIMULATOR_TRUTH_POSTHOC`

### `MechanismSlot`
描述一个局部机理槽位。

### `SlotEvidence`
描述数据/观测/hard-mechanism/K2 资格。

### `plan_neural_architecture(slots, evidence_by_slot)`
输出 `NeuralArchitecturePlan`。

## Lowering / Compile

### `FieldAssignment(path, value)`
在 frozen dataclass IR 中局部修改字段。

### `FamilyPatch(family, assignments)`
把一个 K2 family 名字映射到 IR 修改。

### `SlotBinding(slot_id, family_patches)`
把知识 slot 和可执行 IR 绑定。

### `lower_architecture_plan(...)`
生成有限候选 IR 空间。

### `compile_executable_ir(ir)`
编译当前支持的 IR 为 PyTorch module。

## Selection

### `ArchitectureSelectionPolicy`
默认 `practical_ratio=0.80`, `restart_stability=0.75`。

### `run_design_session(...)`
一站式执行 candidate lowering、compile、normal-validation selection。

## IR

- `SecondOrderMechanicsIR`
- `ConservationIR`
- `FourWayValveIR`
- `BlockLNLIR`
- `SeparablePeakIR`
- `CanonicalIR`

## Portfolio / Evidence

### `ArchitecturePortfolio(models, unresolved_slots)`
执行多个有限候选，保留 candidate identity 和 model-form spread。

### `RobustSupportReference.fit(X)`
冻结正常支持域。

### `portfolio_derivative_evidence(...)`
生成 residual/disagreement 等模型证据。

## Observation

### `ObservationContract`
声明状态和观测状态。

### `plan_observation(...)`
在需要隐藏状态物理解释时做局部 observability 资格判断。

---

# v1.3 外部 Backend API

## PySINDy

```python
from m2n import DynamicsDiscoveryRequest, PySINDyCandidateBackend
batch = PySINDyCandidateBackend.propose(request)
```

返回的 `CandidateProposal` 默认不具备物理真值资格，必须使用：

```python
ProposalAdmissionCheck
admit_candidate_proposal(...)
```

## SymMatika

```python
from m2n import (
    SymMatikaExplicitRequest,
    probe_symmatika_source,
    prepare_symmatika_job,
    run_symmatika_from_source,
)
```

`run_symmatika_from_source()` 会复制用户提供的上游 source tree 到输出目录，在副本中替换 `main.cpp` 为 M2N driver，再执行 CMake build/run；不会修改用户原始源码。

## SBMI feasibility

```python
from m2n import (
    ModelComponentSlot,
    ParameterPrior,
    SBMIModelSetRequest,
    assess_sbmi_feasibility,
    export_sbmi_contract,
)
```

其输出是“是否值得进入 simulation-based model-set inference”的工程资格，不是 SBMI posterior 本身。


# v2.0 P1~P5 API

## Generic Graph

- `build_generic_canonical_ir(spec)`：Device Card → CanonicalIR。
- `build_generic_problem(...)`：从声明式 graph + Knowledge Contract 生成可 lowering 问题。

## Automatic qualification

- `qualify_tabular_design(...)`：required columns、finite、样本量、target variation、rank/condition、train/validation support。

## Backend routing

- `run_backend_requests(...)`：仅在 qualification READY 后运行显式请求的 PySINDy/SymMatika。
- `assess_model_set_routing(...)`：有限组合过大时输出 model-set inference contract。

## Qualified K3

- `finite_gate(...)`：判断 finite K2 是否真的被绝对拟合门拒绝。
- `compare_k3(...)`：K3 必须同时满足相对改善和可选绝对 NRMSE 门。

## PHM/FEC

- `generate_model_evidence(...)`：冻结模型/模型集合 → residual/disagreement/support。
- `export_fec_contact_pack(...)`：sample evidence → window Contact Pack。
- `calibrate_fec_from_contacts(...)` / `infer_fec_contacts(...)`：调用外部 FEC Core。
