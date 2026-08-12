# M2N 方法边界：使用时禁止越界的几条规则

1. **Mechanism IR 不等于从自然语言自动发现机理。** 当前需要工程师提供机理骨架。
2. **K2 不等于普通超参数搜索。** 候选必须有工程/物理依据，且是有限集合。
3. **K3 不等于拟合不好就加 MLP。** 必须先完成 hard-mechanism、观测、激励、K2 拒绝资格。
4. **最低 loss 不等于物理真值。** 选择还要看 practical advantage、restart 稳定性、test、support 和语义资格。
5. **latent state 不自动具有物理语义。** Hidden-state identifiability/gauge 需要单独判断。
6. **仿真 truth 不得进入正常架构选择。** 受控盲测必须标 `BLINDED_FOR_VALIDATION`；truth 只做 posthoc。
7. **M2N evidence 不等于故障概率。** residual/disagreement/OOD 是模型证据，诊断融合属于下游。
8. **AMBIGUOUS/UNSUPPORTED 是合法输出。** 不要通过降阈值或换 loss 强行制造 SELECTED。
9. **参数可拟合不等于参数是健康指标。** 物理参数语义必须单独证明。
10. **v6.3 机理架构方法仍冻结。** v2.3 新增的是 fault-diagnostic model orchestration control plane，不修改 K0~K3/IR 核心；只有确实无法表达的通用物理原语才考虑扩 mechanism core。
11. **诊断模型竞争不等于 DOR 诊断。** M2N 可训练/比较有限 classifier/detector family，但不拥有 Fault Ontology、传播/根因推理或最终诊断仲裁。
12. **validation 与 calibration 不得复用。** model-family selection 只看 validation；FEC class calibration 使用独立 calibration split；test 不参与任何选择。
13. **正常数据不足时不制造 detector winner。** 没有独立 fault-labelled validation 时，ANOMALY_DETECTION 返回 PORTFOLIO，而不是按正常拟合误差强选 top-1。
