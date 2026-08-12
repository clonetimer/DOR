# Known Limitations v2.3

1. **Contact Pack 仍是设备相关工作。** FEC 不自动发现所有物理机理。
2. **Provenance 声明的真实性由 Adapter/部署负责。** FEC 可以检查结构一致性、fingerprint 和 lineage，但不能凭字符串证明一个物理传感器/实验真的可信。
3. **Shared lineage != semantic equivalence。** v2.1 只审计，不自动改变 topology/equivalence。
4. **Reference compatibility 是已声明维度的严格匹配。** 它不能自己证明两个 domain 实际分布等价。
5. **Simulated/derived source 不自动获得 real/observed reference 权限。** 若要 transfer，必须建立并验证独立 compatibility/Sim2Real contract。
6. **MaskConditionalClassConformal 仍为 Experimental。** 它只处理 single-label genuine class evidence，依赖每个 mask 内 exchangeability 和足量 labeled calibration；组合数随 modality 数增加。
7. **缺失 mask 的 fallback 不自动完成。** sparse/unseen mask 默认 unsupported；物理兼容 parent group 必须由 Adapter 明确声明/另行实现。
8. **默认 Core leaf Evidence 是双侧经验偏差。** 有方向性机理时由合法 Adapter/Extension 表达。
9. **Calibration validity 与 diagnostic power 分离。** healthy 不证明检出强，conservative 可能 power-limited。
10. **FEC 不提供 universal drift correction / root cause / RUL / novel fault。**
11. **Durable store 稳定 profile 仍限 CoreEmpiricalLeafProvider + EmpiricalRootBackend。** Optional backend 需要显式 schema。
12. **本地 JSON SHA-256 是完整性校验，不是身份认证。** HSM/远程签名/HA 不在工具包范围内。

## v2.2 Lineage Availability 边界

- Parent 语义当前为 AND dependency；OR/conditional dependency 需要 Adapter 明确建多个 provider/source，而不是在 resolver 中猜测。
- Resolver 只判断“按声明 dependency 是否可生成”，不证明 virtual sensor 数值足够准确。
- Provider failover 必须显式选择；没有自动 provider ranking。
- 真实 ZeMA 验证中的 sensor offset/drift 是对真实过程测量做 counterfactual injection，不是现场真实传感器故障事件。
- ZeMA 数据只有 10 个严格全系统健康稳定 cycles，因此不能用于重新宣称整机 v1.5/v2.x Production Admission。


## v2.3 Class-score Bridge 边界

- 上游 score 必须是真实共享 class frame 的归一化非负概率；FEC 不负责把任意 logits 或 scalar anomaly E 猜成 class probability。
- calibration split 必须独立于上游 model-family selection validation；否则 prediction-set validity 可能被选择过程污染。
- unseen/sparse modality mask 与 empty class set 均显式 `ABSTAIN`。
- class prediction set 是统计支持集合，不等于物理 root cause、故障传播路径或部件定位。
