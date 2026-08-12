# DOR v0.6 Propagation / Root-Cause Reasoner — Kickoff Gate

状态：**STARTED / NOT CLOSED**

v0.5 完成 Localization 后，v0.6 开始研究：

```text
localized fault/component hypotheses
+ directional topology
+ temporal evidence
+ command / operating-mode context
+ mechanism compatibility
+ contradictions
→ root-cause candidate sets
+ downstream propagation paths
```

但本轮不宣布 v0.6 完成。原因：完整 NASA ADAPT telemetry/topology/fault-injection artifacts 尚未进入本地可重复验证链。

## v0.6 必须满足的安全规则

1. earliest anomaly ≠ root cause；
2. topology upstream ≠ root cause，除非 evidence 与 temporal/mechanism constraints 同时支持；
3. propagation path 必须引用具体 topology edges；
4. sensor fault 与 physical component fault 必须可分离；
5. concurrent independent root causes 必须被允许；
6. 若多条 root-cause sets observationally equivalent，返回 ALTERNATIVES；
7. Test fault log 只能用于 post-hoc evaluation，不能参与 reasoner rule 选择。

## 数据门槛

优先 NASA ADAPT：真实硬件 EPS testbed、受控 fault injection、commands + sensors + topology。正式 v0.6 closure 需要把这些公开资源转化为：

- ADAPT topology graph；
- fault-event manifest；
- time-aligned evidence；
- independent truth log；
- propagation/root-cause evaluation protocol。
