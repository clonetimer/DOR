# DOR Diagnostic Stack — 验证状态

## 0. 当前 R4-derived 集成状态（2026-08-14）

当前源码：DOR 1.0.2+timer.1 / ISGN 2.3.0+timer.1 / M2N 2.16.1+timer.1 / FEC 2.4.2+timer.1。

| 套件 | fresh full regression |
|---|---:|
| DOR | 101 passed |
| ISGN | 109 passed |
| M2N | 95 passed |
| FEC | 118 passed |
| 合计 | **423 passed** |

这些结果支持当前源码的软件兼容性和安全合同，不是新设备盲测准确率、跨设备泛化或飞行部署证据。完整吸收审计见 `DOR_V102_R4_ASSIMILATION_CN.md`。下文保留早期 v0.6 历史状态，避免覆盖既有开发证据。

## 1. 生命周期结论

**v0.5：CLOSED as DEV RC1**  
**v0.6：STARTED / EXECUTABLE PROTOTYPE / NOT CLOSED**

## 2. DOR fresh regression

```text
DOR v0.6 full suite       34 passed
root-cause dedicated       10 passed
```

v0.6 full suite 包含 v0.3/v0.4/v0.5 回归及 v0.6 root-cause tests。

## 3. 前一开发节点的 fresh engine regression

在 v0.5 发布硬化时重新执行：

| 组件 | 结果 | 说明 |
|---|---:|---|
| DOR 0.5.0 | **24 passed** | 修复 contract branch 一致性后的最终结果 |
| M2N 2.3.1 | **36 passed** | fresh full tests |
| FEC 2.4.0 | **73 passed** | fresh full tests |
| ISGN 2.2.0 | `isgn doctor = PASS` | 未把历史 full-suite 冒充 fresh result |

v0.6 没有修改 ISGN/M2N/FEC Core，因此本原型包不重复制造新的三个引擎 benchmark 数字。

## 4. v0.6 Schema

新增：

```text
schemas/root_cause_result_0.6.schema.json
```

已对 source-constrained NASA fixture output 做实际 schema validation。

## 5. v0.6 safety regression

已覆盖：

- unique root 必须有 timestamp + provenance path；
- temporal contradiction 不被强行解释；
- unprovenanced topology 不支持 propagation；
- no timestamp → UNKNOWN；
- concurrent independent root causes；
- equivalent root sets → ALTERNATIVES；
- localization-only → UNKNOWN，不冒充 root cause。

## 6. NASA ADAPT 当前证据等级

NASA ADAPT 是真实 NASA Ames aerospace-representative EPS testbed，并支持受控 fault insertion；但本原型包内的 v0.6 fixture 不是一条真实实验 run。

当前 fixture：

```text
source-supported component/topology semantics
+ synthetic timestamps/evidence
→ software contract validation only
```

因此禁止报告 root-cause empirical accuracy。

## 7. v0.6 尚未通过的退出门槛

- 完整 ADAPT telemetry 未本地获取；
- 完整 EPS diagram topology 未转为 DOR ontology；
- Experiment Log 尚未转为 independent truth manifest；
- real-run fault onset 与 evidence 尚未 time-align；
- sensor/component/command fault 全类型真实回归未完成；
- multiple-fault real runs 尚未进入 benchmark。

这些是 v0.6 关闭条件，而不是可以通过增加更多单元测试替代的证据。
