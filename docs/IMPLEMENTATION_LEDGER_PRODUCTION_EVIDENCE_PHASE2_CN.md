# DOR 1.0.5 Production Evidence Closure Phase 2 — 实施台账

| ID | 状态 | 修改 | 验证 |
|---|---|---|---|
| PEC2-01 | DONE | 新增 `prospective_evidence.py`，精确 hash 绑定冻结 policy | 当前 OPSSAT 16/16 metrics PASS 但 `INELIGIBLE` |
| PEC2-02 | DONE | 新增 `prospective-evaluate` CLI | 非 prospective 输入返回 exit code 2 |
| PEC2-03 | DONE | 新增 `deployment_ops.py` | selected segments 返回 `PROXY_ONLY` |
| PEC2-04 | DONE | OPSSAT duration 与 remediation predictions 绑定 | 31 FP / 28.3575 selected nominal h = 1.093 proxy/h |
| PEC2-05 | DONE | 新增 `source_intake.py` | hash、文件名、size、ZIP member fail-closed tests |
| PEC2-06 | DONE | 固化 UCI Hydraulic 官方 intake 约束 | 18 required members；后续 adapter 校验 2205-cycle shapes |
| PEC2-07 | DONE | 固化 NASA ADAPT 三个官方资源 intake spec | runtime 无 DNS，仍未获得 binary bytes |
| PEC2-08 | DONE | 新增 3 个 schema | prospective / deployment ops / public source intake |
| PEC2-09 | DONE | 新增 9 个单测 | 首轮 9/9 passed |
| PEC2-10 | IN PROGRESS | 全 DOR / M2N / FEC / ISGN 回归与 wheel 重建 | 发布前执行 |
| PEC2-11 | BLOCKED_EXTERNAL_DATA | 完整 UCI 2,205 raw prospective/new-session 证据 | 官方 archive 仍需进入可执行环境 |
| PEC2-12 | BLOCKED_EXTERNAL_DATA | NASA ADAPT real-run root cause | 仍需 dataset_text + Experiment Log + topology bytes |

## 关键不升级项

本阶段**不**把 `PROSPECTIVE_ACCEPTANCE_POLICY`、`NEW_SESSION_OR_ASSET`、`DEPLOYMENT_OPERATIONS`、`REAL_FAULT_INJECTION` 改成 PASS。新增代码只使这些 gate 可被客观关闭；当前真实证据仍不足。
