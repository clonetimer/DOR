# DOR v1.0.3 Satellite R5 吸收实施台账

## 基线与来源

- 基线：`master` / `525c334`
- 实施分支：`integration/dor-v1.0.3-r5`
- 外部 ZIP SHA-256：`a3f529b81dfc2d2695d3d02a70f9e416b8ae0dc73fd04a1f5ef5baa84d4ba02b`
- ZIP 条目：1260；CRC、路径穿越和 symlink 检查通过。
- 既有未跟踪 `ref/`：未修改、未清理、未纳入提交。

## 实施记录

| 编号 | 更新点 | 决策 | 状态 | 验证 |
|---|---|---|---|---|
| R5-00 | ZIP 安全、版本、报告和源码审计 | `/tmp` 隔离解包 | 完成 | SHA/CRC/path/symlink 通过 |
| R5-01 | ISGN 参数粒子 + 仿真似然 + SMC | 吸收 | 完成 | 专项与全量通过 |
| R5-02 | 结构信息增益/成本实验排序 | 吸收 | 完成 | 区分性实验排序测试通过 |
| R5-03 | predictive adequacy、M-open、可辨识性 | 吸收并保留拒答 | 完成 | 等价结构/覆盖不足测试通过 |
| R5-04 | leave-one-intervention-out 稳定性 | 吸收 | 完成 | 稳定/不稳定分支通过 |
| R5-05 | 候选、预算、银行、阈值输入验证 | 本地加固 | 完成 | 非法值压力测试通过 |
| R5-06 | 上游宽松 DOR bridge | 拒绝直接采用，重写 firewall | 完成 | 缺隔离/坏后验/唯一性拒答通过 |
| R5-07 | LIVE 和任意执行载荷 | 禁止跨 DOR 边界 | 完成 | BLOCKED 且载荷被剥离 |
| R5-08 | 严格 advisory Schema | 吸收并收紧 | 完成 | 结构检查；可选 jsonschema 环境跳过 1 项 |
| R5-09 | CLI `active-discovery-advisory` | 新增 | 完成 | CLI smoke 通过 |
| R5-10 | R5 整包覆盖 M2N/FEC/本地 DOR | 拒绝 | 完成 | 保留符号编译、UDE、mask 审计、自动设计 |
| R5-11 | 上游 `build/` 重复目录 | 拒绝 | 完成 | 未进入仓库 |
| R5-12 | 四套全量回归 | 合并硬门 | 完成 | **445 passed，1 optional skipped** |
| R5-13 | 文档、登记、版本、发布元数据 | 更新 | 完成 | 本台账、审计报告和登记表 |
| R5-14 | wheel、隔离 smoke、SHA、卫生清理 | Git 前门 | 完成 | DOR/ISGN 新 wheel + 四工具隔离优先导入/API/CLI smoke；`verify_stack=PASS` |
| R5-15 | Git 提交 | 阶段成果 | 完成 | 提交位于 `integration/dor-v1.0.3-r5`，最终 hash 见交付信息 |
| R5-16 | `.venv` 源码/安装元数据一致性 | editable 重装 DOR/ISGN | 完成 | 源码与 distribution 均为 1.0.3+timer.1 / 2.4.0+timer.1 |

## 全量测试

| 套件 | 结果 |
|---|---:|
| DOR | 107 passed, 1 skipped（项目 `.venv` 未装可选 `jsonschema`；系统环境 Schema 专项另行 1 passed） |
| ISGN | 125 passed |
| M2N | 95 passed |
| FEC | 118 passed |
| 合计 | **445 passed, 1 skipped** |

## 版本策略

- DOR `1.0.3+timer.1`
- ISGN `2.4.0+timer.1`
- M2N `2.16.1+timer.1`（不改）
- FEC `2.4.2+timer.1`（不改）

`+timer.1` 表示 R5 语义经过本仓库边界加固，不冒充上游原包逐字发布物。
