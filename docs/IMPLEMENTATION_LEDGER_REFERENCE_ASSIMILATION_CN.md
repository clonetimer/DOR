# 参考文献吸收更新台账

## 2026-08-13：阶段 1

| 编号 | 工作项 | 状态 | 成果/证据 |
|---|---|---|---|
| RA-001 | 审计 14 篇用户补充文献与四工具代码重合度 | 完成 | 高价值链路锁定为机理发现、可微编译/UDE、缺失模态校准 |
| RA-002 | 扩展 DOR 2024–2026 方法目录 | 完成 | 新增 5 条近期方法；PFM 记录为 `DEFER_OUT_OF_SCOPE` |
| RA-003 | 实现 M2N safe symbolic compiler | 完成 | allowlist AST、严格/guarded domain、合同与精确梯度测试 |
| RA-004 | 实现 M2N fixed-step Hybrid UDE | 完成 | symbolic vector field、局部/全量残差、RK4轨迹和梯度 |
| RA-005 | 升级 M2N differential representation | 完成 | 资格满足时从 `CONTRACT_ONLY` 变为 `NATIVE_FIXED_STEP_RK4` |
| RA-006 | 实现 ISGN evidence-aware discovery route | 完成 | SINDy/trajectory/SymMatika/blocked 四类路径；隐式桥不夸大 |
| RA-007 | 实现 FEC mask calibration held-out audit | 完成 | 覆盖、集合效率、top accuracy、unsupported逐mask报告及CLI |
| RA-008 | 实现 DOR mask calibration budget gate | 完成 | 显式计算 `2^M-1`，总数不替代逐组资格，5+模态提示组合爆炸 |
| RA-009 | 定向新增测试 | 完成 | 13 passed |
| RA-010 | 四工具全量回归 | 完成 | DOR 55、M2N 40、FEC 75、ISGN 99；合计 269 passed |
| RA-011 | 组件卡、比较矩阵、架构、实验、技术扫描和登记 | 完成 | `REFERENCE_ASSIMILATION_STAGE1_CN.md` 与 validation 登记 |
| RA-012 | 目录和缓存清理 | 完成 | 已清理仓库内 Python/pytest 缓存；未触碰 `.venv/`、`ref/` 和原始数据 |
| RA-013 | Git 阶段提交 | 完成 | 由本台账所在提交保存；只暂存本阶段代码、测试、文档和验证摘要，不暂存用户 `ref/` PDF |

## 遗留队列

- P0：取得 SymMatika 上游源码后执行真实 build/run；当前只验证适配和路由合同。
- P1：在具有可信部分微分方程的设备数据上做 UDE vs 手写 vs 等参数/等延迟 MLP。
- P1：用真实 M2N class score pack 执行 FEC `mask-calibration-audit`，逐 mask 报告覆盖和效率。
- P1：为两个航天器合成集补足分层正常 calibration；当前 3/5 条不满足既有部署门。
- P2：按任务资格逐步增加自适应/刚性 solver；DeepONet/FNO/HNN 保持合同级，不因论文存在就默认启用。
