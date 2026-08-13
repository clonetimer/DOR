# 航天器双数据集实施台账

## 2026-08-13

| 序号 | 工作项 | 状态 | 产出/证据 |
|---|---|---|---|
| 1 | 审计 ADCS/PHM 文件树、标签、官方分组和报告 | 完成 | 910 + 560 条样本；识别配对反事实与真值边界 |
| 2 | 实现 `spacecraft-prepare` | 完成 | 流式读取、路径约束、SHA-256 绑定、分组切分 |
| 3 | 实现 causal_raw/deployable_full 双视图 | 完成 | 98/144 与 144/218 通道 |
| 4 | 缺测窗口支持 | 完成 | finite-only 统计 + `observed_fraction` |
| 5 | 修复测试门控与部署门控不一致 | 完成 | test 按 `ALTERNATIVES` 语义执行 |
| 6 | 新增端到端故障宏 F1 | 完成 | 漏检记为 `NOT_EXECUTED`，不再只看容易子集 |
| 7 | 新增阈值校准就绪门 | 完成 | 3/5 正常校准样本触发 `CONDITIONALLY_QUALIFIED` |
| 8 | 修复未稳定候选缺少部署制品 | 完成 | top-2 survivor + retained artifact invariant |
| 9 | 四组真实文件树实验 | 完成 | ADCS/PHM × causal_raw/full |
| 10 | DOR 四组件回归与统一入口 | 完成 | 新增 `scripts/verify_all_tests.py`；DOR 52、M2N 36、FEC 73、ISGN 95，共 256 通过 |
| 11 | 文档、组件登记与机器结果 | 完成 | 本台账、全量审计、summary JSON、registry CSV |
| 12 | 构建与安装烟雾测试 | 完成 | wheel 构建成功；隔离 target 安装后 `spacecraft-prepare --help` 成功 |

原始 `/mnt/PRO6000_disk/Timer/sat/data/generated` 未修改。大体积派生特征与模型均位于 `/tmp/dor_spacecraft_validation`，不纳入 Git；仓库只保留可复现代码、请求、摘要和审计文档。仓库根直接 `pytest` 因三个内嵌包未统一安装而收集失败；现已用统一入口按组件设置隔离的 `PYTHONPATH`。
