# DOR v1.0.2 Satellite R4 吸收实施台账

## 基线与来源

- 基线分支/提交：`master` / `a1c9408`
- 实施分支：`integration/dor-v1.0.2-r4`
- 外部 ZIP SHA-256：`31521fcc3eee57cb7102b62bd865e5be8200b4599438a59f221c485b4694b9b7`
- ZIP 条目：903；CRC 通过；无绝对路径、`..` 路径或符号链接。
- 用户目录 `ref/`：既有未跟踪资料，未修改、未纳入提交。

## 实施记录

| 编号 | 更新点 | 决策 | 状态 | 验证 |
|---|---|---|---|---|
| R4-00 | ZIP 安全、版本与发布声明审计 | 静态检查后解包到 `/tmp` | 完成 | CRC、路径、symlink 检查通过 |
| R4-01 | ISGN 时间基完整性 | 吸收 | 完成 | observation gate 专项和全量通过 |
| R4-02 | ISGN 时间充分性预检 | 吸收；结果只作可辨识性门 | 完成 | fail-closed 专项通过 |
| R4-03 | ISGN bounded Tabulated1D | 吸收；禁止外推/多值混合 | 完成 | 越界、多值、同工况测试通过 |
| R4-04 | M2N 波形回归/阶次域/候选级表示 | 吸收 | 完成 | 新增三组测试通过 |
| R4-05 | M2N H 步递归选择 | 吸收；DOR 合同为唯一时距来源 | 完成 | 合同绑定、冲突拒答、全量通过 |
| R4-06 | R4 `weights_only=False` 回退 | 拒绝 | 完成 | 运行路径保持 `weights_only=True` |
| R4-07 | R4 将 UDE 降为合同状态 | 拒绝 | 完成 | UDE/符号编译回归通过 |
| R4-08 | FEC 资产/工况 Reference 绑定 | 吸收 | 完成 | scope/context 测试通过 |
| R4-09 | FEC 舰队逐资产 commissioning/robustness | 吸收 | 完成 | fleet 测试通过 |
| R4-10 | R4 移除 mask audit CLI | 拒绝 | 完成 | 原 mask CLI 测试通过 |
| R4-11 | DOR ISGN observation fail-closed bridge | 吸收 | 完成 | DOR bridge 测试通过 |
| R4-12 | DOR fleet scope simple-bridge guard | 吸收 | 完成 | asset/fleet admission 测试通过 |
| R4-13 | Schema、版本与 CLI | 使用 `+timer.1` 标明下游增强 | 完成 | py_compile 与 CLI 测试通过 |
| R4-14 | 四套全量回归 | 合并硬门 | 完成 | **423 passed** |
| R4-15 | 文档、架构审计、组件登记 | 更新 | 完成 | 本台账及配套文档 |
| R4-16 | wheel、release hash、卫生清理 | Git 前门 | 完成 | 四 wheel 构建与隔离优先导入/API/CLI smoke 通过；旧 wheel、build、egg-info、cache 已清理 |
| R4-17 | Git 提交 | 阶段成果 | 准备完成 | 最终 commit hash 以 `git log -1` 与交付信息为准 |

## 全量测试

| 套件 | 结果 |
|---|---:|
| DOR | 101 passed |
| ISGN | 109 passed |
| M2N | 95 passed |
| FEC | 118 passed |
| 合计 | **423 passed** |

已知非失败告警：Joblib/NumPy shape 弃用告警；PyTorch nested tensor prototype 告警。二者未被记为机制或精度证据。

## 版本策略

- DOR `1.0.2+timer.1`
- ISGN `2.3.0+timer.1`
- M2N `2.16.1+timer.1`
- FEC `2.4.2+timer.1`

本地后缀表明它们是 R4 语义加 Timer 仓库增强，不冒充提供方原始 wheel 的逐字发布物。
