# 外部 DOR/M2N 工具包集成实施台账

## 实施基线

- 原分支：`master`
- 原提交：`80d16ce feat: assimilate mechanism and calibration research`
- 集成分支：`integration/external-dor-m2n-v212`
- 用户既有未跟踪目录 `ref/`：不纳入、不修改。

## 台账

| 阶段 | 更新 | 状态 | 验证证据 |
|---|---|---|---|
| S0 | 两个 ZIP 的路径、symlink、CRC、SHA 和静态源码审计 | 完成 | ZIP 测试通过；DOR 558 项发布 hash、M2N 内部 hash 通过 |
| S1 | 建立独立 Git 集成分支 | 完成 | 分支名已冻结 |
| S2 | 引入 identification / ISGN bridge / multitask / vertical / freeze | 完成 | DOR 96 tests passed |
| S3 | 引入 ModelFamily、TemporalDesign、waveform、context/time/operating support | 完成 | M2N 78 tests passed |
| S4 | 保留 diagnostic_modeling、symbolic_compiler、UDE 并统一公共 API/CLI | 完成 | 既有和新增 M2N 测试共同通过 |
| S5 | 安全加载、Schema、设计审计、组件登记 | 完成 | `weights_only=True`；JSON Schema 文件入库 |
| S6 | 四工具全量回归与 CLI smoke | 完成 | DOR 96 + M2N 78 + FEC 75 + ISGN 99 = **348 passed**；实际 temporal CLI 设计/预测通过 |
| S7 | wheel 构建与隔离安装 | 完成 | DOR wheel `90b16b...2629a2`；M2N wheel `e7313e...23e1`；隔离导入/API/CLI 通过 |
| S8 | 缓存、旧 wheel、误建空目录清理 | 完成 | release hygiene PASS；用户 `ref/` 未触碰 |
| S9 | Git 集成分支提交与回并 | 执行中 | 集成提交信息 `feat: integrate external DOR and M2N capabilities`；最终 hash 见 Git log |

## 已知边界

- 外部 `joblib`、wheel 和模型 pickle 没有加载或复制入运行链。
- 外部包的测试日志仅作为自报证据；本仓库重新执行的测试才作为合并门。
- M2N 时序支路当前要求显式 trajectory/context 和近似规则采样；多速率原生联合建模仍未闭合。
- 本次合并扩展有限候选搜索，不构成任意设备全局最优保证。
