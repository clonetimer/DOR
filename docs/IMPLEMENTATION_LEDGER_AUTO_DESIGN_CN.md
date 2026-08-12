# Auto Diagnostic Architect 实施台账

版本目标：`DOR 0.7.0.dev0`，在不改变 v0.6 根因安全规则的前提下增加设备机理与数据驱动的诊断流水线自动设计通道。

## 台账规则

- 每项记录目标、改动、验证、证据边界和遗留风险。
- `完成` 只表示软件合同和已列测试通过，不代表任意真实设备诊断精度得到证明。
- 测试集不参与候选选择；机理确定性变换不登记为新观测。

## 2026-08-12

| 编号 | 状态 | 实施内容 | 验证/产物 |
|---|---|---|---|
| AD-001 | 完成 | 审计 DOR 0.6、M2N 2.3.1 与现有有限模型组合 | 确认旧主链仅稳定支持数值表格检测/分类，定位和根因是独立语义推理 |
| AD-002 | 完成 | 新增设备卡、自动设计请求和多模态数据清单合同 | `schemas/*_1.0.schema.json` |
| AD-003 | 完成 | 新增 TABULAR、TIMESERIES_LONG、TIMESERIES_NPZ 数据 IR | 时序压缩明确记录相位/事件顺序信息损失 |
| AD-004 | 完成 | 新增结构化机理特征编译 | 支持 difference/ratio/sum/product/linear_residual；输出标记为确定性表示 |
| AD-005 | 完成 | 新增检测、定位、分类、辨识资格审查 | 缺标签、验证新类、辨识秩不足均 fail closed |
| AD-006 | 完成 | 新增有限候选竞争、分种子验证、实际差距与稳定性选择 | SELECTED/FIXED/ALTERNATIVES/PORTFOLIO/UNSUPPORTED |
| AD-007 | 完成 | 新增条件级联 DAG 和可运行 joblib 模型包 | `pipeline_bundle/bundle.json`、`dor auto-predict` |
| AD-008 | 完成 | 新增搜索哈希绑定、完成态恢复和非保留模型清理 | `search_state.json`；只保留胜出或等价模型文件 |
| AD-009 | 完成 | 新增组件卡、比较矩阵、架构方案、实验计划、技术扫描和组件登记 | 每次设计生成 `audit_report.md` 与 `component_registry.csv` |
| AD-010 | 完成 | 完整回归、示例 smoke、目录清理和 Git 保存 | DOR 38、M2N 36、FEC 73、ISGN 95 项测试通过；提交号见 Git 日志 |

## 2026-08-13

| 编号 | 状态 | 实施内容 | 验证/产物 |
|---|---|---|---|
| AD-011 | 完成 | 检索七个官方公开诊断/退化数据源及原始方法文献，区分可支持和不可支持的阶段证据 | `docs/PUBLIC_DATASET_AND_LITERATURE_GAP_AUDIT_CN.md` |
| AD-012 | 完成 | 将数据许可、故障来源、stage evidence、DOR support、split unit、泄漏风险和禁止宣称固化为机器可读目录 | `benchmarks/public_datasets.json`、catalog schema |
| AD-013 | 完成 | 新增公开目录 fail-closed 校验和按数据集/阶段生成 benchmark plan 的 CLI | `dor dataset-catalog`、`dor benchmark-plan` |
| AD-014 | 完成 | 将剩余差距拆为 U1—U7，并绑定优先级、实验对照、度量和验收门槛 | `docs/AUTO_DIAGNOSTIC_UPDATE_ROADMAP_CN.md` |
| AD-015 | 完成 | 执行目录单元测试、UCI 派生视图冒烟、全栈回归和缓存清理 | DOR 44、M2N 36、FEC 73、ISGN 95；UCI 派生视图 1,449 行计划资格为 `CONDITIONALLY_QUALIFIED`；Git 提交见日志 |
| AD-016 | 完成 | 新增官方 UCI Hydraulic ZIP 严格适配器和 `MULTIRATE_NPZ` IR；按相邻周期 block 切分并绑定源 hash | `dor uci-hydraulic-prepare`、`src/dor/public_adapters.py` |
| AD-017 | 完成 | 自动设计器新增并发多部件定位、独立 calibration 假阳性阈值和 prediction-set 输出 | `tests/test_dor_auto_design.py::test_multilabel_localization_is_calibrated_and_emits_sets` |
| AD-018 | 完成 | 辨识模型新增独立 calibration 残差区间，并在推理时输出 lower/upper | 四阶段 bundle 测试覆盖 |
| AD-019 | 完成 | 新增 test-only 非 teacher-forced 级联审计与机理特征打乱压力审计 | `end_to_end_test_audit.json`、`mechanism_stress_audit.json` |
| AD-020 | 完成 | 加入可选 temporal-pyramid 顺序敏感视图，与统计视图在相同预算内竞争 | NPZ 时序测试验证候选视图存在；不宣称 MiniRocket 已实现 |
| AD-021 | 完成 | 下载官方 UCI Hydraulic ZIP 到临时目录，核验 SHA-256，以 PS1/FS1/TS1 覆盖 100/10/1 Hz 做 2,205 周期真实源冒烟并清理大文件 | source hash `24128aad...f1f2`；设计 `QUALIFIED`；220 calibration / 225 test；三传感器范围不可外推为完整 17 传感器比较 |

## 当前能力边界

- 时序首版采用可审计统计/频带表示，没有将 CNN/TCN/GRU/Transformer 冒充为已验证候选。
- 定位首版为单标签部件分类；并发多部件定位继续由已有 ontology/evidence localizer 处理。
- 辨识只处理有显式参数变化监督且目标变化满秩的回归任务。
- FEC 校准仍由既有运行链负责；自动设计器严格隔离 calibration/test，但本开发预览不自动执行所有阶段的 FEC 校准。
- “最合适”仅指声明候选、预算和 validation 合同内的稳定最佳或等价集合。

## 后续真实设备关闭门槛

1. 至少一个真实多通道时序设备数据集完成按 run/asset 隔离的四阶段验证；
2. 增加事件级检测延迟、误报/小时和端到端非 teacher-forced 指标；
3. 对时序神经编码器做参数/延迟匹配与多种子验证后再准入默认 registry；
4. 对多标签定位和参数置信区间完成独立校准；
5. 所有机理分支完成移除、打乱与错误拓扑压力测试。

以上门槛现统一映射为路线图 U2—U7；其中公开数据证据门 U1 已完成，但它只证明 benchmark 准入规则可执行，不证明任何模型在原始公开数据上的性能。
