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
