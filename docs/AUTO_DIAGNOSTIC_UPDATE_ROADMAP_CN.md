# Auto Diagnostic Architect 更新路线图

路线以 `docs/PUBLIC_DATASET_AND_LITERATURE_GAP_AUDIT_CN.md` 的 U1—U7 为唯一编号来源。

| 优先级 | 更新 | 当前状态 | 完成定义 |
|---|---|---|---|
| P0 | U1 公开数据证据门 | 已完成 | 官方来源、许可、监督范围、split 风险、DOR 支持状态可机器校验并生成 benchmark plan |
| P0 | U2 UCI 原始多率时序 adapter | 已完成软件闭环和代表性真实源冒烟 | 官方 ZIP 严格核验 2,205×5 profile 和各传感器采样宽度；PS1/FS1/TS1 覆盖 100/10/1 Hz 真实源冒烟，输出 source hash、block split、multirate NPZ、manifest、device card；完整 17 传感器比较仍属 U4 |
| P0 | U3 多标签部件定位 | 已完成软件闭环 | `target_columns` 并发二值定位、One-vs-Rest 模型、独立 calibration 假阳性阈值、集合推理输出 |
| P1 | U4 原始时序模型阶梯 | 部分完成 | 已加入顺序敏感 temporal-pyramid 与统计视图竞争；MiniRocket/InceptionTime 参数匹配确认层待实施 |
| P1 | U5 连续参数区间辨识 | 已完成软件闭环 | 可辨识性门控、独立 calibration 绝对误差分位区间及上下界输出；分布迁移覆盖仍需真实设备验证 |
| P1 | U6 端到端级联评估 | 部分完成 | 已生成 test-only、非 teacher-forced 门控审计；事件延迟和误报/小时需带时间事件真值的数据关闭 |
| P2 | U7 机理压力测试 | 部分完成 | 已自动执行保留模型的机理特征打乱测试；错误拓扑注入和准入联动待实施 |

实施顺序为 U2 → U3 → U4/U5 → U6 → U7。当前已完成能安全进入产品通道的软件闭环；标记“部分完成”的项目仍需指定公开数据/真实设备证据，不能仅凭合成测试升级为已验证默认能力。每项后续更新仍须同步修改合同、测试、使用说明、实施台账和禁止宣称，随后清理生成物并 Git 保存。
