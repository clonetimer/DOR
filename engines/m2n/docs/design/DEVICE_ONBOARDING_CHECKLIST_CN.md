# 新设备接入 M2N：执行清单

## A. 任务定义

- [ ] 预测目标/状态目标写清楚；
- [ ] 正常训练、正常 validation、独立 test 的作用分开；
- [ ] 故障标签不参与架构选择。

## B. 机理知识

对每个局部 slot 填：

- [ ] ID；
- [ ] 输入量及单位；
- [ ] 输出量及单位；
- [ ] K0/K1/K2/K3；
- [ ] KnowledgeSource；
- [ ] hard constraints；
- [ ] 如果 K2，候选集合在看 validation/test 前冻结；
- [ ] 如果是受控盲测，标 `BLINDED_FOR_VALIDATION`。

## C. 观测

- [ ] 哪些生产可测；
- [ ] 哪些只在研发台架可测；
- [ ] 哪些是 simulator truth；
- [ ] `SIMULATOR_TRUTH_POSTHOC` 不进入 Planner；
- [ ] 隐藏状态有无 observability 支持。

## D. 数据

- [ ] train/val/test 明确分离；
- [ ] 单位一致；
- [ ] 时间同步/延迟处理有记录；
- [ ] 正常工作区间覆盖足够；
- [ ] OOD/support 域有记录。

## E. Estimation Contract

- [ ] one-step / rate / free-rollout / inverse 明确选择；
- [ ] 多 restart；
- [ ] 参数边界有物理理由；
- [ ] validation risk 的归一化方式固定；
- [ ] selection policy 预先冻结。

## F. 结果

- [ ] SELECTED：test 在架构冻结后打开；
- [ ] AMBIGUOUS：保留有限候选，不自动 K3；
- [ ] UNSUPPORTED：指出缺哪个观测/资格；
- [ ] 参数物理语义与预测性能分开声明；
- [ ] 保存 architecture plan / IR / model / provenance。
