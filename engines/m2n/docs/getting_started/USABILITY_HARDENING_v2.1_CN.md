# M2N v2.1 使用级硬化指南

v2.1 不增加新的 Scientific-ML 算法，重点减少第一次真实使用时的误操作和定位成本。

## 推荐工作流

```bash
m2n doctor
m2n validate --spec device_card.json --data data/normal_data.csv --out reports/preflight.json
m2n inspect-data --spec device_card.json --data data/normal_data.csv --out reports/data_inspection.json
m2n design --spec device_card.json --data data/normal_data.csv --out output
m2n check-bundle --design-dir output
m2n predict-design --design-dir output --data heldout.csv --out predictions.csv
```

## `validate`

检查 Device Card 顶层合同、模板、Generic Graph 节点/slot/binding 一致性，以及与 CSV 的列映射。FAIL 时不应进入架构竞争。

## `inspect-data`

检查：

- 行数与 train/validation 切分；
- 数值列非有限值；
- 常值列；
- 时间戳严格递增、重复/倒序；
- 采样间隔变化；
- 设备卡要求的列是否存在；
- 摩擦类候选是否拥有双向速度激励。

该命令是工程预检，不等价于全局结构可辨识性证明。

## `NEXT_ACTIONS.md`

`m2n design` 结束后自动生成。若状态为 AMBIGUOUS/UNSUPPORTED，它优先给出“需要补什么观测/激励/正常覆盖”，而不是建议降低阈值强行获得唯一模型。

## Design Bundle

每个 design output 会自动产生 `bundle_manifest.json`，记录文件 SHA-256。使用：

```bash
m2n check-bundle --design-dir output
```

不要在封存后原地修改模型或合同；应创建新的 output 目录。

## Architecture Portfolio 预测

`m2n predict-design` 同时支持单模型与多个 retained finite candidates。多模型时输出：

- 每个 candidate prediction；
- `portfolio_mean`；
- `portfolio_std`；
- `portfolio_min/max`。

`portfolio_std` 是模型形式分歧，不是故障概率。
