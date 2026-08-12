# M2N 常见问题排查

## `m2n doctor` 失败

先检查：

```bash
python -c "import numpy, scipy, torch; print(torch.__version__)"
```

若 PyTorch 未安装，请按你的 CPU/CUDA 环境安装合适版本，再安装 M2N wheel。

## `architecture plan is AMBIGUOUS; ... must not be compiled`

这是保护机制，不是代码错误。检查：

- `SlotEvidence.data_qualification`；
- `observation_qualification`；
- `hard_mechanism_status`；
- K2 是否真的已经有资格搜索。

## `no executable binding for family`

Planner 的 `finite_candidates` 和 `SlotBinding.family_patches` 名字没有一一对应。

## candidate 数量爆炸

M2N 默认阻止过大的 K2 交叉乘积。应拆成局部顺序实验或减少没有物理依据的候选，而不是提高 `max_candidates` 变成 NAS。

## `AMBIGUOUS` 很多

这可能是正确结果。先检查：

1. 工作条件是否激发候选差异；
2. 传感器是否对局部机理有信息；
3. Estimation Contract 是否噪声敏感；
4. 是否遗漏 hard mechanism。

不要第一反应就降低 `practical_ratio`。

## K1 参数和真实物理参数差很多

预测参数可能在 reduced model 中发生补偿。只有经过独立辨识/语义资格后，才能把参数变化解释成健康退化。
