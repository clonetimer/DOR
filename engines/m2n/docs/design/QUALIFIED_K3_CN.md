# Qualified K3 使用边界

K3 不是“finite 模型没选出唯一赢家”就开启。必须区分：

- K2 竞争 `AMBIGUOUS`：多个有限模型都合理 → 保留 finite set，K3 OFF；
- finite family **绝对拟合资格失败**：所有有限模型都解释不了正常数据 → 才可能进入 K3；
- observation/data qualification 不足 → `AMBIGUOUS/UNSUPPORTED`，K3 OFF。

v2.0 自动 K3 仅支持一个事先声明的 local placement。若有多个可能 placement，必须先设计 wrong-placement controls；Goal-Driven Frontend 不会自动猜 placement。

## v2.2：K3 还必须回答“未知对象是什么”

K3 是认识论等级，不等于“使用 MLP”。v2.2 在有限 K2 被正式拒绝后，再检查 `representation_semantics`：

- `LOCAL_FUNCTION` 才进入当前 native local NN；
- `DIFFERENTIAL_DYNAMICS` 建议 UDE-style；
- `CONSERVATIVE_HAMILTONIAN` 需要 HNN eligibility；
- operator/field/spatiotemporal 任务分别进入 DeepONet/FNO/PhyDNet-style contract。

如果专用 representation 已经被声明，而相应 runtime 尚未实现，M2N 返回 `SPECIALIZED_REPRESENTATION_REQUIRED`，不会静默用 local MLP 替代。

可选 `minimal_augmentation.enabled=true` 会在 K3 可行拟合后压缩 local neural complement，并输出 credit-assignment diagnostics；它不改变 K2→K3 的冻结 promotion gate。
