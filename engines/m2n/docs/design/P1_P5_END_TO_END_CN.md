# P1~P5 端到端使用指南

最短路径：

```bash
m2n init-design device --template graph
m2n design --spec device/device_card.json --data device/data/normal_data.csv --out device/output
m2n evidence --design-dir device/output --data observed.csv --out evidence.csv
m2n fec-export --evidence evidence.csv --out contacts.csv --contract fec_contract.json
```

`design` 内部顺序固定为：资格审查 → K0/K1 固定/参数化 → K2 competition → 必要时 backend → 必要时 qualified K3 → 冻结模型/模型集合。

## Generic Graph Device Card

`mechanism.inputs`、`mechanism.nodes`、`mechanism.outputs` 直接对应 CanonicalIR。K2 family 只通过 `bindings.<slot>.<family>` 修改局部结构。

典型 patch：

```json
{"path":"nodes.3.params.learnable","value":false}
```

或 K3：

```json
{"path":"nodes.1.op","value":"dissipative_local_neural"}
```

## Backend on demand

Device Card 可加入：

```json
{
  "backends": {
    "requests": [
      {"backend":"pysindy","slot_id":"unknown_law","feature_columns":["x1","x2"],"time_column":"time"},
      {"backend":"symmatika","slot_id":"unknown_static","feature_columns":["x1","x2"],"target_column":"y"}
    ]
  }
}
```

所有 discovery 输出均为 `UNADMITTED_PROPOSAL`。

## Qualified K3

只有 K2 slot 同时声明：

```json
{"allow_k3_escape":true,"local_neural_family":"local_nn"}
```

并设置：

```json
{"k3_policy":{"enabled":true,"finite_rejection_nrmse":0.12,"minimum_relative_improvement":0.15}}
```

才可能触发 K3。

## PHM/FEC

`m2n evidence` 不输出 fault probability/class/RUL。`m2n fec-export` 只完成 Contact Pack 语义桥；实际 FEC inference 需要独立正常 reference 和 trusted claim。
