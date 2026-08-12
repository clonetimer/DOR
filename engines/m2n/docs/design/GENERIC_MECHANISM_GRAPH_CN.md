# Generic Mechanism Graph Device Card 参考

## 输入

```json
{"name":"velocity","dim":{"length":1,"time":-1}}
```

## 节点

节点必须使用 M2N CanonicalIR 已支持的 primitive，例如 `add`、`subtract`、`multiply`、`positive_gain`、`signed_gain`、`sign`、`abs`、`sqrt_pos`、`power_pos`、`stribeck_resistance`、局部 neural primitive 等。

## 输出

每个输出必须显式声明物理维度；编译前执行维度一致性验证。

## Knowledge slot

K0/K1 无需 family branching；K2/K3 通过 `bindings` 将 family label 映射到局部 graph patch。禁止在 JSON 中嵌入任意 Python 表达式。
