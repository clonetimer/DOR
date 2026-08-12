# Device Card templates

推荐命令：

```bash
m2n init-design my_device --template graph
m2n init-design my_mechanics --template mechanics
m2n init-design my_peak --template peak
```

CLI 生成的模板与已安装版本绑定，优先于手工复制历史模板。

v2.2 以后如需预览物理知识如何进入模型、K3 应采用何种表示，可先运行：

```bash
m2n policy-preview --spec device_card.json --out reports/policy_preview.json
```

这一步不会训练网络。
