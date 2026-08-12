# Persistence & Recovery Guide

## 保存

Python API：

```python
from fec.extensions.reference_bundle_store import save_reference_bundle_manager
save_reference_bundle_manager(manager, "reference_state.json")
```

## 加载

```python
from fec.extensions.reference_bundle_store import load_reference_bundle_manager
manager = load_reference_bundle_manager("reference_state.json")
```

加载前会校验 payload SHA-256。

## 为什么不用 pickle

Reference state 是生产控制面数据，不应包含可执行 Python object graph。v2.1 延续使用显式 JSON schema，只允许已声明的 key/provider/backend 类型。

## 保存原子性

保存过程：

1. 写同目录临时文件；
2. flush + fsync；
3. `os.replace()` 原子替换目标文件。

这降低进程异常导致半写文件的风险。

## Shadow candidate

shadow/rejected candidate 不进入 durable state。重启后必须重新 stage 和独立 sentinel validation，避免未经 promotion 的中间状态意外恢复为生产状态。

## 备份建议

工具本身提供 active/retired 逻辑 rollback；设备部署仍建议把每次 promotion 后的 state 文件做外部版本化备份。外部备份策略属于部署系统，不由 FEC Toolkit 自动实现。
