# 离线依赖获取与 piwheels 兜底

## 原则

M2N 不要求所有可选 backend 从 PyPI 在线安装。推荐顺序：

```text
1. 用户已有可信 wheel / 项目提供的 third_party wheel
2. PyPI
3. piwheels
4. 上游 GitHub Release / source build
```

## piwheels 使用

若 PyPI 无法访问，可查询：

```text
https://www.piwheels.org/project/<package>/
```

或使用其 simple index（具体命令按当前 pip/平台环境执行）。

但必须检查：

- 项目名和版本；
- wheel 文件名中的 Python/ABI/platform tag；
- SHA-256；
- 上游许可证；
- `METADATA` 中的依赖；
- wheel 是否适用于当前 CPU/OS。

## 本轮已验证

用户提供：

```text
PySINDy 2.1.0 wheel
derivative 0.6.3 wheel
```

两者在 Python 3.13 环境中已完成 import 与 PySINDy 内部数值微分 runtime 验证。

`derivative 0.6.3` wheel metadata 声明 MIT，且依赖 NumPy、SciPy、scikit-learn、importlib-metadata 等。
