# SymMatika External Adapter 使用说明

## 定位

SymMatika 用于 **K2 symbolic candidate expansion**，不是 M2N 的替代者。

```text
qualified local slot
      ↓
SymMatika symbolic search
      ↓
UNADMITTED proposal
      ↓
M2N ProposalAdmissionCheck
      ↓
ADMITTED_K2_CANDIDATE
      ↓
normal validation / held-out evidence
```

## 为什么不是 Python dependency

上游 SymMatika 是 C++23/CMake 项目。其公开接口包括：

```cpp
DataSet(...)
Model(...)
model.run()
model.return_final_candidates()
```

M2N v1.3 采用 **source-job adapter**：用户提供上游源码路径，M2N 在临时副本中生成自己的 `main.cpp` driver，不修改用户原始源码。

## 当前支持

v1.3 只开放 explicit mapping：

```text
y = f(x1, ..., xn)
```

原因是 implicit relation 有不同的数据合同（trial/time/variables），不能偷偷复用 explicit API。

## Python API

```python
from m2n import SymMatikaExplicitRequest, prepare_symmatika_job

req = SymMatikaExplicitRequest(
    x=X,
    y=y,
    feature_names=("x1", "x2"),
    target_name="force",
    population_size=10000,
    max_depth=3,
)
job = prepare_symmatika_job(req, "symmatika_job")
print(job.driver_cpp)
```

若已经有可构建的上游 source tree：

```python
from m2n import run_symmatika_from_source
batch = run_symmatika_from_source(req, "/path/to/SymMatika", "run_out")
```

所有 `batch.proposals` 都是：

```text
UNADMITTED
```

之后必须逐条做：

- physical semantics；
- units；
- support；
- executable IR binding。

## CLI probe

```bash
m2n symmatika-probe --source-root /path/to/SymMatika
```

会检查：

- `CMakeLists.txt`；
- `model.h`；
- `main.cpp`；
- `cmake`；
- C++ compiler。

这只能证明 source-job adapter 的最小运行前提，不代表上游 SymEngine/GMP/MPFR/FLINT/Eigen 等依赖已满足。

## v1.3 验证边界

已完成：

- request validation；
- CSV/driver/contract 生成；
- machine-readable candidate parser；
- API-compatible fake CMake source harness 的 copy/build/run/parse 端到端测试。

尚未完成：

- 在当前 Linux 容器中编译真实 SymMatika upstream。

原因：上游公开安装说明主要面向 macOS/Windows，并要求完整 C++23 与外部数学库依赖；当前容器没有现成 upstream source/dependency stack。

因此当前状态是：

```text
M2N SymMatika Adapter = PASS
Upstream SymMatika Runtime in this container = NOT VERIFIED
```
