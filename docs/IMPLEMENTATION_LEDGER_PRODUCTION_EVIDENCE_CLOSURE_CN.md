# DOR 1.0.4 Production Evidence Closure Phase 1 — 实施台账

日期：2026-09-30

## A. 目标

把“软件可运行”与“生产有效性”分离，并针对上一轮独立验证暴露的问题进行可复现修复，不新增未经证据支持的诊断能力声明。

## B. 修改项

| ID | 修改 | 状态 | 验证 |
|---|---|---|---|
| PEC-01 | 新增 `DOR-PRODUCTION-EVIDENCE-GATE/1.0` | DONE | 3 个新单测；CLI wheel smoke |
| PEC-02 | 新增 OPSSAT per-channel candidate remediation | DONE | F1 0.6814 -> 0.8250；test 不参与 candidate selection |
| PEC-03 | 新增 Hydraulic independent model/stress replay | DONE | test macro-F1 0.9706；nuisance baseline 0.25 accuracy |
| PEC-04 | 修复 `verify_all_tests.py` clean-source import roots | DONE | DOR 111 passed；M2N 95 passed；FEC 118 passed |
| PEC-05 | 修复 release cache hygiene | DONE | final `verify_stack.py` release_hygiene gate |
| PEC-06 | 冻结下一轮 prospective acceptance policy | DONE | 当前 test 不允许 retroactive qualification |
| PEC-07 | 构建 DOR 1.0.4 wheel | DONE | 四 wheel 安装 + `dor stack-info` + production gate smoke |
| PEC-08 | 完整 UCI 2,205 raw prospective benchmark | BLOCKED | 官方 archive 不在包内；当前环境下载端点失败 |
| PEC-09 | NASA ADAPT real-run root cause | BLOCKED | 仍缺完整 telemetry / Experiment Log / time alignment |

## C. Fresh regression

- DOR: 111 passed
- M2N: 95 passed
- FEC: 118 passed
- ISGN: 125 tests collected；monolithic run 超出当前执行窗口，不声明 fresh 125 passed

当前 assisted POC route 的 OPSSAT detection 与 hydraulic classification 不 admit ISGN，因此本轮 POC gate 可成立；任何 Active Discovery / root-cause production 声明仍被阻断。

## D. OPSSAT remediation

相同 523 supported rows：

| metric | old | 1.0.4 remediation |
|---|---:|---:|
| precision | 0.6638 | 0.7615 |
| recall | 0.7000 | 0.9000 |
| F1 | 0.6814 | 0.8250 |
| balanced accuracy | 0.8028 | 0.9125 |
| FPR | 0.0944 | 0.0751 |
| FN | 33 | 11 |

`CADC0892` F1 从旧结果约 0.182 提升到 0.7059。

证据限制：历史 test 已知，因此本项是 remediation，不是 blind benchmark。

## E. Hydraulic stress

- selected on validation: HistGradientBoosting
- test macro-F1: 0.9706
- nuisance-only accuracy: 0.25
- RAW+MECHANISM vs RAW HGB delta: +0.00845 < 0.02 practical margin
- 5% synthetic feature-noise macro-F1: 0.8066
- 10%: 0.6405
- artificial FS2 full-mask macro-F1: 0.8518

这些 synthetic stresses 不升级为真实 drift / outage qualification。

## F. Release semantics

预期最终门禁：

```text
ASSISTED_POC              SUPPORTED
ASSISTED_PRODUCTION       BLOCKED
SAFETY_CRITICAL_AUTONOMY  BLOCKED
```

任何未来修改若把后两项改为 `SUPPORTED`，必须同时提供对应 gate 的新独立证据，不能只改 JSON。
