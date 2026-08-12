# M2N Practitioner Kit v2.3

**Mechanism-to-Model / Scientific-ML + Diagnostic Model Orchestration Toolkit**

M2N v2.3 有两条受同一资格原则约束的主入口：

```text
A. 正常机理/Scientific-ML 设计
设备机理 + 真实观测合同 + 正常数据
                ↓
       资格审查 / K0~K3
                ↓
   受约束 Scientific-ML 架构/模型集合
                ↓
      PyTorch Design Bundle
                ↓
        Model Evidence / FEC

B. 故障诊断模型设计（v2.3）
DOR 任务合同 + 故障/正常数据 + raw/mechanism features
                ↓
     任务/数据/特征视图资格审查
                ↓
  有限统计/传统ML/神经基线候选竞争
                ↓
 Diagnostic Model Bundle + class/anomaly scores
                ↓
        FEC calibration → DOR
```

v2.3 不增加第五个工具，而是在 M2N 内补上此前缺失的“谁负责训练和比较诊断模型”这一层。它不是全局 AutoML：候选集合有限、必须显式分割 train/validation/calibration/test、允许 AMBIGUOUS/PORTFOLIO/UNSUPPORTED，最终故障语义与根因仍属于 DOR。v2.2 的 Scientific-ML bias/representation 原则保持不变。

## 1. 目录

```text
M2N_Practitioner_Kit_v2.3/
├── README.md
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── _SHA256SUMS.txt
├── src/                  # M2N runtime
├── configs/              # Schema / 声明式配置模板
├── examples/             # basics / design / backends / phm
├── tests/                # 自动化测试
├── scripts/              # 安装、验证、发布 hygiene
├── docs/                 # 面向使用者的分层文档
├── reports/              # release / validation / backend / research
├── third_party/          # 明确许可证与 hash 的离线依赖
└── dist/                 # wheel
```

历史研究报告、NKT JSON、验证日志不再平铺在包根目录。完整源码包按工程规范把使用文档、报告、测试和第三方依赖分区管理。

## 2. 最短使用路径

```bash
python -m pip install dist/m2n_toolkit-2.3.0-py3-none-any.whl
m2n doctor
m2n init-design my_device --template graph
cd my_device
m2n validate --spec device_card.json --data data/normal_data.csv
m2n design --spec device_card.json --data data/normal_data.csv --out output
m2n check-bundle --design-dir output
m2n predict-design --design-dir output --data data/normal_data.csv --out predictions.csv
```

设计完成后 `output/NEXT_ACTIONS.md` 给出下一步建议，`output/bundle_manifest.json` 固化文件 hash。

## 3. 关键使用命令

```bash
m2n policy-preview --spec device_card.json --out reports/policy_preview.json
m2n validate --spec device_card.json [--data normal.csv]
m2n inspect-data --spec device_card.json --data normal.csv
m2n predict-design --design-dir output --data heldout.csv --out predictions.csv
m2n seal-bundle --design-dir output
m2n check-bundle --design-dir output
```

`predict-design` 对唯一模型和有限 Architecture Portfolio 使用同一入口；Portfolio 的 mean/std 表示模型形式不确定性，不是故障概率。

## 4. 故障诊断模型入口（v2.3）

```bash
m2n diagnostic-families --task FAULT_CLASSIFICATION
m2n diagnostic-design --request diagnostic_request.json --data fault_data.csv --out diagnostic_output
m2n diagnostic-predict --design-dir diagnostic_output --data deployment.csv --out model_scores.csv
```

当前稳定范围是**数值 tabular** 的 `FAULT_CLASSIFICATION` 与 `ANOMALY_DETECTION`。支持 RAW、MECHANISM、RAW_PLUS_MECHANISM 三种显式特征视图；机制特征可以来自 ISGN/M2N 已认证的参数、残差或工程特征，但“机制特征有用”不得直接解释成“该部件就是根因”。分类模型输出是未经过 FEC 最终校准的 model scores。

## 5. 安装

在线：

```bash
python -m pip install -r requirements.txt
python -m pip install .
```

离线 discovery：`third_party/` 中保留已核验的 PySINDy / derivative wheel，并提供 `scripts/install_offline_discovery_*.sh|ps1`。

## 6. 文档入口

从 `docs/README.md` 开始。第一次使用建议依次读：

1. `docs/getting_started/README_FIRST_CN.md`
2. `docs/getting_started/QUICKSTART_10MIN_CN.md`
3. `docs/getting_started/USABILITY_HARDENING_v2.1_CN.md`
4. `docs/design/BIAS_AND_REPRESENTATION_POLICY_CN.md`
5. `docs/design/P1_P5_END_TO_END_CN.md`

研究/NKT/采用报告位于 `reports/`，不属于日常使用入口。

## 7. 发布验证

```bash
python -m compileall -q src scripts tests
python -m pytest -q
python scripts/check_release_archive_hygiene.py .
```

最终 ZIP 还应重新解压、隔离安装 wheel、执行核心 CLI 与 bundle smoke 后再交付。

## 8. 重要边界

M2N v2.3 不是：

- 自然语言到任意设备完整物理模型；
- 全局 NAS oracle；
- DOR fault ontology / root-cause reasoner / RUL；
- “高预测精度 = 参数物理可解释”的自动保证。

核心原则保持不变：**已知硬机理优先；有限 K2 优先；只有证据支持时才开放局部 K3；诊断模型只在显式任务与独立验证合同下竞争；数据不足时允许 AMBIGUOUS/PORTFOLIO/UNSUPPORTED。**
