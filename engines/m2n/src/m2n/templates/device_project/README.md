# M2N 新设备项目模板

按顺序完成：

1. 在 `mechanism_contract.py` 写出状态、输入、局部机理槽位及 K0/K1/K2/K3；
2. 在 `data/README.md` 写清楚正常数据列名、单位、train/val/test 划分；
3. 在 `run_design.py` 的 `load_data()` 和 `evaluator()` 中接入你的正常数据与估计准则；
4. 运行 `python run_design.py`；
5. 若结果为 `SELECTED`，冻结所选架构；若为 `AMBIGUOUS/UNSUPPORTED`，不要自动升级为 K3。

默认模板是一个可运行的二阶机械 K2 示例，先跑通，再替换成真实设备。
