# 公共 Python 包

以可编辑方式安装后，通过 `emginsight` 导入可复用代码。计划中的子包：

- `datasets/`：UCI、putEMG-Force 与 Zenodo 的独立适配器。
- `preprocessing/`：按记录滤波、分窗及仅在训练集上拟合的变换。
- `features/`：RMS、MAV、WL 及有文档说明的扩展特征。
- `models/`：使用 NumPy 自行实现分类与回归算法。
- `evaluation/`：分组划分、任务指标及对比工具。

参见[公共接口](../../docs/research_protocol.md)。这些目录目前仅包含包占位文件，尚无可运行算法。
