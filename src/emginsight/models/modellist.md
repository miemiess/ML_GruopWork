# 可用模型

从 `emginsight.models` 导入。模型均支持 `fit(X, y)` 和 `predict(X)`，其中 `X` 是每个肌电窗口提取的特征矩阵。

| 类型 | 模型 | 概述 | 对应任务 |
| --- | --- | --- | --- |
| 分类 | `KNNClassifier`（kNN） | 找到最相似的 `k` 个训练窗口，由它们投票确定类别。 | C 手势、E 疲劳 |
| 分类 | `LogisticRegression`（逻辑回归） | 根据窗口特征预测类别，支持用 `predict_proba()` 取各类概率。 | C 手势、E 疲劳 |
| 分类基线 | `MajorityClassifier`（多数类） | 每次都预测训练集里出现最多的类别。 | C 手势、E 疲劳 |
| 回归 | `RidgeRegressor`（岭回归） | 用带 L2 正则化的线性模型预测连续力量，可同时预测多个力量目标。 | D 力量 |
| 回归基线 | `MeanRegressor`（均值） | 每次都预测各力量目标在训练集中的均值。 | D 力量 |

`Standardizer` 用于特征标准化。只在训练集上调用 `fit(X_train)`，然后用同一个实例转换各组数据。

运行示例和输入输出形状见[公共特征与模型调用](../../../docs/shared_features_models.md)。
