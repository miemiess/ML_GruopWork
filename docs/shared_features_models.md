# 公共特征与模型调用

在项目根目录运行合成窗口示例，可以看到分类预测、两列回归预测及两个基线的输出。示例只用于验证调用流程。

```powershell
.\.venv\Scripts\python.exe demo/shared_features_models.py
```

## 从数据窗口到特征矩阵

先用数据模块按用户划分训练、验证和测试集，再分窗。三个模块的 `batch.signal` 均为 `(窗口数, 采样点数, 通道数)`。

```python
from emginsight.features import extract_time_features

X = extract_time_features(batch.signal)
```

每个窗口每个通道计算均方根 RMS、平均绝对值 MAV，以及窗口内相邻样本差的绝对值之和 WL。输出形状为 `(窗口数, 3 × 通道数)`，列顺序为 `通道1 RMS, MAV, WL, 通道2 RMS, MAV, WL, ...`。WL 随窗口长度变化，同一实验应统一采样配置、窗口点数及通道顺序。特征函数按输入信号计算，滤波等预处理由各实验在提取前配置。

| 任务 | 训练目标 | 接入方式 |
| --- | --- | --- |
| C 手势、E 疲劳 | `batch.labels`，形状 `(窗口数,)` | 每组按相同顺序拼接各记录的特征和标签，用 `KNNClassifier`、`LogisticRegression` 对比 `MajorityClassifier`。 |
| D 力量 | `batch.targets`，形状 `(窗口数, 输出数)` | 遍历 `iter_force_windows(record)`，逐批提取特征，只保存特征和选定的实测力量目标。记录 `batch.target_names` 的列顺序，用 `RidgeRegressor` 对比 `MeanRegressor`。 |

疲劳任务若复用已有单路实验，继续使用该实验的特征定义和配置。公共 WL 是相邻差绝对值之和，单路实验用的是相邻差绝对值的均值并取对数。

## 训练和预测

以下以分类为例。`X_train`、`X_val`、`X_test` 是对应用户组的特征矩阵，`y_train` 是训练组标签。

```python
from emginsight.models import LogisticRegression, MajorityClassifier, Standardizer

scaler = Standardizer().fit(X_train)
X_train_scaled = scaler.transform(X_train)
model = LogisticRegression(l2=0.01).fit(X_train_scaled, y_train)
baseline = MajorityClassifier().fit(X_train_scaled, y_train)

val_predictions = model.predict(scaler.transform(X_val))
# 用验证结果选定参数后，再评估保留的测试用户。
X_test_scaled = scaler.transform(X_test)
predictions = model.predict(X_test_scaled)
baseline_predictions = baseline.predict(X_test_scaled)
```

分类预测返回原标签编码。逻辑回归的 `predict_proba` 按 `model.classes_` 排列各类概率，`converged_` 和 `n_iter_` 分别记录是否达到收敛阈值及迭代次数。力量实验使用相同的标准化流程，将模型换成 `RidgeRegressor(alpha=1.0)` 和 `MeanRegressor()`，两者保留目标列数和原单位。
