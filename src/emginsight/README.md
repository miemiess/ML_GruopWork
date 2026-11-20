# emginsight 包结构

公共层只放与数据集无关、至少两个任务都在用的代码；任务包只放数据集专用的读取、标签和实验流程。任务包之间不互相引用。

```text
emginsight/
  common/          公共基础设施（任何任务都可以引用）
    splitting.py     按用户划分：split_subject_ids、split_records_by_subject
    signal.py        因果 Butterworth 带通 bandpass
    artifacts.py     模型文件保存/读取，SHA-256 与 scikit-learn 版本校验
    batch.py         预测命令的批处理与 summary.json
    plotting.py      报告图的中文字体设置
  features/        公共时域特征：RMS、MAV、WL
  models/          公共模型：kNN、逻辑回归、岭回归、多数类/均值基线、标准化
  learning_curve.py  跨任务分析：训练用户数对效果的影响

  gesture/         UCI 手势（C）
  force/           putEMG-Force 指力（D）
  fatigue/         Zenodo 自感疲劳（E）
```

## 任务包的统一分层

三个任务包按同样的顺序组织，同名文件职责相同：

| 层 | 文件 | 作用 |
| --- | --- | --- |
| 数据 | `reading.py` → `windowing.py` | 读取并检查单个原始文件；按记录分窗 |
| 检查 | `check.py` | 在真实数据上跑一遍读取、划分和分窗，输出中文或 JSON 报告 |
| 实验 | `experiment.py`（手势）、`regression.py` / `levels.py` / `change_experiment.py`（指力）、`single_channel.py` / `full_dataset.py`（疲劳） | 只用训练用户拟合、验证用户选型、测试用户评估一次，写出 `results/<任务>/` 报告并保存模型 |
| 模型 | `model.py`（指力的跨日模型在 `change.py`） | 已训练模型的数据类、保存与读取 |
| 部署 | `predict.py`（指力另有 `change_predict.py`） | 读取原始文件批量预测，供 `deploy/` 中的 Docker 镜像调用 |

任务内部共用但与数据集相关的代码留在任务包里，例如指力的 `preprocessing.py`（滤波、静息校准、特征缓存，回归、三档、跨日变化和学习曲线共用）、`targets.py`（力量目标、三档标签与回归指标），以及手势的 `rejection.py`（“未知”拒识）。

## 公共接口用法

先按用户划分，再分窗、提取特征，标准化只在训练集上拟合：

```python
from emginsight.common import split_records_by_subject
from emginsight.features import extract_time_features
from emginsight.models import LogisticRegression, MajorityClassifier, Standardizer

groups = split_records_by_subject(records)          # 每条记录需有 subject_id；同一用户只进一组
X_train = extract_time_features(train_windows)      # (窗口数, 采样点数, 通道数) → (窗口数, 3 × 通道数)
scaler = Standardizer().fit(X_train)
model = LogisticRegression(l2=0.01).fit(scaler.transform(X_train), y_train)
baseline = MajorityClassifier().fit(X_train, y_train)
```

- `split_subject_ids` 默认比例 0.6/0.2/0.2；恰好三名用户时自动每组一人；比例取整出空集合时报错，不会退化为按窗口随机划分。
- `extract_time_features` 的列顺序为“通道 1 的 RMS、MAV、WL，通道 2 的 RMS、MAV、WL……”。WL 是相邻差绝对值之和，随窗口长度变化。疲劳单路实验使用自己的对数幅值与频谱特征，不使用这一函数。
- 模型统一提供 `fit(X, y)` 和 `predict(X)`。逻辑回归另有 `predict_proba`，类别顺序见 `classes_`。回归模型 `RidgeRegressor`、`MeanRegressor` 支持多列目标。

| 类型 | 模型 | 用于 |
| --- | --- | --- |
| 分类 | `KNNClassifier`、`LogisticRegression` | 手势、疲劳、指力三档 |
| 分类基线 | `MajorityClassifier` | 手势、疲劳、指力三档 |
| 回归 | `RidgeRegressor` | 指力 |
| 回归基线 | `MeanRegressor` | 指力 |

随机森林等直接使用 scikit-learn。保存模型统一调用 `common.artifacts.save_model_file`，读取时用 `load_model_file` 核对校验值与库版本。
