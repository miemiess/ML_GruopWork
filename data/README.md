# 数据怎么放

先下载少量样本确认能读取，再考虑完整下载。原始数据放 `raw/<数据集名>/`，中间文件放 `interim/`，处理结果放 `processed/`；这些数据不会上传 Git。

| 数据 | 下载与读取注意事项 |
| --- | --- |
| [UCI 手势](https://archive.ics.uci.edu/dataset/481/emg+data) | 保留原始参与者目录，读取 8 通道、时间和类别。0 是未标注，1 是静息，第 7 类并非人人都有。许可为 CC BY 4.0，保留来源署名。 |
| [putEMG-Force](https://biolab.put.poznan.pl/putemg-dataset/) | 选择 emg_force，可使用[官方下载工具](https://github.com/biolab-put/putemg-downloader)。先选 CSV 或 HDF5 一种格式；实测力量是 FORCE_*，TRAJ_* 是目标轨迹。确认单位、校准、实际可用用户及许可。 |
| [Zenodo 疲劳 v2](https://zenodo.org/records/14182446) | 获取肌电、评分和采集说明；肌电压缩包约 3.3 GB。说明中的肌电/评分频率为 1259/50 Hz，按时间戳对齐，检查缺失标签和许可。 |

负责读取的同学在自己的实验记录中写清：来源版本、实际使用文件、通道与单位、标签含义、排除情况、用户划分，以及数据如何传给后续函数。时间对齐不成功或目标单位不明时先和组员讨论。

数据仍需在本地核对。不要强行上传被忽略的数据文件，也不要公开参与者的敏感信息。
