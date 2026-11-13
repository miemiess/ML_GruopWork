# 数据怎么放

先下载少量样本确认能读取，再考虑完整下载。原始数据放 `raw/<数据集名>/`，中间文件放 `interim/`，处理结果放 `processed/`；这些数据不会上传 Git。

| 数据                                                         | 下载与读取注意事项                                                                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| [UCI 手势](https://archive.ics.uci.edu/dataset/481/emg+data) | 保留原始参与者目录，读取 8 通道、时间和类别。0 是未标注，1 是静息，第 7 类并非人人都有。许可为 CC BY 4.0，保留来源署名。 |

人话 该链接download好直接扔到EMG_data_for_gestures-master

| [putEMG-Force](https://biolab.put.poznan.pl/putemg-dataset/) | 选择 emg*force，可使用[官方下载工具](https://github.com/biolab-put/putemg-downloader)。先选 CSV 或 HDF5 一种格式；实测力量是 FORCE*\_，TRAJ\_\_ 是目标轨迹。确认单位、校准、实际可用用户及许可。 |

之前的链接可能失效了，用https://chmura.put.poznan.pl/s/45NY5snj0U4tgQz尝试，目前只做肌电到力量预测，视频和深度图可以先不下载；CSV 和 HDF5 选择一种即可。实验类型和目录名称来自作者的下载脚本。
另外，官方仓库确实有尚未关闭的下载链接失效反馈，所以旧脚本也不能保证直接可用。你先打开上面的云盘，若能看到目录，发截图我可以帮你选具体文件
人话 该链接download好直接扔到EMG_data_for_putemgforce，但是这个数据比较大可以按人来分文件，putEMG-participants.csv和 record.txt是一定要下的有用的

| [Zenodo 疲劳 v2](https://zenodo.org/records/14182446) | 获取肌电、评分和采集说明；肌电压缩包约 3.3 GB。说明中的肌电/评分频率为 1259/50 Hz，按时间戳对齐，检查缺失标签和许可。 |
人话 全部下然后扔进去EMG_data_for_ZenodoTired

负责读取的同学在自己的实验记录中写清：来源版本、实际使用文件、通道与单位、标签含义、排除情况、用户划分，以及数据如何传给后续函数。时间对齐不成功或目标单位不明时先和组员讨论。

## putEMG-Force 本地进度（2026-11-13）

- 存储路径：`data/raw/EMG_data_for_putemgforce/emg_force_03/`、`emg_force_04/`、`emg_force_05/`；保留原始文件名，ZIP 先解压。
- 当前共 23 份信号 CSV：用户 03 有 3 份；用户 04、05 各 10 份，包含各自两次采集的记录。根目录清单的准确文件名为 `records.txt`。
- 默认只对实验记录分窗，10 份 bias/mvc 校准记录不作为训练样本；13 份实验记录共生成 39020 个窗口。
- 三用户默认每组一人，seed=42 对应训练 05、验证 04、测试 03；同一用户的不同日期与实验类型不跨组。
- 用户 03 仅包含 repeats_long，另外两名用户包含三种实验类型。正式评估前需统一记录类型或补齐数据，并扩大用户覆盖。
- [外部调用说明](../src/emginsight/putemg_force/README.md)、[实际验证报告](../results/putemg_force/processing_check.json)。原始数据继续被 Git 忽略。

## Zenodo 疲劳本地进度（2026-11-13）

- 已解压肌电及评分，共 13 名用户、156 份试验。按用户＋试验编号配对，MVC 校准文件不作为试验样本。
- 模块 `emginsight.zenodo_fatigue` 支持读取、独立评分时间对齐、用户划分及纯等级分窗。默认按协议取主动力肌，每窗 5036 点、步长 2518 点。
- 全部试验可读取；默认分组为 8/3/2 人，产出 13732/6738/2540 个窗口，用户不重叠，评分 0/1/2 均保留。
- 处理器在内存中记录并剔除 46 行肌电补零尾行、8 行同评分但时间倒退/重复的评分尾行，涉及 44 份记录，原始文件不变。其他异常不自动修复。
- [调用说明](../src/emginsight/zenodo_fatigue/README.md)、[完整检查报告](../results/zenodo_fatigue/processing_check.json)。时间基准、主观评分解释与通道选择应在最终实验中披露。
