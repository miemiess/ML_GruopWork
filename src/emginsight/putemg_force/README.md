# putEMG-Force 数据处理

公开调用入口：`emginsight.putemg_force`。从仓库根目录使用项目 `.venv` 运行。

```text
src/emginsight/putemg_force/
  __init__.py      对外导出接口
  reading.py       CSV 读取、数据形状检查和来源信息
  splitting.py     按用户划分，排除校准记录
  windowing.py     分批分窗与窗口回归目标
  example.py       可运行的中文检查报告
  README.md       本文档

tests/putemg_force/
  conftest.py      合成测试样本
  test_reading.py
  test_splitting.py
  test_windowing.py
  test_example.py
```

## 数据放置

```text
data/raw/EMG_data_for_putemgforce/
  emg_force_03/
    emg_force-03-bias-2018-05-11-11-20-41-581.csv
    emg_force-03-mvc-2018-05-11-11-20-54-280.csv
    emg_force-03-repeats_long-2018-05-11-11-22-53-467.csv
  emg_force_04/    # 两次采集的 bias、mvc、repeats_long、repeats_short、sequential
  emg_force_05/    # 两次采集的相同五类记录
  putEMG-participants.csv
  records.txt
```

请先解压 ZIP；程序只发现 `emg_force-*.csv`，不重复读取压缩包，不将参与者背景表当作信号。
数据不上传 Git，目录中的 `.gitkeep` 用于保留空目录结构。
来源：[作者下载器](https://github.com/biolab-put/putemg-downloader)、[大学云盘](https://chmura.put.poznan.pl/s/45NY5snj0U4tgQz)。

## 1. 读取一份记录

```python
from emginsight.putemg_force import load_putemg_force_record

record = load_putemg_force_record(
    "data/raw/EMG_data_for_putemgforce/emg_force_03/"
    "emg_force-03-repeats_long-2018-05-11-11-22-53-467.csv"
)
print(record.signal.shape)   # (1819136, 24)，EMG_1..EMG_24
print(record.force.shape)    # (1819136, 10)，FORCE_1..FORCE_10
print(record.time[:5])       # 原始秒时间轴，可以包含负的记录起始时间
print(record.subject_id, record.session_id, record.trial_type)
print(record.target_names, record.force_unit)
```

- 肌电、时间与力量取自同一 CSV 行，保持同步，不另加插值、时间偏移或重采样。
- 作者 CSV 的首列是未命名时间索引，`pandas` 读入后为 `Unnamed: 0`；程序也接受明确命名的 `Timestamp`、`TIMESTAMP`、`time`。
- 逐项检查 24 路肌电、选定力量目标、有限数值和严格递增的时间戳，不静默删除坏行或把 NaN 改成零。
- `subject_id` 从文件名提取；`session_id` 使用采集日期，`record_id` 为完整文件主名。日期级会话划分需与实际实验安排核对。
- **力量物理单位与通道对应手指关系尚待核对**。当前保留原始值和原始列名，不能将 MAE/RMSE 标为牛顿，也不自行声称十列等于十个独立手指。
- `TRAJ_*` 是指令轨迹，不作为实测目标；`FORCE_MVC` 不属于默认的十路力量目标，也不直接当作归一化系数。
- 保留原始肌电数值、力量负值和基线偏移，不自动滤波、裁剪、减 bias 或按 MVC 归一化。没有确认单位和校准语义前，不把此版本称为已完成物理校准。

只预测部分目标时可以指定列名与顺序，例如：

```python
record = load_putemg_force_record("你的记录.csv", targets=("FORCE_1", "FORCE_2"))
```

## 2. 按用户划分

已经读取多名用户的实验记录时：

```python
from emginsight.putemg_force import split_force_records_by_subject

groups = split_force_records_by_subject(records, seed=42)
train_records = groups["train"]
validation_records = groups["validation"]
test_records = groups["test"]
```

以用户数而非采样点数计数；同一用户的全部会话与记录同组。
省略 `ratios` 时：恰好三名用户自动各分配一人，其他人数默认采用 0.6/0.2/0.2。
这是为了让当前三用户流程可以完整检查；不是按模型表现挑选划分。
显式传入 `ratios` 时严格遵守指定比例，若取整产生空组则明确报错，不偷偷更换比例。
`bias`、`mvc` 被排除，不用于训练样本。保留组内输入顺序。
为避免复制相同算法，内部复用 UCI 已测试的用户分组算法；力量模块的 `split_subject_ids` 对三用户默认比例做明确适配，记录分配由本模块实现。
相同用户集合、比例和种子可复现，仍需在实验报告保存名单。

目前本地包含用户 03、04、05，默认种子 42 得到 **训练 05、验证 04、测试 03**，每组一人。
这是跨用户流程的小样本检查，不足以稳定评估泛化能力。不会改成按行或按窗随机划分。
至少需要三名用户且比例不能产生空组；扩展用户后需保存新的名单并重新记录实验配置。
大量数据应先使用 `split_subject_ids` 对文件名中的实验用户分组，再逐条读取和提取特征，避免同时驻留全部记录。

## 3. 分批分窗，交给特征和回归模块

```python
from emginsight.putemg_force import iter_force_windows

for batch in iter_force_windows(record, window_size=1024, step_size=512, batch_size=128):
    print(batch.signal.shape)    # (本批窗口数, 1024, 24)，B 可提取 RMS/MAV/WL
    print(batch.targets.shape)   # (本批窗口数, 目标数)，D 的多输出回归目标
    print(batch.target_names)    # 与 targets 的列逐一对应
    print(batch.time.shape)      # (本批窗口数, 1024)，原始秒时间戳
    print(batch.subject_ids, batch.session_ids, batch.record_ids)
    print(batch.start_indices)   # 在原始记录中的起点索引
    # 逐批提取特征，再收集小的特征矩阵；不要把全部 batch 存进列表。
```

- `iter_force_windows` 返回迭代器，使用 `for` 逐批处理；不足一窗时没有任何批次。
- 默认 1024 点窗口、512 点步长，对名义 5120 Hz 约为 200/100 毫秒，但应以实际时间戳核对。
  每份记录的时间戳估计行频率见运行报告，不作为独立硬件采样率测量。
- 每窗目标为同一窗口内各 `FORCE_*` 的**算术平均值**，保留多输出形状；不做手势标签投票。
- 一次输入一条记录，不跨记录、不跨用户、不拼接不足窗的尾部。
- 窗口内部任意相邻时间差超过 `max_gap_seconds=0.01` 秒时丢弃，阈值可配置。
- `bias` 与 `mvc` 输入分窗函数会报错，示例中只读取检查它们。它们与实验文件的量纲/偏移差异不能靠盲目相减解决。
- 先按用户分组，再分别分窗；特征标准化仅用训练集拟合，测试用户的 MVC 或其他校准数据也不能默认用于预处理参数拟合。

## 4. 直接运行示例和测试

```powershell
.venv\Scripts\python.exe -m emginsight.putemg_force.example
.venv\Scripts\python.exe -m pytest tests/putemg_force -v
```

示例打印中文报告；`--json` 切换机器可读报告。可设置 `--data-dir`、`--window-size`、`--step-size`、
`--batch-size`、`--max-gap-seconds`、`--seed` 和 `--ratios TRAIN VALIDATION TEST`。
默认报告并跳过异常文件，`--strict` 可改为遇错停止。
中文与 JSON 报告均包含实际划分策略、各组实验记录数、窗口数与实验类型覆盖。
没有实验窗口时明确报错。用户不足时仅展示读取和分窗，不报告模型成绩或虚构三组结果。

测试通过 `emginsight.putemg_force` 的公开入口调用，位于根目录 `tests/putemg_force/`。
pytest 使用 importlib 模式，支持不同工程下重复使用 `test_reading.py` 等文件名，VS Code 仍从统一的 `tests/` 发现测试。

## 当前验证结果

2026-11-13 新增用户 04、05 后，本地共 **23 份 CSV，全部可读取**：10 份校准记录、13 份实验记录。
默认 1024 点窗口、512 点步长、seed=42，三名用户各分配一组：

| 集合 | 用户 | 实验记录数 | 窗口数 | 实验类型 |
| --- | --- | --- | --- | --- |
| 训练 | 05 | 6 | 17748 | 两个日期的 repeats_long、repeats_short、sequential |
| 验证 | 04 | 6 | 17720 | 两个日期的 repeats_long、repeats_short、sequential |
| 测试 | 03 | 1 | 3552 | 一个日期的 repeats_long |

共 **39020 个窗口**，三组用户无交集。每窗肌电 `(1024, 24)`、回归目标 `(10,)`，
校准记录不生成实验窗口；当前没有时间断点丢弃或异常文件。
参见[实际运行报告](../../../results/putemg_force/processing_check.json)。

**各组实验覆盖不一致，且每组只有一人**：当前适合检查数据链路，不是充分的模型评估。
正式实验可先统一选择 `trial_type == "repeats_long"` 的记录，再按用户划分；
或者补齐用户 03 的其他类型，并事先明确试验范围。不要根据测试成绩选择类型或用户划分。
三组间记录数与采样点数不必相等，比例按用户数计算。
力量单位、通道物理含义和校准仍需确认，之后与 D 工作负责人对接。
