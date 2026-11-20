# putEMG-Force 数据处理

公开调用入口：`emginsight.force`。从仓库根目录使用项目 `.venv` 运行。

```text
src/emginsight/force/
  __init__.py          对外导出：读取、分窗、ForceModel
  reading.py           CSV 读取、数据形状检查和来源信息
  windowing.py         分批分窗与窗口回归目标
  preprocessing.py     各实验共用：滤波、静息校准、分窗特征、按用户划分后的特征缓存（prepare_data）
  targets.py           力量目标、三档标签与回归指标
  check.py             真实数据检查报告（中文 / JSON）
  regression.py        D 任务：跨用户十路力量回归实验
  levels.py            补充实验：整只手三档力量分类（静息 / 轻 / 重）
  change.py            跨日变化：第二阶段模型、动作保持段特征、测力计标定命令
  change_experiment.py 跨日变化的公共数据实验
  change_predict.py    跨日变化的批处理预测
  model.py             连续回归与三档模型的特征变换、因果平滑、保存与读取
  predict.py           批处理预测入口：原始 CSV → 十路力量预测与三档力量

tests/force/           conftest.py 合成样本，以及与上述模块同名的 test_*.py
```

用户划分、滤波、模型文件读写和批处理来自公共层 `emginsight.common`，见[包结构说明](../README.md)。

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
from emginsight.force import load_putemg_force_record

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

```python
from emginsight.common import split_records_by_subject

records = [r for r in records if r.trial_type not in {"bias", "mvc"}]   # 校准记录不参与训练
groups = split_records_by_subject(records, seed=42)
```

以用户数计数，同一用户的全部会话与记录同组。省略 `ratios` 时，恰好三名用户自动各分配一人，其他人数默认 0.6/0.2/0.2；显式比例取整出空组时报错，不会改成按行或按窗随机划分。相同用户集合、比例和种子可复现，实验报告中仍需保存名单。
大量数据应先用 `emginsight.common.split_subject_ids` 对文件名中的实验用户分组（`preprocessing.prepare_data` 就是这样做的），再逐条读取和提取特征，避免同时驻留全部记录。

## 3. 分批分窗，交给特征和回归模块

```python
from emginsight.force import iter_force_windows

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
.venv\Scripts\python.exe -m emginsight.force.check
.venv\Scripts\python.exe -m pytest tests/force -v
```

示例打印中文报告；`--json` 切换机器可读报告。可设置 `--data-dir`、`--window-size`、`--step-size`、
`--batch-size`、`--max-gap-seconds`、`--seed` 和 `--ratios TRAIN VALIDATION TEST`。
默认报告并跳过异常文件，`--strict` 可改为遇错停止。
中文与 JSON 报告均包含实际划分策略、各组实验记录数、窗口数与实验类型覆盖。
没有实验窗口时明确报错。用户不足时仅展示读取和分窗，不报告模型成绩或虚构三组结果。

测试通过 `emginsight.force` 的公开入口调用，位于根目录 `tests/force/`。
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
参见[实际运行报告](../../../results/force/processing_check.json)。

**各组实验覆盖不一致，且每组只有一人**：当前适合检查数据链路，不是充分的模型评估。
正式实验可先统一选择 `trial_type == "repeats_long"` 的记录，再按用户划分；
或者补齐用户 03 的其他类型，并事先明确试验范围。不要根据测试成绩选择类型或用户划分。
三组间记录数与采样点数不必相等，比例按用户数计算。
力量单位、通道物理含义和校准仍需确认，之后与 D 工作负责人对接。

## D 任务力量回归实验

```powershell
.venv\Scripts\python.exe -m emginsight.force.regression
```

只使用全部用户的 `repeats_long` 记录，按用户 0.6/0.2/0.2 划分（44 人时为 26/9/9）。每条记录固定因果 20–500 Hz 带通后分窗；
开头 4 秒（确认 `TRAJ_*` 全为 0）用作静息校准，得到力量零点和静息肌电特征，这段窗口不参与训练与评估。
提取公共 RMS/MAV/WL，比较特征尺度、岭回归、随机森林与因果平滑，按验证用户平均 RMSE 选型，对比训练集均值基线。
特征缓存在 `results/runs/force/`（Git 忽略）。结果、限制和复现说明见[实验报告](../../../results/force/regression/report.md)。
获选模型保存为 `results/runs/force/force_model.joblib`（`--model-path` 可改），用 `load_force_model` 读取，
会核对 SHA-256 与 scikit-learn 版本。文件说明和校验值见[模型卡](../../../results/force/regression/model_card.json)。
滤波、分窗和静息校准由 `preprocessing.py` 为训练与推理共同提供。
用已训练模型预测原始 CSV，运行 `python -m emginsight.force.predict --input 文件或目录 --output-dir 输出目录`。
Docker 打包与输入输出格式见[部署说明](../../../deploy/force/README.md)。
三档力量分类：`python -m emginsight.force.levels`（需先完成回归实验），见[三档报告](../../../results/force/levels/report.md)。

## 个人动作力量变化（两步回归）

公共数据实验依次运行 `regression`、`change_experiment`、`change_predict`。训练好的 `force_change_model.joblib` 可以直接比较两日的 putEMG 原始刻度。[独立用户的验证与测试](../../../results/force/change/report.md) · [Docker 与完整输出](../../../deploy/force_change/README.md)。下文适用于有个人测力计参考值的牛顿标定。

第一步沿用 24 路肌电模型，预测十路指力。第二步把同一次动作保持段内的十路预测分别取中位数，用标准化岭回归拟合同步测力计读数（N）。每天重复动作时，先估算每次动作的力量，再取当天中位数。患者端只显示“复测日中位数 − 基线日中位数”，正号表示这次动作估计用力更大。

### 准备两份清单

把同一人的匿名编号、手侧、动作和测量协议固定下来，例如用 `left-pinch-v1` 表示左手在约定姿势、设备和电极位置下完成的捏力任务。将原始肌电 CSV 和下面两份清单放在本地，例如 `deploy/force/input/personal/`。每份原始 CSV 仍需 24 路肌电、约 5120 Hz 和开头 4 秒静息。

**标定清单 `calibration.csv`** 每行代表一次动作。以下为格式示例，`实测值` 须换成同一保持段内已调零、已标定测力计的牛顿读数中位数。时间从该肌电记录开头算起，两台设备的时间需对齐。putEMG 的原始 FORCE、动作指令和 MVC 百分比均不能填作牛顿参考。

```csv
patient,protocol,date,record,start_s,end_s,reference_n,split
P01,left-pinch-v1,2026-06-01,cal.csv,8,11,实测值,fit
P01,left-pinch-v1,2026-06-01,cal.csv,18,21,实测值,fit
P01,left-pinch-v1,2026-06-02,check1.csv,8,11,实测值,check
P01,left-pinch-v1,2026-06-03,check2.csv,8,11,实测值,check
```

拟合日采集多次动作，覆盖预期的力量范围。随后至少留出两个日期核对跨日变化；留出日与正式比较日使用相同的动作要求、出力指令和重复方案。每次重复各占一行。清单中的时间应位于稳定保持段内，足以覆盖 200 ms 窗及因果平滑历史。

**比较清单 `visits.csv`** 只列基线日和本次复测日，两天都在拟合日之后。程序按日期排序，随访时只需要肌电。

```csv
patient,protocol,date,record,start_s,end_s
P01,left-pinch-v1,2026-06-04,baseline.csv,8,11
P01,left-pinch-v1,2026-06-04,baseline.csv,18,21
P01,left-pinch-v1,2026-06-11,followup.csv,8,11
P01,left-pinch-v1,2026-06-11,followup.csv,18,21
```

### 在项目根目录运行

```powershell
$records = 'deploy/force/input/personal'
$cal = 'deploy/force/output/personal/P01-left-pinch.joblib'
.\.venv\Scripts\python.exe -m emginsight.force.change fit --trials "$records/calibration.csv" --calibration $cal
.\.venv\Scripts\python.exe -m emginsight.force.change change --trials "$records/visits.csv" --calibration $cal
```

`fit` 保存个人模型及同名的 `.card.json` 模型卡。卡片中的 `validation` 给出十路融合、单独标定 FORCE_8、始终预测零变化的误差（N）。每个留出日都与最早的留出日比较。十路的平均绝对误差须低于零变化基线，`change` 才会显示数值。两天留出数据只能提供一次变化对照。

`change` 成功时只输出一行，例如 `本次动作力量较基线的估计变化　+3.2 N`（格式示例）。现有公开数据没有这个个人标定所需的配对牛顿真值。实际用于患者前还需在目标人群验证跨日误差和重复测量波动。一次标准动作的力量变化，不能直接换算成康复百分比。
