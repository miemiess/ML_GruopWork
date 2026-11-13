# UCI Gesture

统一调用入口：`emginsight.uci_gesture`。下面的例子从仓库根目录运行，使用已安装本项目的 Python 环境。

## 功能结构

```text
uci_gesture/
  __init__.py       对外导出函数与数据类型
  reading.py        单文件读取与格式检查
  splitting.py      按用户划分训练、验证、测试集
  windowing.py      单条记录分窗
  example.py        真实数据检查示例入口
  README.md         调用说明
```

原始数据仍放在仓库根目录的 `data/raw/`，中间数据放 `data/interim/`，处理结果放 `data/processed/`。
数据来源和下载注意事项见[数据说明](../../../docs/datasets.md)。

旧的 `emginsight.datasets` 和 `emginsight.preprocessing` 中的 UCI 接口已经迁移到这里，请统一使用新入口。

## 读取一个 UCI 手势文件

在仓库根目录的 Python 或 Notebook 中运行（使用安装了本项目的环境）：

```python
from emginsight.uci_gesture import load_uci_gesture_record

record = load_uci_gesture_record(
    "data/raw/EMG_data_for_gestures-master/05/2_raw_data_10-29_30.03.16.txt"
)
print(record.signal.shape)  # (采样点数, 8)，8 列为 channel1..channel8
print(record.time[:5])     # 原始时间，单位毫秒
print(record.labels[:5])   # 整数标签；0 为未标注，1 为静息
print(record.subject_id, record.record_id)
```

用户编号默认取父目录名（例如 `05`），记录编号取不含扩展名的文件名。
文件移出用户目录时需显式指定 `subject_id="05"`。读取函数保留所有行及原始数值；
遇到列格式错误、非有限值、非法标签或倒序时间戳会报错。重复时间戳允许，不能直接据此假设均匀采样率。
本步骤只读取原始数据；用户划分和分窗见下文。

本地数据检查：72 份记录中 71 份可通过格式检查。`34/1_raw_data_10-51_07.04.16.txt`
第 79766 行（末行）只有 9 列，缺少 class 标签，当前读取会报错。
后续需核对下载完整性和来源，再确定清洗策略；原始文件保持原样。

## 按用户划分训练、验证、测试集

可以先仅查看用户划分，不加载信号，也不会受缺列文件影响：

```python
from pathlib import Path
from emginsight.uci_gesture import split_subject_ids

root = Path("data/raw/EMG_data_for_gestures-master")
users = [folder.name for folder in root.iterdir()
         if folder.is_dir() and folder.name.isdigit()]
subject_groups = split_subject_ids(users, ratios=(0.6, 0.2, 0.2), seed=42)
for name, subjects in subject_groups.items():
    print(name, len(subjects), subjects)
```

如果已经读取了多名用户的记录，`records` 为这些 `GestureRecord` 对象的列表：

```python
from emginsight.uci_gesture import split_records_by_subject

groups = split_records_by_subject(records, ratios=(0.6, 0.2, 0.2), seed=42)
train_records = groups["train"]
validation_records = groups["validation"]
test_records = groups["test"]
```

同一用户的所有记录只进入同一组；组内保留输入记录顺序，原始数组不被修改。
比例按不同用户数量计算，不代表采样点或类别数量比例，也不保证类别均衡。
先按比例向下取整，剩余名额按小数部分从大到小分配；相同时优先训练、验证、测试。
36 名用户的默认结果为 22/7/7 人。固定用户集合、比例和种子可复现；实验中仍需保存实际用户名单。
至少需要三名用户；若比例取整产生空组会报错，需要调整比例或增加用户。
划分后分别分窗，标准化只在训练集拟合，验证集用于调参，测试集用于最终评估。

## 分窗与外部调用

先按用户划分，再对每组中的每条记录独立分窗。不要先把不同记录拼接起来。

```python
from emginsight.uci_gesture import window_gesture_record

batch = window_gesture_record(record, window_size=200, step_size=100, max_gap_ms=20)
print(batch.signal.shape)      # (窗口数, 200, 8)：B 的特征函数以此作为输入
print(batch.labels.shape)      # (窗口数,)：C 的分类目标
print(batch.time.shape)        # (窗口数, 200)：原始毫秒时间戳
print(batch.subject_ids)       # 每个窗口的用户编号
print(batch.record_ids)        # 每个窗口的记录编号
print(batch.start_indices)     # 每个窗口在原始记录中的起点索引
print(batch.dropped)           # 被丢弃的完整候选窗数量，按原因统计
```

当前默认参数是实验起点：200 点窗口、100 点步长，而不是已验证最优参数。
本地用户 05 的第二条记录平均约 965 行/秒，时间间隔为 1 到 9 毫秒，并非严格均匀采样。
该统计不能确认硬件采样率或独立传感器更新率，因此不声称 200 点精确等于 200 毫秒；
如果实验需要固定物理时长，应先确认采集说明，再实现重采样或时间窗。

标签与边界规则：

- 仅保留标签全部相同且属于 1..7 的窗口；静息类 1 保留。
- 含任意未标注 0 的窗口丢弃，混合手势窗口也丢弃，不做多数票标签。
- 窗口内相邻时间差超过 20 毫秒时丢弃，该阈值可配置。
- 候选起点为 0、step_size、2*step_size……；末尾不足完整窗口时丢弃，不填充。
- 每次只输入一条记录，窗口不会跨记录；相同标签的重复时间戳允许。
- 每个候选窗只计一种丢弃原因，优先级：time_gap → unmarked → mixed_labels。
- 没有有效窗口时返回形状正确的空数组；调用方应检查数量。

### 可直接运行的检查示例

从仓库根目录运行：

```powershell
.venv\Scripts\python.exe -m emginsight.uci_gesture.example
```

示例先固定三组用户名单，再逐条读取和分窗，检查信号形状、标签数量、来源信息以及用户互斥，
默认用中文输出各组人数、用户名单、窗口形状、标签分布、丢弃原因和被拒绝的文件，附字段含义。
它不保存数据或执行模型训练。需要机器可读的 JSON 报告时加 `--json`。
默认遇到异常记录时，立即在终端报告用户、文件、出错行（可定位时）及原因，跳过该文件后继续处理。
异常还会汇总到报告的异常记录区（JSON 模式为 `rejected_records`）；同一用户的其他正常记录继续使用，不整用户排除。
不自动补标签或修改原始数据。需要遇错停止时加 `--strict`；旧的 `--skip-invalid` 命令仍可用。
全局参数错误、没有数据或某一集合无可用窗口时仍会报错，避免产生无效的实验输入。
即使部分记录被拒绝，所属用户也不被重新分组。报告中的 `subjects_with_windows` 可用于核对实际覆盖用户。
可用 `--window-size`、`--step-size`、`--max-gap-ms` 和 `--seed` 调整参数。
B 可从 `batch.signal` 提取特征，C 使用特征和 `batch.labels`，并保留用户/记录编号追溯来源。
共享特征函数尚未实现；标准化需在特征提取后仅用训练集拟合。

### 本地验证记录（2026-11-13）

以 seed=42、200 点窗口、100 点步长、20 毫秒断点阈值运行示例，默认报告并跳过异常文件。全部 42 项自动测试通过：

| 集合 | 用户数 | 成功读取记录数 | 窗口数 | 单个窗口形状 |
| --- | --- | --- | --- | --- |
| 训练 | 22 | 43 | 7890 | (200, 8) |
| 验证 | 7 | 14 | 2860 | (200, 8) |
| 测试 | 7 | 14 | 2457 | (200, 8) |

三组窗口用户无交集，窗口与标签数量相等；每个用户至少有一条记录产出窗口。
被拒绝的唯一记录为上文用户 34 的缺列文件。共丢弃 28259 个含未标注标签的候选窗、
7 个含时间断点的候选窗；混合标签丢弃数为 0（含 0 的窗口已优先计入 unmarked）。
完整名单、标签计数和参数见[验证报告](../../../results/uci_gesture/windowing_check.json)。
这份划分中标签 7 只有训练集有 121 个窗口，验证/测试集没有该类；
不能据此声称已评价未见用户的第 7 类识别效果。实验组需明确报告类别覆盖，
如调整为六类任务或修改划分，应事先约定并更新实验记录，不能按测试性能选划分。

## 运行测试

在仓库根目录执行：

```powershell
.venv\Scripts\python.exe -m pytest tests/uci_gesture -v
```

测试统一存放在仓库根目录，按工程和具体测试项组织：

```text
tests/
  uci_gesture/
    test_reading.py      读取与格式检查
    test_splitting.py    用户划分
    test_windowing.py    分窗、标签和记录边界
    test_example.py      示例输出与异常处理
```

测试通过 `from emginsight.uci_gesture import ...` 调用公开接口，不直接导入内部文件。
全部工程的测试可用 `python -m pytest` 运行；VS Code 统一从根目录 `tests/` 发现测试，刷新测试面板即可。
