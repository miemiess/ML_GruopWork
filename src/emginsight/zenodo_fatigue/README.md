# Zenodo 自感疲劳数据处理

统一调用入口：`emginsight.zenodo_fatigue`。代码与说明按项目组织，测试位于 `tests/zenodo_fatigue/`。

```text
src/emginsight/zenodo_fatigue/
  reading.py       发现配对文件、读取肌电和评分、记录有限的末尾修正
  alignment.py     将独立评分按秒时间轴对齐肌电
  splitting.py     按用户划分训练/验证/测试
  windowing.py     按点数分窗与窗口标签规则
  example.py       全数据检查与中文/JSON 报告
  __init__.py      外部调用入口
  README.md       本说明
```

## 数据布局与来源

```text
data/raw/EMG_data_for_ZenodoTired/
  sEMG_data/subject_1/trial_1.csv
  self_perceived_fatigue_index/subject_1/Trial_1.csv
  ...                         # 共 13 位参与者，每人 12 次试验
  protocol.xlsx
  Metadata.xlsx
  code.ipynb
```

来源：[Zenodo v2](https://zenodo.org/records/14182446)，使用时保留来源引用与 CC BY 4.0 署名。
读取已解压的 CSV，忽略 ZIP、MVC 校准目录和作者 Notebook，不运行或改写作者代码。
肌电和评分按**用户编号＋试验编号**配对，忽略文件名大小写；不能依靠两个文件列表的排序位置配对。
缺评分或重复身份会明确报告，不能随便配给相邻文件。

## 1. 读取一条记录

```python
from emginsight.zenodo_fatigue import load_fatigue_record

record = load_fatigue_record(
    "data/raw/EMG_data_for_ZenodoTired/sEMG_data/subject_1/trial_1.csv",
    "data/raw/EMG_data_for_ZenodoTired/self_perceived_fatigue_index/subject_1/Trial_1.csv",
)
print(record.signal.shape)   # (采样点数, 1)，默认主动力肌
print(record.channel_names, record.main_muscle)
print(record.time[:5])       # 肌电秒时间轴
print(record.rating_time[:5], record.ratings[:5])  # 独立评分，尚未逐点对齐
print(record.subject_id, record.trial_id, record.repairs)
```

数据集覆盖双臂八块肌肉，但每份试验 CSV 通常仅有**单侧四路肌电与各自的时间列**。
默认 `channel_mode="agonist"` 按本地 `protocol.xlsx` 选择该试验主动力肌，和作者 Notebook 的主要分析思路一致，避免按固定列号误选。
映射：1/2、9/10 为右/左三角肌前束，3/4、11/12 为右/左后束，5/6 为右/左肱二头肌，7/8 为右/左三角肌中束。
`channel_mode="all"` 可保留当前文件内所有原始 EMG 列；不读作者附加的 RMS 滤波列。
多通道时要求所选通道时间同步，否则报错，不自动插值。
不同试验可能来自不同肌肉/侧别，使用全部通道训练时必须保留并统一通道语义；不能把四列按位置当作同四块肌肉。

信号按原始 `[V]` 数值保留，时间单位为秒，不自动滤波、标准化、MVC 归一化或时间平移。
这是一版可靠的数据准备接口，不代表已完成分类实验或全部生理预处理。

## 2. 时间对齐

```python
from emginsight.zenodo_fatigue import align_fatigue_labels

labels = align_fatigue_labels(record, max_label_age=0.04)
print(labels.shape)  # 与肌电采样点数相同
```

协议给出的肌电采样率为 1259 Hz，评分为 50 Hz，二者不能逐行对应。
本模块使用共同的原始秒时间轴，在每个肌电时刻取**最近一次已发生的评分**：

- 不从未来取标签，不对离散的 0/1/2 做线性插值。
- 不在评分起点之前或终点之后外推。
- 距上一条评分超过 0.04 秒时标为 `-1`，避免跨缺失区间无限填充。
- **0/1/2 都是有效的主观疲劳等级，0 不是 UCI 的未标注值。** 只有 `-1` 表示无有效评分。
- 当前假设文件内秒时间轴采用共同时间基准，未估计评分反应延迟或额外设备偏移；若后续发现不同步，应先确认采集说明，不能只按总时长拉伸配对。

## 3. 用户划分与分窗

多用户记录可调用 `split_fatigue_records_by_subject(records, seed=42)`，比例默认 0.6/0.2/0.2。
复用统一分组算法：同一用户全部试验必须同组，不按行或窗口随机划分。
完整数据的示例先根据文件身份分配用户，再逐条读取和分窗，避免同时加载全部数据。

```python
from emginsight.zenodo_fatigue import window_fatigue_record

batch = window_fatigue_record(record, window_size=5036, step_size=2518)
print(batch.signal.shape)      # (窗口数, 5036, 所选通道数)
print(batch.labels.shape)      # (窗口数,)，B/E 可接特征与分类模块
print(batch.time.shape)        # (窗口数, 5036)，原始秒时间戳
print(batch.subject_ids, batch.trial_ids, batch.record_ids)
print(batch.channel_names, batch.start_indices, batch.dropped)
```

默认窗口 5036 点、步长 2518 点，按名义 1259 Hz 约为 4 秒/2 秒，呼应作者频率分析示例；
并非声称此参数最优，实际时长仍应核对时间戳。
每窗评分必须全部相同且为 0/1/2；跨评分变化、评分未覆盖/过期、肌电内部时间间隔超过 0.01 秒的窗口丢弃。
丢弃原因只计一项，优先顺序为时间断点、评分未覆盖、混合等级。
窗口不跨记录、不填充不足一窗的尾部；无有效窗时返回形状正确的空数组。
`window_fatigue_record` 会自动调用时间对齐，无需调用方先覆盖记录的评分。

后续可从 `batch.signal` 提取 RMS/MAV/WL 或频域特征，以 `batch.labels` 为分类目标。
特征标准化只在训练集拟合，验证集调参，最后才评估测试集。主观标签不能当作临床客观疲劳诊断。

## 4. 已知异常和修正边界

原始文件始终不修改，默认在内存中有限处理以下尾部导出异常：

1. 删除连续的全零尾行；或删除末尾某个所选肌肉的“时间/信号”都为零、而此前时间已大于零的行。
2. 评分末行时间倒退/重复时，仅在它与前一行评分相同的情况下删去该尾行。

修正数量保存在 `record.repairs` 和报告的每条记录中，不将内部倒序、不同等级尾行、非有限数据或不同步通道偷偷修好。
可用 `repair_terminal=False` 禁用修正；示例对应 `--no-terminal-repair`，便于核对原始问题。
这不是随意排序时间或补出缺失标签，修改规则仍需在最终报告中披露。

## 5. 运行示例与测试

在仓库根目录运行：

```powershell
.venv\Scripts\python.exe -m emginsight.zenodo_fatigue.example
.venv\Scripts\python.exe -m pytest tests/zenodo_fatigue -v
```

默认中文报告。加 `--json` 得到机器可读报告；可设置 `--data-dir`、`--window-size`、`--step-size`、
`--max-label-age`、`--max-gap-seconds`、`--channel-mode`、`--seed`。
异常文件默认报告后继续，`--strict` 改为遇错停止；没有数据或某组无有效窗口时明确报错。
测试位于根目录统一的 `tests/zenodo_fatigue/`，通过公开接口验证，不需下载数据即可运行。

实际检查明细见[验证报告](../../../results/zenodo_fatigue/processing_check.json)。

## 本地验证（2026-11-13）

已配对并读取全部 156 份试验，来自 13 名用户。默认主动力肌、seed=42，结果如下：

| 集合 | 用户数 | 记录数 | 窗口数 | 0/1/2 级窗口数 |
| --- | --- | --- | --- | --- |
| 训练 | 8 | 96 | 13732 | 5335 / 4442 / 3955 |
| 验证 | 3 | 36 | 6738 | 1825 / 2621 / 2292 |
| 测试 | 2 | 24 | 2540 | 527 / 903 / 1110 |

合计 **23010 个** `(5036, 1)` 窗口，三组用户无交集，各组均覆盖三个评分等级。
评分未覆盖/过期丢弃 154 个候选窗，混合等级丢弃 608 个候选窗，时间断点丢弃 0 个。
44 份记录发生有限尾部修正：肌电补零尾行共 46 行，同评分倒退/重复尾行共 8 行。
异常拒绝记录 0 份；修正前的问题和规则必须保留在实验说明中。
疲劳模块新增 22 项测试，与前两个模块合计 **92 项通过**。
这些统计证明读取、对齐与分窗流程可运行，不代表模型训练或临床有效性。
