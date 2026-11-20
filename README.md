# EMGInsight

**61.501 Production Machine Learning · 第 12 组**

用传统机器学习分析表面肌电（sEMG），完成手势识别、手指力量估计和肌肉疲劳分类。希望为辅助交互、假肢与康复研究提供参考。项目仅做离线研究。

成员：Ma Mingyang、Han Yuchen、Zhou Ying、Wei Xin、Gao Ruiyan、Li Ji。**不固定分工，大家按兴趣和进度选择任务，也可以合作完成。**

三个数据集的读取、用户划分和分窗已完成。包结构与公共接口见[代码说明](src/emginsight/README.md)，实际提交步骤见[贡献指南](CONTRIBUTING.md)。

## 快速开始

| 目标 | 入口 |
| --- | --- |
| 看网页 | [GitHub Pages 已发布版本](https://miemiess.github.io/ML_GruopWork/) · [本地手势与指力演示、导出和发布](demo/web/README.md) |
| 用 Docker 预测自己的文件 | [手势、指力和疲劳的运行说明](deploy/README.md)。将合规数据放进对应的输入目录，再在 VS Code 对 Compose 文件执行 **Compose Up**。 |
| 复现手势实验 | `python -m emginsight.gesture.experiment` · [C 实验报告](results/gesture/classification/report.md) |
| 复现指力实验 | `python -m emginsight.force.regression` 训练连续回归，`python -m emginsight.force.levels` 训练整手三档分类。见[回归报告](results/force/regression/report.md)与[三档报告](results/force/levels/report.md)。 |
| 复现疲劳实验 | `python -m emginsight.fatigue.full_dataset` · [全试验四路与单路对照](results/fatigue/full_dataset/report.md) · [原右肱二头肌实验](results/fatigue/single_channel_biceps/report.md) |
| 比较两天的动作力量 | `python -m emginsight.force.change_experiment` · [两步模型与测试结果](results/force/change/report.md) · [用 Docker 预测](deploy/force_change/README.md) |
| 学习曲线 | `python -m emginsight.learning_curve` · [训练用户数的影响](results/learning_curve/report.md) |
| 跑测试 | 安装下文的 Python 环境后，在项目根目录运行 `python -m pytest`。 |

复现实验前，把公开数据解压到各任务的本地目录。[数据准备说明](docs/datasets.md)

## 我可以做什么？

| 工作                      | 具体要做什么                                                                                                                       | 做完后交付什么                                                         |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **~~A. 数据读取与处理~~** | 分别读取三个数据集，检查通道、标签、单位与用户编号；处理缺失和时间对齐，按用户划分训练/验证/测试集，再分窗。可以只认领一个数据集。 | 可调用的读取、分窗函数，以及一份能说明数据形状、标签和划分方式的例子。 |
| **~~B. 公共特征与算法~~**     | 提取 RMS、MAV、WL，使用 scikit-learn 实现 kNN、逻辑回归、岭回归以及多数类和均值基线。 | 其他组员能直接调用的函数或模型，用小数据验证计算正确，并附使用例子。   |
| **~~C. 手势识别实验~~**       | 使用 UCI 数据，调用 A/B 的代码比较分类模型，分析哪些手势容易混淆。 | [实验结果、混淆矩阵与复现命令](results/gesture/classification/report.md) |
| **~~D. 手指力量实验~~**   | 使用 putEMG-Force，从肌电预测选定的实测力量，比较均值基线与回归模型。                                                              | [连续力量回归](results/force/regression/report.md)、[三档力量分类](results/force/levels/report.md) |
| **E. 疲劳分类实验**       | 使用 Zenodo 数据，核对信号与评分时间对齐，预测 0/1/2 自感疲劳等级。 | [全试验比较、独立用户验证与实验限制](results/fatigue/full_dataset/report.md) |
| **F. 结果整理与展示**     | 汇总三个任务；在各数据集内部比较全通道与减少通道的效果；整理报告、图表和简单离线演示。                                             | 有来源的结果对比、最终报告，以及别人按说明能运行的演示。               |

**怎么认领：** 在 [Issues](https://github.com/miemiess/ML_GruopWork/issues) 先看有没有人正在做，没有就新建一条，例如“进行中：B / RMS 特征”，写上姓名和准备做的内容。卡住或完成时更新这条 Issue；不用另外维护任务表。

可以先做 A 的数据探索和 B 的小样例；C/D/E 随对应的数据与模型完成后接入；F 可先整理报告结构，最后汇总结果。涉及同一函数时先和对方商量，避免重复写。

## 代码放在哪里？

```text
src/emginsight/
  common/      公共层：按用户划分、带通滤波、模型文件校验、批量预测、作图设置
  features/    公共特征：RMS、MAV、WL
  models/      公共模型：kNN、逻辑回归、岭回归、基线与标准化
  gesture/     C 手势：读取 → 分窗 → 实验 → 模型 → 预测
  force/       D 指力：同上，另有三档分类与跨日变化
  fatigue/     E 疲劳：同上，含单路与全试验两套实验
  learning_curve.py  跨任务：训练用户数对效果的影响

tests/         与 src 同构：common/、gesture/、force/、fatigue/ …
results/       各任务报告与结果表：gesture/、force/、fatigue/、learning_curve/；模型和缓存在 runs/（不上传）
demo/web/      网页：页面、css/、js/，export/ 导出快照到 data/（不上传）并打包到 build/pages
deploy/        四个 Docker 推理镜像：gesture、force、force_change、fatigue
docs/          数据说明、实验记录模板、开发日志、项目说明与提案
data/          本地原始数据，不上传
```

**公共与私有的分界：** 与数据集无关、两个以上任务都在用的代码放在 `common/`、`features/`、`models/`；只属于某个数据集的读取、标签和实验流程放在对应任务包里，任务包之间不互相引用。各任务包使用相同的文件名分层，详见[代码说明](src/emginsight/README.md)。

各任务的训练、预测入口在对应的 Python 模块中。Notebook 可以调用同一套公共函数进行数据探索。

方便对接的约定只有这些：

- 一条原始信号按“采样点 × 通道”排列，读取时一起返回时间、标签、用户和记录编号。
- 分窗后特征为 `X: (窗口数, 特征数)`；分类标签为 `y: (窗口数,)`，力量目标为 `y: (窗口数, 输出数)`。
- 模型提供 `fit(X, y)` 和 `predict(X)`；函数说明输入、输出和单位。改接口前与使用它的组员沟通。

## 数据集

| 任务 | 数据来源                                                                              | 要注意什么                                                               |
| ---- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| 手势 | [UCI EMG Data for Gestures](https://archive.ics.uci.edu/dataset/481/emg+data)，8 通道 | 0 是未标注，1 是静息；不是每人都有第 7 类手势。                          |
| 力量 | [putEMG-Force](https://biolab.put.poznan.pl/putemg-dataset/)，24 通道                 | 预测实测 `FORCE_*`，不能把要求跟随的 `TRAJ_*` 轨迹当作真值；先确认单位。 |
| 疲劳 | [Zenodo 自感疲劳数据](https://zenodo.org/records/14182446)，8 块肌肉                  | 肌电与评分采样率不同，要按时间对齐；评分为 0/1/2。                       |

负责数据读取的同学可查看[数据说明](docs/datasets.md)和 [UCI 调用说明](src/emginsight/gesture/README.md)。三个任务的最终范围根据数据可用性确定。

## 所有任务共同注意

1. **可以用 scikit-learn 等现成的传统机器学习库训练主模型或建立基线。** 项目不使用深度学习。实验记录库版本、模型参数和数据处理流程，便于复现。
2. **先按用户划分，再分窗。** 同一用户不跨训练、验证和测试集；如果只能做会话留出，要说明，不能称为未见用户评估。
3. 标准化只在训练集上拟合，调参用验证集，方案确定后再看测试结果。
4. 算法先用小样例检查；每次实验记下数据版本、用户划分、参数、种子、运行方法和结果，可直接复制[实验记录模板](docs/experiment_template.md)。
5. 比较通道时使用同一数据集与划分，不能拿不同数据集的通道数直接判断优劣。失败或效果不好也如实记录。

## 环境安装

使用 Python 3.11 或更新版本：

```bash
git clone https://github.com/miemiess/ML_GruopWork.git
cd ML_GruopWork
python -m venv .venv
```

Windows PowerShell 激活环境：`.venv\Scripts\Activate.ps1`；macOS/Linux 使用 `source .venv/bin/activate`。然后运行：

```bash
python -m pip install -r requirements.txt
python -c "import emginsight; print(emginsight.__version__)"
python -m jupyterlab
```

PowerShell 无法激活时，可用 `.venv\Scripts\python.exe` 替代 `python`。Notebook 选择同一环境。用 `python -m pytest` 运行测试。

背景资料：[项目提案](docs/Group_12_EMGInsight_Visual_Proposal_Revised.pdf)。
