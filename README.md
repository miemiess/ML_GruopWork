# EMGInsight

**61.501 Production Machine Learning · 第 12 组**

用传统机器学习分析表面肌电（sEMG），完成手势识别、手指力量估计和肌肉疲劳分类，为辅助交互、假肢与康复研究提供参考。项目只做离线研究，不用于临床。

成员：Ma Mingyang、Han Yuchen、Zhou Ying、Wei Xin、Gao Ruiyan、Li Ji。

**进度：基础工作（A 数据处理、B 公共特征与模型、C 手势、D 指力、E 疲劳）已全部完成。** 剩下 F：汇总结果、通道减少对比、最终报告与展示。

## 结果一览

所有成绩都在**训练时从没见过的用户**上测得：按用户划分训练、验证、测试集，测试集只在选定方案后评估一次。

| 任务 | 数据 | 获选模型 | 测试结果 | 对照 | 报告 |
| --- | --- | --- | --- | --- | --- |
| C 手势六分类 | UCI，8 通道，7 名测试用户 | RMS/MAV/WL + 随机森林 | macro-F1 0.773，准确率 77.6% | 多数类 0.049 | [报告](results/gesture/classification/report.md) |
| D 连续指力 | putEMG-Force，24 通道，9 名测试用户 | 静息校准 + 随机森林 + 五窗平滑 | 平均 RMSE 73.2，R² 0.24 | 均值 83.5 | [报告](results/force/regression/report.md) |
| D 力度三档 | 同上 | 类别平衡随机森林 | macro-F1 0.511 | 多数类 0.230；回归后分档 0.376 | [报告](results/force/levels/report.md) |
| D 两日力量变化 | 同上，跨日复测 | 十路预测 + 岭回归 | 变化 MAE 27.8，方向判对 14/18 | 始终报零 38.0 | [报告](results/force/change/report.md) |
| E 自感疲劳三级 | Zenodo，同侧四路，13 人四折 | 平滑与个人基线特征 + 逻辑回归 + 单调约束 | macro-F1 0.496（第一版 0.388） | 仅运动时长 0.526；多数类 0.215 | [报告](results/fatigue/full_dataset/report.md) |

- **主要发现：** 跨用户泛化主要受训练人数影响（[学习曲线](results/learning_curve/report.md)）；用记录开头的静息段做个人校准是最有效的改进。
- **疲劳任务的结论：** 肌电模型改进后仍不如只看运动时长，现有数据不足以证明肌电比运动时长更能预测新用户的主观疲劳。
- **手势“未知”拒识：** 手势模型对低置信度窗口输出“未知”，被接受窗口的准确率升到 93.6%。

## 快速开始

| 目标 | 入口 |
| --- | --- |
| 看网页 | [GitHub Pages](https://miemiess.github.io/ML_GruopWork/)，或本地打开 `demo/web/index.html`（先按[网页说明](demo/web/README.md)导出快照）。每个实验台有观察案例、训练过程动画和开发过程三种视图。 |
| 用 Docker 预测自己的文件 | [部署说明](deploy/README.md)：把数据放进对应的 `input/`，对 Compose 文件执行 **Compose Up**。手势、指力、两日变化、疲劳各一个镜像。 |
| 复现实验 | `python -m emginsight.gesture.experiment`<br>`python -m emginsight.force.regression` · `python -m emginsight.force.levels` · `python -m emginsight.force.change_experiment`<br>`python -m emginsight.fatigue.full_dataset` · `python -m emginsight.learning_curve` |
| 跑测试 | `python -m pytest` |

复现前把公开数据解压到 `data/raw/` 下的对应目录，见[数据准备说明](docs/datasets.md)。模型文件较大，不上传 Git，由各实验命令生成到 `results/runs/`。

## 代码结构

```text
src/emginsight/
  common/      公共层：按用户划分、带通滤波、模型文件校验、批量预测、作图设置
  features/    公共特征：RMS、MAV、WL
  models/      公共模型：kNN、逻辑回归、岭回归、基线与标准化
  gesture/     C 手势：读取 → 分窗 → 实验 → 模型 → 预测
  force/       D 指力：同上，另有三档分类与跨日变化
  fatigue/     E 疲劳：同上，含单路与全试验两套实验
  learning_curve.py  跨任务：训练用户数对效果的影响

tests/         与 src 同构
results/       各任务报告与结果表；模型和缓存在 runs/（不上传）
demo/web/      网页：页面、css/、js/，export/ 导出快照并打包到 build/pages
deploy/        四个 Docker 推理镜像
docs/          数据说明、实验记录模板、开发日志、课程说明与提案
data/          本地原始数据（不上传）
```

与数据集无关、多个任务共用的代码放在 `common/`、`features/`、`models/`；只属于某个数据集的读取、标签和实验流程放在对应任务包里，任务包之间不互相引用。详见[代码说明](src/emginsight/README.md)。

## 数据集

| 任务 | 数据来源 | 注意 |
| --- | --- | --- |
| 手势 | [UCI EMG Data for Gestures](https://archive.ics.uci.edu/dataset/481/emg+data)，8 通道 | 0 是未标注，1 是静息；不是每人都有第 7 类手势。 |
| 力量 | [putEMG-Force](https://biolab.put.poznan.pl/putemg-dataset/)，24 通道 | 预测实测 `FORCE_*`，不把指令轨迹 `TRAJ_*` 当真值；力量单位为设备原始刻度，不是牛顿。 |
| 疲劳 | [Zenodo 自感疲劳数据](https://zenodo.org/records/14182446)，8 块肌肉 | 肌电与评分采样率不同，按时间对齐；评分为 0/1/2。 |

## 实验原则

1. 只用 scikit-learn 等传统机器学习方法，不用深度学习；记录库版本、参数和种子。
2. **先按用户划分，再分窗。** 同一用户不跨训练、验证和测试集。
3. 标准化只在训练集拟合，调参用验证集，方案确定后才看测试结果。
4. 每个模型都与简单基线比较；效果不好也如实记录。实验记录可复制[模板](docs/experiment_template.md)。

## 环境安装

使用 Python 3.11 或更新版本：

```bash
git clone https://github.com/miemiess/ML_GruopWork.git
cd ML_GruopWork
python -m venv .venv
```

激活环境（Windows PowerShell：`.venv\Scripts\Activate.ps1`；macOS/Linux：`source .venv/bin/activate`），然后：

```bash
python -m pip install -r requirements.txt
python -m pytest
```

PowerShell 无法激活时，可用 `.venv\Scripts\python.exe` 代替 `python`。协作流程见[贡献指南](CONTRIBUTING.md)，开发记录见[开发日志](docs/开发日志.md)，背景见[项目提案](docs/Group_12_EMGInsight_Visual_Proposal_Revised.pdf)。
