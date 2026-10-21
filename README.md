# EMGInsight

### 使用传统机器学习解读肌肉信号

**61.501 Production Machine Learning · 第 12 组 · PML 2026**

EMGInsight 使用公开的表面肌电（sEMG）数据，研究手势识别、手指力量估计与肌肉疲劳分类。我们将共同构建可复现的实验框架，比较传统机器学习模型，研究通道数量减少对预测的影响，并评估模型对未见用户的泛化能力。

**当前阶段：基础工作区搭建。** 数据适配器、特征提取、模型、训练流程与离线演示尚待实现，目前没有实验结果或训练好的模型。[项目提案](Group_12_EMGInsight_Visual_Proposal_Revised.pdf)是项目参考依据，三个任务的最终范围将根据数据可行性确定。

[需求清单与认领](docs/requirements.md) · [架构与模块](docs/architecture.md) · [贡献流程](贡献.md) · [数据集指南](data/README.md)

## 项目约束

- **不使用深度学习。** 采用传统分类与回归方法。
- **使用 Python/NumPy 自行实现核心机器学习算法。** 可以使用库模型进行参考对比，但必须明确标注，不能代替自行实现的算法。
- **开展三个数据集对应的任务。** 复用公共工具，同时分别维护数据适配器、标签定义与评估方案。
- **评估预测效果、通道子集与未见用户泛化能力。** 在相同数据划分下比较简单基线与自行实现的模型。
- **保证实验可复现。** 记录数据集版本、文件校验值、数据划分、配置、随机种子、运行环境与命令。
- **说明社会价值与研究局限。** 结论应与离线研究获得的证据相符。

计划交付：可复现代码、自行实现的机器学习算法、三个任务的研究结果（取决于数据可行性）、模型与通道对比、评估图表、离线演示及最终报告。

## 三个数据集与对应任务

| 研究任务 | 公开数据来源与输入 | 预测目标 | 主要限制 |
| --- | --- | --- | --- |
| 手势识别 | [UCI EMG Data for Gestures](https://archive.ics.uci.edu/dataset/481/emg+data)：8 通道前臂肌电，36 名参与者 | 手势类别 | 标签 0 表示未标注数据；并非所有参与者都完成了第 7 类手势。需明确样本纳入规则，不能把相邻窗口当作来自不同用户的独立样本。 |
| 手指力量估计 | [putEMG-Force](https://biolab.put.poznan.pl/putemg-dataset/)：24 通道肌电及实测力量 | 选定的一个或多个实测力量信号 | 使用 `emg_force` 试验。实测 `FORCE_*` 列与要求参与者跟随的 `TRAJ_*` 目标轨迹不同。建模前需核对目标映射、单位、校准方式及参与者和采集会话的完整性。 |
| 肌肉疲劳分类 | [Zenodo sEMG 与自感疲劳数据集 v2](https://zenodo.org/records/14182446)：8 块肌肉，13 名健康成年人 | 自感疲劳等级 0/1/2 | 肌电信号与疲劳评分采样率不同，需按时间戳对齐。标签具有主观性，健康人群的小样本限制了结论的适用范围。 |

三个数据集的设备、肌肉位置、采集流程与标签各不相同。通道子集应在**同一个数据集内**比较，不能根据不同数据集的通道数量直接判断传感方案优劣。每个任务分别维护模型与结果。数据获取、来源记录及再分发要求见[数据集指南](data/README.md)。

## 研究方法与评估

```text
公开数字化 sEMG 数据与元信息
  -> 按用户或采集会话划分数据
  -> 预处理与滑动窗口
  -> RMS / MAV / 波形长度特征
  -> 自行实现传统机器学习模型，并与基线对比
  -> 在留出的测试集上评估，比较通道方案
```

先建立简单基线：分类任务预测训练集中的多数类别，回归任务预测训练目标的均值。候选算法包括 k 近邻、逻辑回归和岭回归，最终选择在各实验中记录。

分类使用宏平均 F1（macro-F1）、各类别指标及混淆矩阵；回归使用 MAE 与 RMSE，并注明目标单位及各输出的单独指标。在可行时报告各用户的表现与差异。

主要泛化研究采用用户留出评估，按采集会话留出的实验需单独标明。先划分原始记录，再进行分窗；标准化、特征选择及其他需要拟合的变换只能使用训练数据。使用按组划分的验证集调参，确定方案后再评估测试集。详细要求见[研究与评估规范](docs/research_protocol.md)。

## AI 的社会价值

潜在应用包括无障碍假肢手交互、康复研究、运动员与工作场景中的肌肉监测，以及替代式 VR 交互。减少通道数量的研究可为更简便的传感方案提供参考。

这些是研究动机，目前尚未验证相关应用效果。项目仅开展离线研究，不作临床诊断或实际设备控制能力的声明。最终报告应讨论参与者覆盖范围、失败案例及无障碍应用的局限。

## 小组成员与协作方式

**成员：** Ma Mingyang · Han Yuchen · Zhou Yin · Wei Xin · Gao Ruiyan · Li Ji

所有成员在同一个仓库共同构建完整系统，**不固定分配个人角色**。通过 Issues 选择和协调工作，成员可以跨模块参与。可选工作包括三个任务的研究、预处理与特征工程、算法实现与测试、评估、集成、文档和演示。

1. 通过 Issue 确认任务范围、接口与验收条件。
2. 从最新的 `main` 创建功能分支，例如 `feature/gesture-baseline`。
3. 完成范围明确的修改，补充文档并运行相关检查。
4. 创建面向 `main` 的 Pull Request（PR），说明验证结果与数据假设。
5. 由另一位组员审阅，处理反馈后合并。

仓库当前只保留 `main`，组员认领需求后自行创建短期分支，合并后删除。完整操作见[贡献.md](贡献.md)，包括进度认领、建分支、提交、同步、冲突处理、PR 审阅、合并和分支清理。本次架构规划直接提交主线；后续功能开发采用 PR。当前没有新增强制分支保护。

## 架构与工作区

采用一个模块化 Python 项目：三个任务分别适配数据，公共模块共享契约与算法，由统一编排层串联。详细职责、依赖方向、数组形状、模型接口及运行文件格式见[架构约定](docs/architecture.md)。

```text
.
├── docs/                       # 架构、需求、研究规范、实验记录与报告
├── data/                       # 原始/中间/处理数据（本地），来源清单与划分
├── src/emginsight/
│   ├── core/                   # 公共数据契约、配置校验与异常
│   ├── datasets/               # UCI、putEMG-Force、Zenodo 独立适配器
│   ├── splitting/              # 分窗前按用户或会话划分
│   ├── preprocessing/          # 滤波、分窗、仅训练集拟合的变换
│   ├── features/               # RMS、MAV、WL 等确定性特征
│   ├── models/                 # 自定义基线、kNN、逻辑回归、岭回归
│   ├── evaluation/             # 指标、图表与公平对比
│   ├── artifacts/              # 保存及恢复模型、配置、环境与结果
│   └── pipelines/              # 统一训练、评估、推理入口
├── notebooks/experiments/       # 探索与可视化，调用公共包
├── configs/                    # 三任务配置与模板
├── tests/                      # 按模块的单元测试、集成测试与合成样例
├── results/                    # 公开摘要/图表；本地运行文件保持忽略
├── demo/                       # 调用统一推理入口的离线展示
├── 贡献.md                     # 完整中文协作流程
├── CONTRIBUTING.md             # GitHub 贡献指南入口
├── pyproject.toml              # Python 包与工具设置
└── requirements.txt            # 安装入口
```

以上模块目录已经建立，算法、命令行与测试实现仍待需求开发。项目提案保留在根目录。公共接口先确定，各模块使用合成数据独立验证，再通过统一流程集成。

## 环境安装

使用 **Python 3.11 或更新版本**。在终端执行：

```bash
git clone https://github.com/miemiess/ML_GruopWork.git
cd ML_GruopWork
python -m venv .venv
```

激活虚拟环境：

| 终端 | 命令 |
| --- | --- |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows 命令提示符 | `.venv\Scripts\activate.bat` |
| macOS/Linux | `source .venv/bin/activate` |

如果 PowerShell 阻止激活，可以在后续命令中使用 `.venv\Scripts\python.exe` 替代 `python`。

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import emginsight; print(emginsight.__version__)"
python -m pip check
```

上述命令会以可编辑方式安装项目包，以及数据分析、Notebook、参考模型和测试所需的基础工具。各数据集的读取工具可能需要在检查文件格式后补充依赖。运行 `python -m jupyterlab` 后，可在 `notebooks/experiments/` 中开展探索。

目前尚无训练或演示命令。添加测试后使用 `python -m pytest`；当前基础工作区没有可执行测试，会报告未收集到测试并返回退出码 5。详见[测试说明](tests/README.md)。

## 需求清单

完整开发范围共 **22 项需求**，每项均列出目标、前置需求、交付目录、验收条件与验证方式。请从[详细需求清单](docs/requirements.md)选择任务，在 [GitHub Issues](https://github.com/miemiess/ML_GruopWork/issues) 搜索编号并认领，然后建立自己的分支。

| 分区 | 需求 |
| --- | --- |
| 基础契约 | [R01 公共类型](docs/requirements.md#r01)、[R02 配置校验](docs/requirements.md#r02) |
| 数据适配 | [R03 UCI](docs/requirements.md#r03)、[R04 力量](docs/requirements.md#r04)、[R05 疲劳](docs/requirements.md#r05) |
| 公共处理 | [R06 分组划分](docs/requirements.md#r06)、[R07 分窗](docs/requirements.md#r07)、[R08 特征](docs/requirements.md#r08)、[R09 标准化](docs/requirements.md#r09) |
| 算法 | [R10 基线与 kNN](docs/requirements.md#r10)、[R11 逻辑回归](docs/requirements.md#r11)、[R12 岭回归](docs/requirements.md#r12) |
| 评估与集成 | [R13 指标图表](docs/requirements.md#r13)、[R14 保存恢复](docs/requirements.md#r14)、[R15 统一流程](docs/requirements.md#r15) |
| 三任务研究 | [R16 手势](docs/requirements.md#r16)、[R17 力量](docs/requirements.md#r17)、[R18 疲劳](docs/requirements.md#r18) |
| 比较与交付 | [R19 通道研究](docs/requirements.md#r19)、[R20 离线演示](docs/requirements.md#r20)、[R21 环境与 CI](docs/requirements.md#r21)、[R22 最终报告](docs/requirements.md#r22) |

状态统一为 **待认领 → 进行中 → 待审阅 → 已完成**，遇到依赖问题标记**受阻**。通过 Issue 标题、负责人和关联分支/PR 告知大家正在做哪一需求。现有实现任务尚未完成，不预先分配人员。

原来的实验复现内容已纳入 R02、R06、R14、R16-R18、R21 和 R22 的验收要求，包括配置、精确划分、数据校验值、环境、随机种子与其他成员复跑。所有结果必须可追溯。

数据集权利归各原始提供方所有。目前尚未选定整个项目的代码许可证。
