# EMGInsight

61.501 Production Machine Learning · Group 12 · 仓库：`ML_GruopWork`

使用传统机器学习分析公开表面肌电（sEMG）数据，研究动作识别、手指力量估计与肌肉疲劳分类。

## 项目任务

| 任务 | 计划使用的数据集 | 预测目标 |
| --- | --- | --- |
| 手势识别 | UCI EMG Gestures | 手势类别 |
| 手指力量估计 | putEMG-Force | 力量信号 |
| 肌肉疲劳分类 | Zenodo sEMG Fatigue | 主观疲劳等级（0/1/2） |

## 方法

- 统一流程：数据预处理 → 滑动窗口 → RMS / MAV / WL 特征 → 模型训练 → 评估。
- 使用 Python / NumPy 实现核心传统机器学习算法，并比较基线模型；不使用深度学习。
- 比较各数据集内的通道子集，按用户或采集会话划分数据以避免泄漏，并评估未见用户上的泛化表现。
- 分类使用 F1，回归使用 MAE / RMSE。项目仅用于离线研究。

## 当前阶段

项目初始化，尚未实现训练代码。下一步确认数据集可用性与组员分工，建立预处理流程并完成首个基线。三个任务的最终范围按数据可行性确定。

计划交付：可复现代码、模型与通道对比、评估图表、离线演示和最终报告。环境安装与运行说明随实现补充。

## 小组协作

成员：Ma Mingyang、Han Yuchen、Zhou Yin、Wei Xin、Gao Ruiyan、Li Ji（分工待定）。

采用 **Collaborative Development（协同开发）**：所有成员在同一个公共 GitHub 仓库共同构建完整的 EMG 机器学习系统。成员可参与不同模块，共同维护和优化整个项目。

- 统一项目结构、数据处理流程和模型接口，方便复用与集成。
- 使用 Issues 记录任务、负责人及验收条件；跨模块修改先沟通接口。
- 从最新 `main` 创建功能分支，例如 `feature/gesture-baseline`。
- 通过 Pull Request 说明修改内容和验证结果，由另一位组员检查后合并到 `main`。
- 不直接向 `main` 提交日常开发修改；此为协作约定，尚未配置强制分支保护。

## 计划结构与接口

以下为起步约定，目录和接口将在开发时逐步实现：

```text
data/          数据说明与本地数据（原始数据不提交到 Git）
notebooks/     数据探索与实验
src/           数据读取、预处理、特征、模型与评估
tests/         核心算法及接口测试
reports/       评估图表与报告
```

- 数据读取统一返回信号、标签及用户/会话标识；各数据集保留独立适配器。
- 特征矩阵统一为 `X: (n_samples, n_features)`，标签为 `y`，并保留样本分组信息。
- 模型统一提供 `fit(X, y)` 和 `predict(X)`，分类与回归分别评估。
- 先按用户或会话划分训练/测试集，再分窗与拟合预处理参数；测试数据不参与训练或调参。

## 开始协作

```bash
git clone https://github.com/miemiess/ML_GruopWork.git
cd ML_GruopWork
git switch -c feature/your-task
```

完成修改后提交并推送个人分支，在 GitHub 创建面向 `main` 的 Pull Request。运行环境和依赖清单将在首个可运行模块完成时补充。

项目依据：[项目提案](Group_12_EMGInsight_Visual_Proposal_Revised.pdf)。
