# 手势识别容器

把 C 任务训练好的手势模型、分窗和特征配置以及固定版本依赖打包成 Docker 镜像：输入原始 8 路肌电文件，逐窗输出六类手势或“未知”，以及各类概率。镜像只做离线批处理识别，不重新训练。

## 1. 准备模型并构建镜像

首次使用或更新打包的模型和代码后，在项目根目录构建镜像。

```powershell
docker build -f deploy/gesture/Dockerfile -t emginsight-gesture .
```

构建前需准备 `results/runs/gesture/gesture_model.joblib` 和 `results/gesture/classification/model_card.json`。

- 准备好 UCI 数据后，运行 `python -m emginsight.gesture.experiment` 生成；本机约 12 秒；或
- 从组内分发处取得模型和对应的模型卡，按上述路径放置。

`.dockerignore` 只放行源码、两个镜像共用的依赖清单 `deploy/requirements.txt` 和模型相关文件，原始数据不会进入镜像。镜像约 0.8 GB。

## 2. 预测

把待识别的 TXT 或 CSV 放进 `deploy/gesture/input/`，对 `deploy/gesture/docker-compose.yml` 执行 **Compose Up**。结果写进 `deploy/gesture/output/`。命令行用法见 [Docker 批量预测](../README.md)。

不用 Docker 时，在项目虚拟环境中运行同一入口：

```powershell
.\.venv\Scripts\python.exe -m emginsight.gesture.predict --input deploy/gesture/input --output-dir deploy/gesture/output
```

### 输入格式

UCI 原始 txt（制表符或空格分隔）或 CSV（逗号分隔）均可。

| 列 | 是否必需 | 说明 |
| --- | --- | --- |
| `channel1` … `channel8` | 必需 | MYO 手环 8 路原始肌电 |
| `time` | 可选 | 毫秒，不能倒退。没有时按 1 ms 间隔生成；相邻采样中位间隔不在 0.8–1.25 ms 时拒绝 |
| `class` | 可选 | UCI 标注 0–7。有则对“窗口内标签统一且属于 1–6”的窗口计算准确率和 macro-F1，与 C 实验口径一致 |

记录至少要有一个 200 点的完整窗口。文件名不限。

### 输出（每个输入文件）

- `<文件名>_predictions.csv`：逐窗结果。窗口 200 点、步长 100 点，约 200/100 毫秒。列为 `start_sample`、`time_ms`（窗口中心）、`prediction`（0 未知或 1–6）、`prediction_name`、`raw_prediction`（不拒识时的六类结果）、`prob_1` … `prob_6`；有标注时另加 `label`，混合标签窗口记为 −1。
- `<文件名>_gestures.png`：上图是预测手势随时间的变化，有标注时以灰色宽带画出单标签窗口；下图是六类概率。
- `summary.json`：在[通用汇总字段](../README.md#结果)之外，包含各类及未知占比、置信度监控和拒识指标。

类别：0 未知、1 静息、2 握拳、3 腕屈、4 腕伸、5 桡偏、6 尺偏。

**“未知”**：六类中最大概率低于 0.75 时输出 0。阈值在 C 实验验证用户上选出，保存在模型文件里。动作之间的过渡、没训练过的手势或模型拿不准的窗口会被标为未知。代价是部分真实手势也会被判为未知：测试用户上约 41%，被接受手势的准确率从 77.6% 升到 93.6%。需要六类结果时使用 `raw_prediction`。各项精度见[实验报告](../../results/gesture/classification/report.md)。

## 3. 一致性验证

在本机用 C 实验留出测试用户的原始文件做了核对：

| 输入 | 本地入口与实验 | 容器 |
| --- | --- | --- |
| 用户 09 两份记录 | 323 个单标签窗口的标签与六类预测与实验 `test_predictions.csv` 完全一致；准确率 99.36%、98.80%；未知占全部窗口 14.1%、18.7% | 输出 CSV 与本地逐值相同 |
| 用户 15 第一份记录 | 六类准确率 64.77%（C 实验中该用户最低，两份合计 62.82%）；未知占 52.7%，被接受窗口准确率 97.3% | 输出 CSV 与本地逐值相同 |
| 用户 09 第一份只保留 8 个通道的 CSV（无时间、无标注） | 预测与原文件完全相同 | 同左 |

- 一份约 1 分钟的记录约 600 窗。
- 特征加推理约 0.05–0.1 ms/窗，含容器启动全程只需几秒。
- 读取模型时会核对 `model_card.json` 中的 SHA-256 和 scikit-learn 版本，二者不符时拒绝运行。

## 4. 部署时需要监控

- **延迟**：`summary.json` 的 `ms_per_window`。实时使用时应远低于 100 毫秒的步长。
- **置信度与未知占比**：`monitoring` 给出平均最大类别概率和低置信度窗口占比，`predicted_share.unknown` 给出未知占比。未知占比明显升高（如超过一半），通常意味着佩戴位置、用户或动作与训练数据不同。
- **类别分布**：`predicted_share` 给出各类预测占比。长时间偏向某一类时，应检查信号质量。
- **适用范围**：模型只用 UCI 数据集 22 名训练用户、MYO 手环 8 路电极和这六种动作训练。7 名留出用户的准确率为 77.61%，各用户从 62.82% 到 99.07% 不等。换设备、换佩戴方式或增加动作时，需要重新采集和验证。仅用于研究，不用于临床。
