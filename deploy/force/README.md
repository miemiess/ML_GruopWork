# 指力预测容器

把 D 任务训练好的两个模型、预处理配置和固定版本依赖打包成 Docker 镜像：输入原始肌电 CSV，同时输出十路手指力量的连续预测和整只手三档力量（静息 / 轻 / 重）。镜像只做离线批处理预测，不重新训练。

## 1. 准备模型并构建镜像

首次使用或更新打包的模型和代码后，在项目根目录构建镜像。

```powershell
docker build -f deploy/force/Dockerfile -t emginsight-force .
```

构建前需准备 `results/runs/putemg_force/` 中的 `force_model.joblib`（连续力量）和 `force_level_model.joblib`（三档）。模型卡分别位于 `results/putemg_force/regression/model_card.json` 和 `results/putemg_force/levels/model_card.json`。回归目录下还需要 `monitoring_reference.json`。

- 依次运行 `python -m emginsight.putemg_force.experiment` 和 `python -m emginsight.putemg_force.levels` 训练生成；或
- 从组内分发处取得模型和对应的模型卡，按上述路径放置。

`.dockerignore` 只放行源码、两个镜像共用的依赖清单 `deploy/requirements.txt` 和模型相关文件，原始数据不会进入镜像。镜像约 0.9 GB，本机首次构建约 45 秒。

## 2. 预测

把 CSV 放进 `deploy/force/input/`，对 `deploy/force/docker-compose.yml` 执行 **Compose Up**。结果写进 `deploy/force/output/`。命令行用法见 [Docker 批量预测](../README.md)。

不用 Docker 时，在项目虚拟环境中运行同一入口：

```powershell
.\.venv\Scripts\python.exe -m emginsight.putemg_force.predict --input deploy/force/input --output-dir deploy/force/output
```

### 输入格式

| 列 | 是否必需 | 说明 |
| --- | --- | --- |
| `EMG_1` … `EMG_24` | 必需 | putEMG 电极排布的 24 路原始肌电，约 5120 Hz |
| 时间列（`time`、`Timestamp` 或 putEMG 的未命名首列） | 可选 | 单位为秒，必须严格递增；没有时按 5120 Hz 生成。采样率偏离 2% 以上会拒绝 |
| `FORCE_1` … `FORCE_10` | 可选 | 有则同时计算 MAE、RMSE、R² 作为对照 |
| `FORCE_MVC` | 可选 | 与 `FORCE_*` 同时存在时，按实验规则生成三档标签，计算三档准确率和 macro-F1 |
| `TRAJ_*` | 可选 | 有则检查开头 4 秒没有出力指令 |

- **静息要求**：记录开头 4 秒必须保持手部放松，模型用这一段做肌电校准。
- **长度要求**：记录至少要长于 4 秒加一个窗口。
- **文件名**：不限。

### 输出（每个输入文件）

- `<文件名>_predictions.csv`：逐窗预测。窗口约 200 毫秒、步长约 100 毫秒，列为 `start_sample`、`time_s` 和 `FORCE_1_pred` … `FORCE_10_pred`；输入含实测力量时，另加 `_true` 列。
  三档结果在同一文件的 `level_pred`（0 静息、1 轻、2 重）、`level_name` 和 `level_prob_rest/light/heavy` 列；可计算标签时另加 `level_true`。
- `<文件名>_curves.png`：十路预测曲线；输入含实测力量时一并画出实测。
- `<文件名>_levels.png`：三档时间线和各档概率。
- `summary.json`：在[通用汇总字段](../README.md#结果)之外，包含漂移检查和三档预测信息。

预测值是**相对静息的 putEMG 原始力量数值，不是牛顿**；它反映趋势和相对大小，幅度通常偏小。三档按每个人自己的最大力定义：轻约 25% MVC，重约 50% MVC。测试用户上三档 macro-F1 为 0.511，“轻”常被判为静息。各项精度见[回归报告](../../results/putemg_force/regression/report.md)和[三档报告](../../results/putemg_force/levels/report.md)。

## 3. 一致性验证

在本机用留出测试用户 46 的完整记录（`emg_force-46-repeats_long-2018-04-20-17-36-28-641.csv`）做了核对：

| 运行方式 | 结果 |
| --- | --- |
| 本地入口 | 预测与实验保存的逐窗结果一致，差异仅来自保存 CSV 时的小数位四舍五入 |
| 容器 | 输出 CSV 与本地入口逐值相同，对照实测 RMSE 80.64、R² −0.048；三档预测、标签和概率与三档实验逐窗完全一致，macro-F1 0.516 |

这名用户是两条记录合计 R² 0.131 的中位用户，单条记录可能更低。

- 读入后，6 分钟的记录约 3,500 窗。
- 特征加推理约 0.5 ms/窗，远小于 100 毫秒的步长。
- 读取 400 MB CSV 是主要耗时，含容器启动全程约 12 秒。

读取模型时会核对 `model_card.json` 中的 SHA-256 和 scikit-learn 版本，二者不符时拒绝运行。

## 4. 部署时需要监控

- **延迟**：`summary.json` 的 `ms_per_window`。实时使用时应远低于窗口步长（约 100 毫秒）。
- **输入漂移**：程序把本次静息段各通道的 log RMS 与训练用的 50 条记录比较。超过 3 个标准差的通道达到 6 路时，给出漂移告警，通常意味着电极位置、接触或设备增益与训练数据不同，预测可信度会下降。
- **力量零点漂移**：实验中同一记录的力量零点会在使用过程中漂移。长时间使用时，建议定期重新做静息校准。
- **适用范围**：模型只用 putEMG 的 24 路电极排布、`repeats_long` 任务和 44 名健康成年人训练。换设备、换电极位置或换人群时，需要重新采集和验证。仅用于研究，不用于临床。
