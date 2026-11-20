# 四路肌电疲劳分类

每份 CSV 是一次完整试验。使用同一侧的肱二头肌、三角肌前束、中束和后束四路原始电压，保留 Zenodo 列名及每路前面的秒时间列 `X [s]`。采样率约 1259 Hz，窗长 4 秒，步长 2 秒。保留记录开头，特征会用首个有效窗口作参考。推理无需评分文件。

## 运行

在项目根目录运行。模型位于 `results/runs/full_fatigue/fatigue_model.joblib`。若从新的检出目录开始，请准备 Zenodo 数据后运行 `python -m emginsight.zenodo_fatigue.full_dataset` 生成模型、模型卡和评估产物。

1. 将原始 CSV 放入 `deploy/fatigue/input/`。多份记录用不同文件名，例如 `subject_10_trial_11.csv`。
2. 启动 Docker Desktop，构建并运行镜像。

```powershell
docker compose -f deploy/fatigue/docker-compose.yml build
docker compose -f deploy/fatigue/docker-compose.yml run --rm predict
```

也可以在 VS Code 的 Container Tools 扩展中右键本目录的 `docker-compose.yml` 并选择 **Compose Up**。本地 Python 等价命令为：

```powershell
.\.venv\Scripts\python.exe -m emginsight.zenodo_fatigue.predict --input deploy/fatigue/input --output-dir deploy/fatigue/output
```

## 查看结果

在 `output/` 中，逐窗 CSV 含起止时间、预测类别及三类概率。0 表示无疲劳，1 表示过渡，2 表示疲劳。每份记录另有一张等级时间线 PNG。`summary.json` 汇总成功与失败记录、类别计数和处理耗时。

网页仅回放有一致自感评分的测试窗口。部署预测覆盖所有符合信号要求的完整窗口，因此同一记录的窗口数可能不同。四折跨人评估中，四路模型的 macro-F1 为 0.388，低于时间基线的 0.526。输出用于复现实验和比较趋势，医学评估仍需独立验证。见[实验报告](../../results/zenodo_fatigue/full_dataset/report.md)。
