# Docker 批量预测

各任务分别提供 Compose 入口。启动 Docker Desktop 后，在项目根目录构建需要的镜像；构建前请先按任务说明准备模型文件和模型卡。

| 任务 | 输入与目录 | 配置与模型准备 |
| --- | --- | --- |
| 手势 | 八通道 UCI TXT 或 CSV，放进 `deploy/gesture/input/` | [手势部署说明](gesture/README.md) · [Compose 文件](gesture/docker-compose.yml) |
| 指力 | 24 通道 putEMG-Force CSV，放进 `deploy/force/input/` | [指力部署说明](force/README.md) · [Compose 文件](force/docker-compose.yml) |
| 两日指力变化 | 两日 24 通道记录和 `visits.csv` 动作清单，放进 `deploy/force_change/input/` | [模型准备、单值输出与运行说明](force_change/README.md) · [Compose 文件](force_change/docker-compose.yml) |
| 自感疲劳 | 同侧四路 Zenodo 原始 CSV，放进 `deploy/fatigue/input/` | [模型与运行说明](fatigue/README.md) · [Compose 文件](fatigue/docker-compose.yml) |

## 在 VS Code 运行

安装微软的 Container Tools 扩展。把文件直接放进对应的 `input/`，右键该任务的 `docker-compose.yml`，选择 **Compose Up**。容器处理完会自动退出，退出码 `0` 表示本批成功。终端或 **Containers** 面板可以查看日志。每次启动会处理 `input/` 第一层当时存在的全部支持文件，结果会写入同级的 `output/`。Git 只跟踪输入和输出目录的占位文件。

也可以在项目根目录的终端运行下面对应的一条命令。

```powershell
docker compose -f deploy/gesture/docker-compose.yml up --exit-code-from predict
docker compose -f deploy/force/docker-compose.yml up --exit-code-from predict
```

两个镜像的命令行都使用 `--input` 和 `--output-dir`，输入可以是一个文件、多个文件或目录。只运行一个指力文件的例子如下。

```powershell
docker compose -f deploy/force/docker-compose.yml run --rm predict --input "/data/input/记录.csv" --output-dir /data/output
```

指力仍接受原来的 `--csv` 参数。已有旧版镜像时，先重新构建一次再使用新版 Compose 和汇总字段。

## 结果

手势、连续指力与疲劳任务均输出逐窗预测 CSV、曲线图和 `summary.json`。汇总包含 `task`、`model`、`counts`（总数、成功数、失败数）以及 `files`（每个输入文件的处理结果）。

- 疲劳任务按文件汇总窗口数、各预测类别计数及每窗耗时。
- 手势与连续指力任务中，成功的文件有 `status: "ok"`、窗口数、时长、处理耗时和每窗的特征提取与推理耗时。有实测值或标签时附对照指标。
- 失败的文件有 `status: "error"` 和原因。程序继续处理其他输入，最后以退出码 `1` 结束。
- 手势另外记录置信度和各预测类别占比；指力另外记录静息肌电漂移和三档预测信息。
