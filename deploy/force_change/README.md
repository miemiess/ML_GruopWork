# 两日指力变化预测

镜像 `emginsight-force-change` 内含训练好的十路指力模型和跨人岭回归模型。一次运行比较同一人的基线日与复测日，输出一个有符号的变化值，同时保存完整推理明细。

## 1. 准备模型

本机已有 `results/runs/force/force_model.joblib` 和 `force_change_model.joblib`。从新检出复现时，可放入组内分发的这两个文件及匹配的模型卡，或按顺序运行：

```powershell
.\.venv\Scripts\python.exe -m emginsight.force.regression
.\.venv\Scripts\python.exe -m emginsight.force.change_experiment
```

第一条生成冻结模型和每条记录的特征缓存。第二条拟合、验证、测试并回读第二阶段模型，输出[实验报告与模型卡](../../results/force/change/report.md)。更新模型或源码后，从项目根目录构建镜像：

```powershell
docker compose -f deploy/force_change/docker-compose.yml build
```

## 2. 放入两天的记录

`input/visits.csv` 每行标记一次动作的稳定保持段。字段为 `patient,protocol,date,record,start_s,end_s`。两天的记录使用相同的受试者编号和动作协议；`record` 是相对于清单的 CSV 路径，时间是相对于记录开头的秒数。每个原始 CSV 需有约 5120 Hz 的 `EMG_1` 至 `EMG_24`，开头 4 秒保持静息。推理只需肌电与标注好的动作区间。

预训练模型支持 `traj4-20` 和 `traj4-40`，分别对应独立第四组指令的 20% 和 40% MVC 保持段。同一次比较选择一个协议。两天的 MVC 各自测定。用户 04 的 40% 示例可这样准备：

```powershell
Copy-Item results/force/change/visits.csv deploy/force_change/input/
Copy-Item data/raw/EMG_data_for_putemgforce/emg_force_04/*repeats_long*.csv deploy/force_change/input/
```

## 3. 运行与查看

在 VS Code 右键本目录的 `docker-compose.yml`，选择 **Compose Up**。也可以从项目根目录运行：

```powershell
docker compose -f deploy/force_change/docker-compose.yml run --rm predict
```

本地 Python 使用相同入口：

```powershell
.\.venv\Scripts\python.exe -m emginsight.force.change_predict --trials deploy/force_change/input/visits.csv --output-dir deploy/force_change/output
```

| output 文件 | 用途 |
| --- | --- |
| `patient_result.txt` | 对外展示的一行，含正负号和单位 |
| `trials.csv` | 每次动作的记录、起止时间、十路中间特征和融合预测 |
| `daily.csv` | 每天的重复次数和预测中位数 |
| `force_change.png` | 两日的逐次估计及中位数连线 |
| `summary.json` | 人员编号、动作协议、日期、变化值、单位和两阶段校验值 |

## 4. 使用个人牛顿标定

公共模型输出 putEMG 原始刻度。采到配对测力计读数后，可按[个人标定清单格式](../../src/emginsight/force/README.md#准备两份清单)准备 `input/calibration.csv`。在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m emginsight.force.change fit --trials deploy/force_change/input/calibration.csv --calibration deploy/force_change/input/personal.joblib
docker compose -f deploy/force_change/docker-compose.yml run --rm predict --trials /data/input/visits.csv --model /data/input/personal.joblib --card /data/input/personal.card.json --output-dir /data/output/personal
```

个人模型携带测力计标定的 N 单位、适用人员、协议和留出日误差；推理会核对这些条件。两份输入 CSV 的患者编号与协议须匹配。每次比较只使用一个模型，输出单位由模型文件决定。
