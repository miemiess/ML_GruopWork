# D 任务指力模型的批处理预测镜像。在项目根目录构建：
#   docker build -f deploy/force/Dockerfile -t emginsight-force .
# 构建前需已有训练好的 results/runs/force/force_model.joblib（运行实验或从网盘下载）。
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY deploy/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

# 模型、校验信息和漂移参考固定在镜像内；predict 读取时核对 SHA-256 与 scikit-learn 版本。
COPY results/runs/force/force_model.joblib /app/model/force_model.joblib
COPY results/force/regression/model_card.json results/force/regression/monitoring_reference.json /app/model/
# 整只手三档力量模型（静息 / 轻 / 重），与回归模型共用同一批特征。
COPY results/runs/force/force_level_model.joblib /app/model/force_level_model.joblib
COPY results/force/levels/model_card.json /app/model/level_model_card.json

RUN useradd --create-home --uid 1000 app
USER app

ENTRYPOINT ["python", "-m", "emginsight.force.predict", "--model", "/app/model/force_model.joblib", "--level-model", "/app/model/force_level_model.joblib"]
CMD ["--help"]
