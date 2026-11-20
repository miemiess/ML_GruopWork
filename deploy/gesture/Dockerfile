# C 任务手势模型的批处理识别镜像。在项目根目录构建：
#   docker build -f deploy/gesture/Dockerfile -t emginsight-gesture .
# 构建前需已有训练好的 results/runs/uci_gesture/gesture_model.joblib（运行 C 实验即可生成）。
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY deploy/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

# 模型和校验信息固定在镜像内；predict 读取时核对 SHA-256 与 scikit-learn 版本。
COPY results/runs/uci_gesture/gesture_model.joblib /app/model/gesture_model.joblib
COPY results/uci_gesture/classification/model_card.json /app/model/

RUN useradd --create-home --uid 1000 app
USER app

ENTRYPOINT ["python", "-m", "emginsight.uci_gesture.predict", "--model", "/app/model/gesture_model.joblib"]
CMD ["--help"]
