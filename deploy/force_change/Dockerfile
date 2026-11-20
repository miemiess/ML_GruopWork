FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY deploy/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .
COPY results/runs/force/force_model.joblib results/runs/force/force_model.joblib
COPY results/runs/force/force_change_model.joblib results/runs/force/force_change_model.joblib
COPY results/force/change/model_card.json results/force/change/model_card.json
RUN useradd --create-home --uid 1000 app
USER app
ENTRYPOINT ["python", "-m", "emginsight.force.change_predict"]
CMD ["--trials", "/data/input/visits.csv", "--output-dir", "/data/output"]
