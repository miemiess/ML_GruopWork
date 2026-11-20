FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY deploy/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .
COPY results/runs/fatigue/full_dataset/fatigue_model.joblib /app/model/fatigue_model.joblib
COPY results/fatigue/full_dataset/model_card.json /app/model/model_card.json
RUN useradd --create-home --uid 1000 app
USER app
ENTRYPOINT ["python","-m","emginsight.fatigue.predict","--model","/app/model/fatigue_model.joblib","--card","/app/model/model_card.json"]
CMD ["--help"]
