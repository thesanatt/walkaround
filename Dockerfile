# Replay-only deployment of Walkaround: serves the committed sample pair, makes no model call.
FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir uv && uv pip install --system --no-cache -e .
ENV PICKUP_RUNS_PER_DAY=0 PAIR_RUNS_PER_DAY=0 RUN_RETENTION_HOURS=8760 PYTHONUNBUFFERED=1
CMD ["sh", "-c", "walkaround serve --host 0.0.0.0 --port ${PORT:-8000}"]
