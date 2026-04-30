FROM python:3.12.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models_cache \
    TRANSFORMERS_CACHE=/app/models_cache \
    SENTENCE_TRANSFORMERS_HOME=/app/models_cache \
    LLM_PROVIDER=groq \
    DEBUG=false \
    RUN_TESTS_ON_STARTUP=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN grep -v "^locust" requirements.txt > requirements.prod.txt \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.prod.txt \
    && rm -rf /root/.cache/pip

COPY download_models.py .
RUN python download_models.py \
    && rm download_models.py

COPY . .
RUN mkdir -p /app/data/db

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" \
    || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]