# =============================================================================
# IAJuris — Dockerfile
#
# Estratégia: multi-stage build
#   Stage 1 (builder) — instala deps + baixa modelos HuggingFace
#   Stage 2 (runtime) — imagem enxuta, sem ferramentas de build
#
# O banco iajuris.db é versionado via Git LFS em data/db/iajuris.db
# e copiado pelo COPY . . — nenhuma geração em runtime necessária.
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1 — builder
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models_cache \
    TRANSFORMERS_CACHE=/app/models_cache \
    SENTENCE_TRANSFORMERS_HOME=/app/models_cache

# Dependências de sistema mínimas para compilar pacotes nativos (numpy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Instala dependências de produção (exclui locust — apenas para testes de carga)
RUN pip install --no-cache-dir --upgrade pip \
    && grep -v "^locust" requirements.txt > requirements.prod.txt \
    && pip install --no-cache-dir -r requirements.prod.txt

# Baixa e cacheia os modelos HuggingFace durante o BUILD.
# Sem isso, o container baixaria ~400 MB a cada cold start na plataforma PaaS.
COPY download_models.py .
RUN python download_models.py

# -----------------------------------------------------------------------------
# Stage 2 — runtime
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models_cache \
    TRANSFORMERS_CACHE=/app/models_cache \
    SENTENCE_TRANSFORMERS_HOME=/app/models_cache \
    LLM_PROVIDER=groq \
    DEBUG=false \
    RUN_TESTS_ON_STARTUP=false

COPY --from=builder /usr/local/lib/python3.12/site-packages \
                    /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY --from=builder /app/models_cache /app/models_cache

# Copia o código-fonte completo (inclui data/db/iajuris.db via Git LFS)
COPY . .

# Garante que o diretório do banco existe mesmo se o .db não estiver presente
RUN mkdir -p /app/data/db

EXPOSE 8000

# Health check — Railway/Render/Fly.io aguardam status 200 antes de rotear tráfego.
# start-period=60s dá tempo para o lifespan (init_db + eventuais testes) completar.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" \
    || exit 1

CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]