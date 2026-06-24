"""
IAJuris — Ponto de entrada da aplicação.

Inicializa a instância FastAPI, registra os routers e
configura o ciclo de vida (lifespan) da aplicação:
  startup  → executa suite de testes, abre conexão SQLite, cria schema FTS5.
  shutdown → fecha conexão SQLite de forma limpa.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware

_BASE_DIR = Path(__file__).parent

from src.api.limiter import limiter
from src.api.routes import health as health_router
from src.api.routes import metrics as metrics_router
from src.api.routes import query as query_router
from src.config.logging_config import setup_logging
from src.config.settings import get_settings
from src.database.connection import close_db, init_db, open_db

setup_logging(debug=get_settings().debug)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Execução da suite de testes no startup
# ---------------------------------------------------------------------------


async def _warmup_models() -> None:
    """
    Pré-carrega modelos ML e cache de embeddings antes da primeira query.

    Sem warm-up, a primeira requisição carrega o MiniLM (~120 MB) + CrossEncoder
    (~120 MB) + 165k vetores do SQLite — latência de 20–30 s visível ao usuário.
    Executado no startup, esse custo fica invisível.
    """
    import src.database.connection as _conn_mod
    from src.services import semantic_service, rerank_service

    conn = _conn_mod._db
    if conn is not None:
        logger.info("Warm-up: carregando cache de embeddings do banco...")
        await semantic_service._load_acordao_cache(conn)
        await semantic_service._load_teses_cache(conn)

    logger.info("Warm-up: carregando modelo de embeddings (MiniLM)...")
    await asyncio.to_thread(semantic_service._get_model)

    logger.info("Warm-up: carregando cross-encoder...")
    await asyncio.to_thread(rerank_service._get_model)

    logger.info("Warm-up concluído — primeira query sem latência de cold start.")


async def _run_tests() -> bool:
    """
    Executa a suite de testes via pytest como subprocesso e loga cada linha.

    Retorna True se todos os testes passaram (exit code 0), False caso contrário.
    """
    logger.info("━" * 55)
    logger.info("STARTUP — Executando suite de testes (pytest tests/)")
    logger.info("━" * 55)

    _TEST_TIMEOUT = 300.0  # 5 minutos
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "--no-header",
        "-p", "no:cacheprovider",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=_TEST_TIMEOUT)
    except asyncio.TimeoutError:
        proc.kill()
        logger.error("STARTUP — Testes: timeout após %.0fs. Processo encerrado.", _TEST_TIMEOUT)
        return False

    for line in stdout.decode(errors="replace").splitlines():
        if line.strip():
            logger.info("[pytest] %s", line)

    logger.info("━" * 55)
    if proc.returncode == 0:
        logger.info("STARTUP — Testes: ✓ TODOS PASSARAM (exit 0)")
    else:
        logger.warning("STARTUP — Testes: ✗ FALHAS DETECTADAS (exit %d)", proc.returncode)
    logger.info("━" * 55)

    return proc.returncode == 0


# ---------------------------------------------------------------------------
# Ciclo de vida (substitui os deprecated @app.on_event)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia startup e shutdown da aplicação de forma assíncrona.

    Startup:
        1. Executa a suite de testes completa e loga os resultados.
        2. Abre a conexão SQLite com PRAGMAs de performance.
        3. Cria as tabelas (jurisprudencia + FTS5) se não existirem.

    Shutdown:
        1. Fecha a conexão SQLite de forma limpa.
    """
    # ── Startup ──────────────────────────────────────────────────────
    logger.info("Iniciando IAJuris...")
    if get_settings().run_tests_on_startup:
        await _run_tests()
    else:
        logger.info("Testes no startup desabilitados (RUN_TESTS_ON_STARTUP=false).")
    await open_db()
    await init_db()
    await _warmup_models()
    from src.services import metrics_service
    metrics_service.load_history()
    logger.info("Banco de dados pronto. Servidor disponível.")

    yield  # aplicação em execução

    # ── Shutdown ─────────────────────────────────────────────────────
    logger.info("Encerrando IAJuris...")
    await close_db()
    logger.info("Encerramento concluído.")


# ---------------------------------------------------------------------------
# Instância principal
# ---------------------------------------------------------------------------

app = FastAPI(
    title="IAJuris",
    description=(
        "Sistema RAG para consulta de jurisprudência brasileira "
        "(STF / STJ) com LLM local via Ollama."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

# ---------------------------------------------------------------------------
# Security headers
# ---------------------------------------------------------------------------


class _SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response


app.add_middleware(_SecurityHeadersMiddleware)

# ---------------------------------------------------------------------------
# Registro de routers
# ---------------------------------------------------------------------------

app.include_router(health_router.router, prefix="/api/v1", tags=["Health"])
app.include_router(query_router.router, prefix="/api/v1", tags=["Consulta RAG"])
app.include_router(metrics_router.router, prefix="/api/v1", tags=["Métricas"])

# ---------------------------------------------------------------------------
# Frontend — interface web
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=str(_BASE_DIR / "static")), name="static")


@app.get("/", include_in_schema=False)
async def serve_frontend() -> FileResponse:
    return FileResponse(_BASE_DIR / "static" / "index.html")


@app.get("/metrics", include_in_schema=False)
async def serve_metrics_dashboard() -> FileResponse:
    return FileResponse(_BASE_DIR / "static" / "metrics.html")

# ---------------------------------------------------------------------------
# Execução direta (desenvolvimento)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    _settings = get_settings()
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=_settings.debug)
