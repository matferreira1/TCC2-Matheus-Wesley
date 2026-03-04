"""
IAJuris — Ponto de entrada da aplicação.

Inicializa a instância FastAPI, registra os routers e
configura o ciclo de vida (lifespan) da aplicação:
  startup  → abre conexão SQLite, cria schema FTS5.
  shutdown → fecha conexão SQLite de forma limpa.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import query as query_router
from src.config.logging_config import setup_logging
from src.config.settings import get_settings
from src.database.connection import close_db, init_db, open_db

setup_logging(debug=get_settings().debug)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ciclo de vida (substitui os deprecated @app.on_event)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia startup e shutdown da aplicação de forma assíncrona.

    Startup:
        1. Garante que o diretório data/db/ existe.
        2. Abre a conexão SQLite com PRAGMAs de performance.
        3. Cria as tabelas (jurisprudencia + FTS5) se não existirem.

    Shutdown:
        1. Fecha a conexão SQLite de forma limpa.
    """
    # ── Startup ──────────────────────────────────────────────────────
    logger.info("Iniciando IAJuris...")
    await open_db()
    await init_db()
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
# Registro de routers
# ---------------------------------------------------------------------------

app.include_router(query_router.router, prefix="/api/v1", tags=["Consulta RAG"])

# ---------------------------------------------------------------------------
# Execução direta (desenvolvimento)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
