"""
Endpoint de health check.

Rota:
  GET /api/v1/health  — retorna status da aplicação e do banco de dados.
"""

import logging
import time

import aiosqlite
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.config.settings import get_settings
from src.database.connection import get_db

logger = logging.getLogger(__name__)
router = APIRouter()

_START_TIME = time.time()


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    database: str
    llm_provider: str
    reranker_enabled: bool


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Verifica se a aplicação e o banco de dados estão operacionais.",
)
async def health_check(
    conn: aiosqlite.Connection = Depends(get_db),
) -> HealthResponse:
    settings = get_settings()

    try:
        await conn.execute("SELECT 1")
        db_status = "ok"
    except Exception as exc:
        logger.warning("Health check — banco indisponível: %s", exc)
        db_status = "unavailable"

    return HealthResponse(
        status="ok" if db_status == "ok" else "degraded",
        uptime_seconds=round(time.time() - _START_TIME, 1),
        database=db_status,
        llm_provider=settings.llm_provider,
        reranker_enabled=settings.reranker_enabled,
    )
