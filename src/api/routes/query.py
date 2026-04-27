"""
Endpoints relacionados à consulta RAG.

Rotas:
  POST /api/v1/query  — recebe uma pergunta e retorna a resposta gerada.
"""

import logging
import re
import uuid

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status

import aiosqlite
from src.api.schemas.query_schema import QueryRequest, QueryResponse, SourceDocument
from src.database.connection import get_db
from src.services import rag_service
from src.api.limiter import limiter
from src.config.settings import get_settings as _get_settings

logger = logging.getLogger(__name__)

# Limite avaliado no startup — override via RATE_LIMIT_PER_MINUTE no .env
_RATE_LIMIT = f"{_get_settings().rate_limit_per_minute}/minute"
router = APIRouter()

# Padrões que indicam tentativa de prompt injection na pergunta
_INJECTION_RE = re.compile(
    r'(?i)(ignore|disregard|forget|bypass)\s.{0,40}(instruction|prompt|rule|directive)'
    r'|(system|admin)\s*(prompt|command|mode)',
)


def _check_injection(question: str) -> None:
    if _INJECTION_RE.search(question):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Pergunta contém padrões não permitidos.",
        )


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Consulta jurídica via RAG",
    description=(
        "Recebe uma pergunta em linguagem natural, recupera fragmentos "
        "relevantes da base jurídica (STF/STJ) e gera uma resposta "
        "fundamentada com o LLM local."
    ),
)
@limiter.limit(_RATE_LIMIT)
async def handle_query(
    request: Request,
    payload: QueryRequest,
    conn: aiosqlite.Connection = Depends(get_db),
) -> QueryResponse:
    """Handler principal do endpoint de consulta RAG."""
    request_id = str(uuid.uuid4())[:8]
    _check_injection(payload.question)
    logger.info(
        "POST /query | id=%s | pergunta='%s'",
        request_id, payload.question,
    )
    try:
        resp = await rag_service.answer(conn, payload.question)
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="O modelo LLM não respondeu a tempo. Tente novamente.",
        )
    except Exception as exc:
        logger.exception("Erro inesperado no pipeline RAG: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno ao processar a consulta.",
        )

    sources_sv = [
        SourceDocument(
            tribunal="STF",
            numero_processo=f"SV {sv.numero}",
            ementa=sv.enunciado,
            tipo="sumula_vinculante_stf",
        )
        for sv in resp.sources_sv
    ]
    sources_acordaos = [
        SourceDocument(
            tribunal=s.tribunal,
            numero_processo=s.numero_processo,
            ementa=s.ementa,
            tipo="acordao",
            orgao_julgador=s.orgao_julgador or None,
            repercussao_geral=s.repercussao_geral,
            data_julgamento=s.data_julgamento or None,
        )
        for s in resp.sources
    ]
    sources_teses = [
        SourceDocument(
            tribunal="STJ",
            numero_processo=f"Ed. {t.edicao_num} — Tese {t.tese_num}",
            ementa=t.tese_texto,
            tipo="tese_stj",
        )
        for t in resp.sources_teses
    ]
    return QueryResponse(
        answer=resp.answer,
        sources=sources_sv + sources_acordaos + sources_teses,
    )
