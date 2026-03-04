"""
Endpoints relacionados à consulta RAG.

Rotas:
  POST /api/v1/query  — recebe uma pergunta e retorna a resposta gerada.
"""

import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException, status

import aiosqlite
from src.api.schemas.query_schema import QueryRequest, QueryResponse, SourceDocument
from src.database.connection import get_db
from src.services import rag_service

logger = logging.getLogger(__name__)
router = APIRouter()


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
async def handle_query(
    payload: QueryRequest,
    conn: aiosqlite.Connection = Depends(get_db),
) -> QueryResponse:
    """Handler principal do endpoint de consulta RAG."""
    logger.info("POST /query | pergunta='%s'", payload.question)
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

    return QueryResponse(
        answer=resp.answer,
        sources=[
            SourceDocument(
                tribunal=s.tribunal,
                numero_processo=s.numero_processo,
                ementa=s.ementa,
            )
            for s in resp.sources
        ],
    )
