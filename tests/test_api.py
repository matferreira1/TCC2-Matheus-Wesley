"""
Testes de integracao para os endpoints FastAPI (/api/v1/query).

Estrategia:
  - httpx.AsyncClient + ASGITransport para chamadas HTTP sem servidor real.
  - Startup/shutdown do banco mockados para nao tocar o disco.
  - rag_service.answer mockado para isolar o pipeline RAG.
  - get_db sobrescrito com banco SQLite em memoria.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from main import app
from src.database.connection import get_db
from src.services.rag_service import RagResponse
from src.services.search_service import SearchResult, TesesResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_search_result(numero: str = "HC 100001", ementa: str = "Habeas corpus.") -> SearchResult:
    return SearchResult(id=1, tribunal="STF", numero_processo=numero,
                        ementa=ementa, rank=-1.0)


def _make_tese_result() -> TesesResult:
    return TesesResult(
        id=1, area="DIREITO CIVIL", edicao_num=143,
        edicao_titulo="PLANO DE SAUDE", tese_num=1,
        tese_texto="O plano pode definir doencas, mas nao o tratamento.",
        julgados="REsp 123/SP", rank=-1.0,
    )


def _make_rag_response(
    answer: str = "Resposta juridica fundamentada.",
    with_tese: bool = False,
) -> RagResponse:
    teses = [_make_tese_result()] if with_tese else []
    return RagResponse(
        answer=answer,
        sources=[_make_search_result()],
        sources_teses=teses,
    )


# ---------------------------------------------------------------------------
# Fixture: cliente HTTP com banco e lifespan mockados
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def api_client(db):
    """
    Cria um AsyncClient apontando para o app FastAPI com:
      - open_db / init_db / close_db mockados (nao tocam disco).
      - get_db sobrescrito com o banco em memoria da fixture `db`.
    """

    async def _override_get_db():
        yield db

    app.dependency_overrides[get_db] = _override_get_db

    with (
        patch("main.open_db", new_callable=AsyncMock),
        patch("main.init_db", new_callable=AsyncMock),
        patch("main.close_db", new_callable=AsyncMock),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client

    app.dependency_overrides.pop(get_db, None)


# ===========================================================================
# Testes - validacao de entrada
# ===========================================================================


async def test_post_query_too_short_returns_422(api_client: AsyncClient) -> None:
    """Pergunta com menos de 10 caracteres deve retornar 422 Unprocessable Entity."""
    resp = await api_client.post("/api/v1/query", json={"question": "Oi?"})
    assert resp.status_code == 422


async def test_post_query_too_long_returns_422(api_client: AsyncClient) -> None:
    """Pergunta com mais de 1000 caracteres deve retornar 422."""
    resp = await api_client.post(
        "/api/v1/query", json={"question": "x" * 1001}
    )
    assert resp.status_code == 422


async def test_post_query_missing_field_returns_422(api_client: AsyncClient) -> None:
    """Requisicao sem o campo question deve retornar 422."""
    resp = await api_client.post("/api/v1/query", json={})
    assert resp.status_code == 422


# ===========================================================================
# Testes - resposta de sucesso
# ===========================================================================


async def test_post_query_success_returns_200(api_client: AsyncClient) -> None:
    """POST valido deve retornar 200 OK."""
    with patch(
        "src.services.rag_service.answer",
        new_callable=AsyncMock,
        return_value=_make_rag_response(),
    ):
        resp = await api_client.post(
            "/api/v1/query",
            json={"question": "Quais os fundamentos para negar habeas corpus?"},
        )
    assert resp.status_code == 200


async def test_post_query_response_has_answer_field(api_client: AsyncClient) -> None:
    """Resposta deve conter o campo answer como string."""
    with patch(
        "src.services.rag_service.answer",
        new_callable=AsyncMock,
        return_value=_make_rag_response(answer="HC nao pode ser sucedaneo."),
    ):
        resp = await api_client.post(
            "/api/v1/query",
            json={"question": "Quais os fundamentos para negar habeas corpus?"},
        )

    body = resp.json()
    assert "answer" in body
    assert body["answer"] == "HC nao pode ser sucedaneo."


async def test_post_query_response_has_sources_list(api_client: AsyncClient) -> None:
    """Resposta deve conter o campo sources como lista."""
    with patch(
        "src.services.rag_service.answer",
        new_callable=AsyncMock,
        return_value=_make_rag_response(),
    ):
        resp = await api_client.post(
            "/api/v1/query",
            json={"question": "Quais os fundamentos para negar habeas corpus?"},
        )

    body = resp.json()
    assert "sources" in body
    assert isinstance(body["sources"], list)


async def test_post_query_source_document_structure(api_client: AsyncClient) -> None:
    """Cada elemento de sources deve ter tribunal, numero_processo, ementa e tipo."""
    with patch(
        "src.services.rag_service.answer",
        new_callable=AsyncMock,
        return_value=_make_rag_response(),
    ):
        resp = await api_client.post(
            "/api/v1/query",
            json={"question": "Quais os fundamentos para negar habeas corpus?"},
        )

    body = resp.json()
    assert len(body["sources"]) >= 1
    source = body["sources"][0]
    for field in ("tribunal", "numero_processo", "ementa", "tipo"):
        assert field in source


async def test_post_query_acordao_tipo_field(api_client: AsyncClient) -> None:
    """Fontes do tipo acordao devem ter tipo='acordao'."""
    with patch(
        "src.services.rag_service.answer",
        new_callable=AsyncMock,
        return_value=_make_rag_response(with_tese=False),
    ):
        resp = await api_client.post(
            "/api/v1/query",
            json={"question": "Quais os fundamentos para negar habeas corpus?"},
        )

    body = resp.json()
    acordao_sources = [s for s in body["sources"] if s["tipo"] == "acordao"]
    assert len(acordao_sources) >= 1


async def test_post_query_tese_tipo_field(api_client: AsyncClient) -> None:
    """Fontes do tipo tese STJ devem ter tipo='tese_stj'."""
    with patch(
        "src.services.rag_service.answer",
        new_callable=AsyncMock,
        return_value=_make_rag_response(with_tese=True),
    ):
        resp = await api_client.post(
            "/api/v1/query",
            json={"question": "Quais os fundamentos para negar habeas corpus?"},
        )

    body = resp.json()
    tese_sources = [s for s in body["sources"] if s["tipo"] == "tese_stj"]
    assert len(tese_sources) >= 1


async def test_post_query_empty_sources_is_valid(api_client: AsyncClient) -> None:
    """Resposta com sources=[] (sem documentos recuperados) deve ser valida."""
    with patch(
        "src.services.rag_service.answer",
        new_callable=AsyncMock,
        return_value=RagResponse(answer="Nao encontrei informacao.", sources=[], sources_teses=[]),
    ):
        resp = await api_client.post(
            "/api/v1/query",
            json={"question": "Pergunta sem documentos relevantes na base?"},
        )

    assert resp.status_code == 200
    assert resp.json()["sources"] == []


# ===========================================================================
# Testes - tratamento de erros
# ===========================================================================


async def test_post_query_timeout_returns_504(api_client: AsyncClient) -> None:
    """TimeoutException no pipeline deve resultar em 504 Gateway Timeout."""
    import httpx

    with patch(
        "src.services.rag_service.answer",
        new_callable=AsyncMock,
        side_effect=httpx.TimeoutException("LLM timeout"),
    ):
        resp = await api_client.post(
            "/api/v1/query",
            json={"question": "Pergunta que vai causar timeout no pipeline?"},
        )

    assert resp.status_code == 504


async def test_post_query_internal_error_returns_500(api_client: AsyncClient) -> None:
    """Excecao generica no pipeline deve resultar em 500 Internal Server Error."""
    with patch(
        "src.services.rag_service.answer",
        new_callable=AsyncMock,
        side_effect=RuntimeError("Erro inesperado"),
    ):
        resp = await api_client.post(
            "/api/v1/query",
            json={"question": "Pergunta que vai causar erro interno no pipeline?"},
        )

    assert resp.status_code == 500


# ===========================================================================
# Testes - health check
# ===========================================================================


async def test_health_returns_200(api_client: AsyncClient) -> None:
    """GET /health deve retornar 200 OK."""
    resp = await api_client.get("/api/v1/health")
    assert resp.status_code == 200


async def test_health_response_structure(api_client: AsyncClient) -> None:
    """Resposta do health deve conter os campos esperados."""
    resp = await api_client.get("/api/v1/health")
    body = resp.json()
    for field in ("status", "uptime_seconds", "database", "llm_provider", "reranker_enabled"):
        assert field in body


async def test_health_database_ok(api_client: AsyncClient) -> None:
    """Com banco em memoria operacional, database deve ser 'ok'."""
    resp = await api_client.get("/api/v1/health")
    assert resp.json()["database"] == "ok"


async def test_health_status_ok_when_db_ok(api_client: AsyncClient) -> None:
    """Com banco operacional, status geral deve ser 'ok'."""
    resp = await api_client.get("/api/v1/health")
    assert resp.json()["status"] == "ok"
