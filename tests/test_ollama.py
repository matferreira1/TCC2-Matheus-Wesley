"""
Testes para ollama_service (cliente HTTP para o Ollama local).

Estrategia: mock do httpx.AsyncClient para nao depender do Ollama em execucao.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.services import ollama_service


# ---------------------------------------------------------------------------
# Helpers: context managers mock para httpx streaming
# ---------------------------------------------------------------------------


def _make_stream_client(lines: list[str], raise_on_enter=None):
    """Constroi um mock de httpx.AsyncClient que simula streaming de linhas."""

    async def _aiter_lines():
        for line in lines:
            yield line

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.aiter_lines = _aiter_lines
    mock_resp.__aenter__ = AsyncMock(
        return_value=mock_resp if raise_on_enter is None else None,
        side_effect=raise_on_enter,
    )
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    return mock_client


# ===========================================================================
# Testes de generate()
# ===========================================================================


async def test_generate_returns_string() -> None:
    """generate() deve concatenar os chunks e retornar uma string."""
    lines = [
        json.dumps({"response": "Habeas ", "done": False}),
        json.dumps({"response": "corpus ", "done": False}),
        json.dumps({"response": "concedido.", "done": True}),
    ]
    mock_client = _make_stream_client(lines)

    with patch("src.services.ollama_service.httpx.AsyncClient", return_value=mock_client):
        result = await ollama_service.generate("Prompt de teste")

    assert result == "Habeas corpus concedido."


async def test_generate_returns_non_empty_string() -> None:
    """generate() nao deve retornar string vazia quando o modelo gera tokens."""
    lines = [json.dumps({"response": "Resposta juridica.", "done": True})]
    mock_client = _make_stream_client(lines)

    with patch("src.services.ollama_service.httpx.AsyncClient", return_value=mock_client):
        result = await ollama_service.generate("prompt")

    assert len(result) > 0


async def test_generate_skips_empty_lines() -> None:
    """Linhas vazias no stream devem ser ignoradas sem erro."""
    lines = [
        "",  # linha vazia
        json.dumps({"response": "Token1", "done": False}),
        "",
        json.dumps({"response": "Token2", "done": True}),
    ]
    mock_client = _make_stream_client(lines)

    with patch("src.services.ollama_service.httpx.AsyncClient", return_value=mock_client):
        result = await ollama_service.generate("prompt")

    assert "Token1" in result
    assert "Token2" in result


async def test_generate_stops_at_done_true() -> None:
    """generate() deve parar de ler o stream apos done=True."""
    # Chunk extra apos done=True nao deve aparecer no resultado
    chunks_received = []

    async def _aiter_lines():
        yield json.dumps({"response": "Parte1 ", "done": False})
        yield json.dumps({"response": "Parte2.", "done": True})
        # Este nao deve ser processado
        yield json.dumps({"response": "EXTRA", "done": False})

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.aiter_lines = _aiter_lines
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("src.services.ollama_service.httpx.AsyncClient", return_value=mock_client):
        result = await ollama_service.generate("prompt")

    assert "EXTRA" not in result
    assert "Parte1" in result
    assert "Parte2" in result


async def test_generate_raises_on_read_timeout() -> None:
    """generate() deve propagar httpx.ReadTimeout quando o modelo nao responde."""
    mock_resp = MagicMock()
    mock_resp.__aenter__ = AsyncMock(
        side_effect=httpx.ReadTimeout("timeout", request=MagicMock())
    )
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("src.services.ollama_service.httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(httpx.ReadTimeout):
            await ollama_service.generate("prompt")


async def test_generate_raises_on_http_error() -> None:
    """generate() deve propagar HTTPStatusError em caso de erro HTTP (4xx/5xx)."""
    mock_resp = MagicMock()
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_resp.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock()
        )
    )
    mock_resp.aiter_lines = AsyncMock(return_value=iter([]))

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("src.services.ollama_service.httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(httpx.HTTPStatusError):
            await ollama_service.generate("prompt")


# ===========================================================================
# Testes de health_check()
# ===========================================================================


async def test_health_check_returns_true_on_200() -> None:
    """health_check() deve retornar True quando o Ollama responde com 200."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("src.services.ollama_service.httpx.AsyncClient", return_value=mock_client):
        result = await ollama_service.health_check()

    assert result is True


async def test_health_check_returns_false_on_non_200() -> None:
    """health_check() deve retornar False quando o Ollama responde com status != 200."""
    mock_resp = MagicMock()
    mock_resp.status_code = 503

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("src.services.ollama_service.httpx.AsyncClient", return_value=mock_client):
        result = await ollama_service.health_check()

    assert result is False


async def test_health_check_returns_false_on_connection_error() -> None:
    """health_check() deve retornar False em caso de erro de conexao."""
    mock_client = MagicMock()
    mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("src.services.ollama_service.httpx.AsyncClient", return_value=mock_client):
        result = await ollama_service.health_check()

    assert result is False
