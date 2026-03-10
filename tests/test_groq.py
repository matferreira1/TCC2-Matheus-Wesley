"""
Testes para groq_service (cliente Groq Cloud API).

Estrategia: mock do AsyncGroq para nao depender de chave de API real.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import src.services.groq_service as groq_mod
from src.services import groq_service


@pytest.fixture(autouse=True)
def reset_groq_client():
    """Limpa o cliente Groq singleton entre testes para evitar contaminacao."""
    groq_mod._client = None
    yield
    groq_mod._client = None


def _make_completion(text: str = "Resposta juridica do Groq."):
    """Cria um objeto completion mock compativel com a SDK Groq."""
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message
    completion = MagicMock()
    completion.choices = [choice]
    return completion


# ===========================================================================
# Testes de _get_client()
# ===========================================================================


def test_get_client_raises_when_api_key_missing() -> None:
    """_get_client() deve lancar ValueError se GROQ_API_KEY nao estiver configurada."""
    with patch("src.services.groq_service.settings") as mock_settings:
        mock_settings.groq_api_key = ""
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            groq_mod._get_client()


def test_get_client_returns_async_groq_instance() -> None:
    """_get_client() deve retornar um AsyncGroq (ou mock) quando a chave esta presente."""
    mock_async_groq_cls = MagicMock()
    mock_instance = MagicMock()
    mock_async_groq_cls.return_value = mock_instance

    with (
        patch("src.services.groq_service.settings") as mock_settings,
        patch("src.services.groq_service.AsyncGroq", mock_async_groq_cls),
    ):
        mock_settings.groq_api_key = "fake-key-123"
        client = groq_mod._get_client()

    assert client is mock_instance


def test_get_client_is_singleton() -> None:
    """_get_client() deve retornar a mesma instancia em chamadas consecutivas."""
    mock_async_groq_cls = MagicMock()
    mock_instance = MagicMock()
    mock_async_groq_cls.return_value = mock_instance

    with (
        patch("src.services.groq_service.settings") as mock_settings,
        patch("src.services.groq_service.AsyncGroq", mock_async_groq_cls),
    ):
        mock_settings.groq_api_key = "fake-key-123"
        c1 = groq_mod._get_client()
        c2 = groq_mod._get_client()

    assert c1 is c2
    mock_async_groq_cls.assert_called_once()


# ===========================================================================
# Testes de generate()
# ===========================================================================


async def test_generate_returns_string() -> None:
    """generate() deve retornar uma string com o conteudo da resposta."""
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_make_completion("Habeas corpus denegado.")
    )

    with patch("src.services.groq_service._get_client", return_value=mock_client):
        result = await groq_service.generate("Prompt juridico")

    assert result == "Habeas corpus denegado."


async def test_generate_returns_empty_string_when_content_is_none() -> None:
    """generate() deve retornar string vazia (nao None) quando content=None."""
    completion = _make_completion(None)
    completion.choices[0].message.content = None

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=completion)

    with patch("src.services.groq_service._get_client", return_value=mock_client):
        result = await groq_service.generate("prompt")

    assert result == ""
    assert isinstance(result, str)


async def test_generate_propagates_api_error() -> None:
    """generate() deve propagar excecoes lanadas pela API Groq."""
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=RuntimeError("Groq API unavailable")
    )

    with patch("src.services.groq_service._get_client", return_value=mock_client):
        with pytest.raises(RuntimeError, match="Groq API unavailable"):
            await groq_service.generate("prompt")


async def test_generate_sends_correct_model() -> None:
    """generate() deve usar o modelo configurado em settings.groq_model."""
    mock_client = AsyncMock()
    mock_create = AsyncMock(return_value=_make_completion("ok"))
    mock_client.chat.completions.create = mock_create

    with (
        patch("src.services.groq_service._get_client", return_value=mock_client),
        patch("src.services.groq_service.settings") as mock_settings,
    ):
        mock_settings.groq_model = "llama-3.3-70b-versatile"
        mock_settings.rag_max_tokens = 2048
        mock_settings.groq_timeout_seconds = 60
        await groq_service.generate("prompt")

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["model"] == "llama-3.3-70b-versatile"


async def test_generate_uses_low_temperature() -> None:
    """generate() deve usar temperatura baixa (deterministica) para o dominio juridico."""
    mock_client = AsyncMock()
    mock_create = AsyncMock(return_value=_make_completion("ok"))
    mock_client.chat.completions.create = mock_create

    with (
        patch("src.services.groq_service._get_client", return_value=mock_client),
        patch("src.services.groq_service.settings") as mock_settings,
    ):
        mock_settings.groq_model = "llama-3.3-70b-versatile"
        mock_settings.rag_max_tokens = 2048
        mock_settings.groq_timeout_seconds = 60
        await groq_service.generate("prompt")

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["temperature"] <= 0.3


# ===========================================================================
# Testes de health_check()
# ===========================================================================


async def test_health_check_returns_true_when_api_accessible() -> None:
    """health_check() deve retornar True quando a API Groq esta acessivel."""
    mock_client = AsyncMock()
    mock_client.models.list = AsyncMock(return_value=MagicMock())

    with patch("src.services.groq_service._get_client", return_value=mock_client):
        result = await groq_service.health_check()

    assert result is True


async def test_health_check_returns_false_on_exception() -> None:
    """health_check() deve retornar False quando ocorre qualquer excecao."""
    mock_client = AsyncMock()
    mock_client.models.list = AsyncMock(side_effect=Exception("Connection error"))

    with patch("src.services.groq_service._get_client", return_value=mock_client):
        result = await groq_service.health_check()

    assert result is False


async def test_health_check_returns_false_when_key_missing() -> None:
    """health_check() deve retornar False silenciosamente se a chave estiver ausente."""
    with patch("src.services.groq_service.settings") as mock_settings:
        mock_settings.groq_api_key = ""
        result = await groq_service.health_check()

    assert result is False
