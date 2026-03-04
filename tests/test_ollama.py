"""
Testes para o serviço de comunicação com o Ollama.

Estratégia: mockar o httpx.AsyncClient para não depender de
Ollama em execução durante os testes unitários.
"""

import pytest
from unittest.mock import AsyncMock, patch

# from src.services.ollama_service import generate, health_check


@pytest.mark.asyncio
async def test_generate_returns_string():
    """
    Verifica que generate() retorna uma string não vazia dado um prompt válido.

    TODO: implementar mock do httpx e chamar generate().
    """
    pytest.skip("ollama_service não implementado ainda")


@pytest.mark.asyncio
async def test_generate_raises_on_timeout():
    """
    Verifica que generate() lança exceção quando o Ollama não responde a tempo.

    TODO: simular TimeoutException no mock do httpx.
    """
    pytest.skip("ollama_service não implementado ainda")


@pytest.mark.asyncio
async def test_health_check_returns_true_on_200():
    """
    Verifica que health_check() retorna True quando Ollama responde 200.

    TODO: implementar mock retornando status 200.
    """
    pytest.skip("ollama_service não implementado ainda")
