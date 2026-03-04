"""
Testes de integração para o pipeline RAG.

Estratégia: substituir search_service e ollama_service por mocks
para testar apenas a lógica de orquestração do rag_service.
"""

import pytest
from unittest.mock import AsyncMock, patch

# from src.services.rag_service import answer, RagResponse


@pytest.mark.asyncio
async def test_answer_returns_rag_response():
    """
    Verifica que answer() retorna um RagResponse com answer e sources.

    TODO: mockar search_service.search e ollama_service.generate.
    """
    pytest.skip("rag_service não implementado ainda")


@pytest.mark.asyncio
async def test_answer_uses_top_k_sources():
    """
    Verifica que o nº de fontes em RagResponse não excede settings.rag_top_k.

    TODO: implementar após rag_service estar funcional.
    """
    pytest.skip("rag_service não implementado ainda")


@pytest.mark.asyncio
async def test_answer_propagates_ollama_error():
    """
    Verifica que erros do Ollama são propagados corretamente pelo pipeline.

    TODO: simular falha no ollama_service e verificar exceção.
    """
    pytest.skip("rag_service não implementado ainda")
