"""
Testes para o serviço de busca FTS5.

Estratégia: banco SQLite em memória (:memory:) para isolamento total —
sem dependência de arquivo em disco ou dados reais.
"""

import pytest
import pytest_asyncio
import aiosqlite

# from src.services.search_service import search, SearchResult


@pytest_asyncio.fixture
async def in_memory_db():
    """
    Fixture que cria um banco SQLite em memória com a tabela FTS5
    populada com registros de teste.

    TODO: executar DDL de criação da tabela FTS5 e inserir fixtures.
    """
    async with aiosqlite.connect(":memory:") as conn:
        yield conn


@pytest.mark.asyncio
async def test_search_returns_results(in_memory_db):
    """
    Verifica que uma busca por termo existente retorna ao menos 1 resultado.

    TODO: implementar após search_service estar funcional.
    """
    pytest.skip("search_service não implementado ainda")


@pytest.mark.asyncio
async def test_search_empty_for_unknown_term(in_memory_db):
    """
    Verifica que uma busca por termo inexistente retorna lista vazia.

    TODO: implementar após search_service estar funcional.
    """
    pytest.skip("search_service não implementado ainda")


@pytest.mark.asyncio
async def test_search_respects_top_k(in_memory_db):
    """
    Verifica que o número de resultados não excede ``top_k``.

    TODO: implementar após search_service estar funcional.
    """
    pytest.skip("search_service não implementado ainda")
