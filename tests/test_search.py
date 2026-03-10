"""
Testes para o servico de busca FTS5 (search_service).

Estrategia: banco SQLite em memoria provido pela fixture `db` de conftest.py.
Todos os testes sao independentes — cada um recebe uma conexao isolada.
"""

from __future__ import annotations

import aiosqlite

from src.services.search_service import (
    SearchResult,
    TesesResult,
    search,
    search_teses,
)


# ===========================================================================
# Testes de search() — acordaos STF
# ===========================================================================


async def test_search_returns_results(db: aiosqlite.Connection) -> None:
    results = await search(db, "habeas corpus", top_k=5)
    assert len(results) >= 1


async def test_search_results_are_search_result_instances(db: aiosqlite.Connection) -> None:
    results = await search(db, "habeas corpus", top_k=5)
    for r in results:
        assert isinstance(r, SearchResult)


async def test_search_result_fields_populated(db: aiosqlite.Connection) -> None:
    results = await search(db, "prisao preventiva", top_k=5)
    assert len(results) >= 1
    first = results[0]
    assert first.numero_processo != ""
    assert first.ementa != ""
    assert first.tribunal == "STF"


async def test_search_empty_for_unknown_term(db: aiosqlite.Connection) -> None:
    results = await search(db, "xyzabcdefqwerty", top_k=5)
    assert results == []


async def test_search_respects_top_k(db: aiosqlite.Connection) -> None:
    results = await search(db, "habeas corpus prisao", top_k=2)
    assert len(results) <= 2


async def test_search_top_k_one_returns_single_result(db: aiosqlite.Connection) -> None:
    results = await search(db, "habeas corpus", top_k=1)
    assert len(results) == 1


async def test_search_empty_query_returns_empty(db: aiosqlite.Connection) -> None:
    results = await search(db, "", top_k=5)
    assert results == []


async def test_search_whitespace_only_returns_empty(db: aiosqlite.Connection) -> None:
    results = await search(db, "   ", top_k=5)
    assert results == []


async def test_search_stopwords_only_returns_empty(db: aiosqlite.Connection) -> None:
    results = await search(db, "de o a os as em", top_k=5)
    assert results == []


async def test_search_rank_is_negative_float(db: aiosqlite.Connection) -> None:
    results = await search(db, "habeas corpus", top_k=5)
    assert len(results) >= 1
    for r in results:
        assert isinstance(r.rank, float)
        assert r.rank < 0  # FTS5 BM25 retorna valores negativos


async def test_search_multiple_terms_increases_recall(db: aiosqlite.Connection) -> None:
    results_narrow  = await search(db, "habeas", top_k=10)
    results_broad   = await search(db, "habeas corpus prisao preventiva", top_k=10)
    assert len(results_broad) >= len(results_narrow)


async def test_search_servidor_publico(db: aiosqlite.Connection) -> None:
    results = await search(db, "servidor publico estabilidade", top_k=5)
    numeros = [r.numero_processo for r in results]
    assert "RE 100004" in numeros


async def test_search_supressao_instancia(db: aiosqlite.Connection) -> None:
    results = await search(db, "supressao instancia", top_k=5)
    numeros = [r.numero_processo for r in results]
    assert "HC 100005" in numeros


# ===========================================================================
# Testes de search_teses() — teses STJ
# ===========================================================================


async def test_search_teses_returns_results(db: aiosqlite.Connection) -> None:
    results = await search_teses(db, "plano saude", top_k=3)
    assert len(results) >= 1


async def test_search_teses_are_teses_result_instances(db: aiosqlite.Connection) -> None:
    results = await search_teses(db, "plano saude", top_k=3)
    for t in results:
        assert isinstance(t, TesesResult)


async def test_search_teses_result_fields_populated(db: aiosqlite.Connection) -> None:
    results = await search_teses(db, "plano saude cobertura", top_k=3)
    assert len(results) >= 1
    first = results[0]
    assert first.area != ""
    assert first.edicao_num > 0
    assert first.tese_texto != ""


async def test_search_teses_respects_top_k(db: aiosqlite.Connection) -> None:
    results = await search_teses(db, "plano saude consumidor penal", top_k=2)
    assert len(results) <= 2


async def test_search_teses_empty_query_returns_empty(db: aiosqlite.Connection) -> None:
    results = await search_teses(db, "", top_k=3)
    assert results == []


async def test_search_teses_unknown_term_returns_empty(db: aiosqlite.Connection) -> None:
    results = await search_teses(db, "xyzabcqwerty", top_k=3)
    assert results == []


async def test_search_teses_empty_table_returns_empty() -> None:
    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        await conn.execute(
            "CREATE TABLE teses_stj (id INTEGER PRIMARY KEY, tese_texto TEXT)"
        )
        await conn.commit()
        results = await search_teses(conn, "plano saude", top_k=3)
        assert results == []


async def test_search_teses_missing_table_returns_empty() -> None:
    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        results = await search_teses(conn, "plano saude", top_k=3)
        assert results == []


async def test_search_teses_prisao_preventiva(db: aiosqlite.Connection) -> None:
    results = await search_teses(db, "prisao preventiva cautelar", top_k=3)
    areas = [t.area for t in results]
    assert "DIREITO PENAL" in areas


async def test_search_teses_julgados_field_populated(db: aiosqlite.Connection) -> None:
    results = await search_teses(db, "plano saude", top_k=3)
    assert len(results) >= 1
    assert results[0].julgados != ""
