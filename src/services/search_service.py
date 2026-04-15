"""Serviço de busca textual via SQLite FTS5."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import aiosqlite

from src.services.query_expansion import expand_query

_DB_QUERY_TIMEOUT = 5.0  # segundos

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Representa um acórdão recuperado pelo FTS5 (tabela jurisprudencia)."""

    id: int
    tribunal: str
    numero_processo: str
    ementa: str
    rank: float


@dataclass
class TesesResult:
    """Representa uma tese STJ recuperada pelo FTS5 (tabela teses_stj)."""

    id: int
    area: str
    edicao_num: int
    edicao_titulo: str
    tese_num: int
    tese_texto: str
    julgados: str
    rank: float


async def search(
    conn: aiosqlite.Connection,
    query: str,
    top_k: int = 5,
) -> list[SearchResult]:
    """
    Busca os ``top_k`` fragmentos mais relevantes para ``query`` via BM25.

    Retorna lista vazia se ``query`` for vazia.
    """
    if not query.strip():
        logger.debug("Query vazia — busca ignorada.")
        return []

    import re as _re
    # Stopwords do português — removidas para não restringir demais o AND implícito do FTS5
    _STOPWORDS = {
        "o", "a", "os", "as", "um", "uma", "de", "do", "da", "dos", "das",
        "em", "no", "na", "nos", "nas", "por", "para", "com", "que", "se",
        "ao", "aos", "à", "às", "e", "é", "ou", "mas", "como", "mais",
        "seu", "sua", "seus", "suas", "me", "te", "lhe", "nos", "não",
        "isso", "isto", "aqui", "ali", "já", "também", "segundo",
    }
    tokens = [
        t for t in _re.sub(r'["\'\'\(\)\*\^\-\+:,\.!?]', ' ', query).lower().split()
        if t not in _STOPWORDS and len(t) > 2
    ]
    if not tokens:
        return []
    expanded = expand_query(tokens)
    if expanded:
        logger.debug("Query expansion acórdãos: +%d termos: %s", len(expanded), expanded)
        tokens = tokens + expanded
    safe_query = " OR ".join(tokens)  # OR para maximizar recall em buscas por linguagem natural
    logger.info("FTS5 query: '%s' | top_k=%d", safe_query, top_k)

    sql = """
        SELECT j.id, j.tribunal, j.numero_processo, j.ementa, jurisprudencia_fts.rank
        FROM jurisprudencia_fts
        JOIN jurisprudencia j ON j.id = jurisprudencia_fts.rowid
        WHERE jurisprudencia_fts MATCH ?
        ORDER BY jurisprudencia_fts.rank
        LIMIT ?
    """
    try:
        cursor = await asyncio.wait_for(
            conn.execute(sql, (safe_query, top_k)), timeout=_DB_QUERY_TIMEOUT
        )
        rows = await asyncio.wait_for(cursor.fetchall(), timeout=_DB_QUERY_TIMEOUT)
    except asyncio.TimeoutError:
        logger.error("FTS5 acórdãos: timeout após %.1fs.", _DB_QUERY_TIMEOUT)
        return []
    logger.info("FTS5 retornou %d resultado(s).", len(rows))
    return [
        SearchResult(
            id=row["id"],
            tribunal=row["tribunal"],
            numero_processo=row["numero_processo"] or "",
            ementa=row["ementa"] or "",
            rank=row["rank"],
        )
        for row in rows
    ]


async def search_teses(
    conn: aiosqlite.Connection,
    query: str,
    top_k: int = 3,
) -> list[TesesResult]:
    """
    Busca as ``top_k`` teses STJ mais relevantes para ``query`` via BM25.

    Retorna lista vazia se ``query`` for vazia ou tabela estiver vazia.
    """
    if not query.strip():
        return []

    import re as _re

    # Verifica se a tabela teses_stj existe e tem dados
    try:
        cur_chk = await conn.execute("SELECT COUNT(*) FROM teses_stj")
        count = (await cur_chk.fetchone())[0]
        if count == 0:
            logger.debug("teses_stj está vazia — busca de teses ignorada.")
            return []
    except Exception:
        logger.debug("teses_stj não encontrada — busca de teses ignorada.")
        return []

    _STOPWORDS = {
        "o", "a", "os", "as", "um", "uma", "de", "do", "da", "dos", "das",
        "em", "no", "na", "nos", "nas", "por", "para", "com", "que", "se",
        "ao", "aos", "à", "às", "e", "é", "ou", "mas", "como", "mais",
        "seu", "sua", "seus", "suas", "me", "te", "lhe", "nos", "não",
        "isso", "isto", "aqui", "ali", "já", "também", "segundo",
    }
    tokens = [
        t for t in _re.sub(r'["\'\'\(\)\*\^\-\+:,\.!?]', " ", query).lower().split()
        if t not in _STOPWORDS and len(t) > 2
    ]
    if not tokens:
        return []
    expanded = expand_query(tokens)
    if expanded:
        logger.debug("Query expansion teses: +%d termos: %s", len(expanded), expanded)
        tokens = tokens + expanded
    safe_query = " OR ".join(tokens)
    logger.info("FTS5 teses query: '%s' | top_k=%d", safe_query, top_k)

    sql = """
        SELECT t.id, t.area, t.edicao_num, t.edicao_titulo,
               t.tese_num, t.tese_texto, t.julgados, teses_stj_fts.rank
        FROM teses_stj_fts
        JOIN teses_stj t ON t.id = teses_stj_fts.rowid
        WHERE teses_stj_fts MATCH ?
        ORDER BY teses_stj_fts.rank
        LIMIT ?
    """
    try:
        cursor = await asyncio.wait_for(
            conn.execute(sql, (safe_query, top_k)), timeout=_DB_QUERY_TIMEOUT
        )
        rows = await asyncio.wait_for(cursor.fetchall(), timeout=_DB_QUERY_TIMEOUT)
    except asyncio.TimeoutError:
        logger.error("FTS5 teses: timeout após %.1fs.", _DB_QUERY_TIMEOUT)
        return []
    logger.info("FTS5 teses retornou %d resultado(s).", len(rows))
    return [
        TesesResult(
            id=row["id"],
            area=row["area"] or "",
            edicao_num=row["edicao_num"] or 0,
            edicao_titulo=row["edicao_titulo"] or "",
            tese_num=row["tese_num"] or 0,
            tese_texto=row["tese_texto"] or "",
            julgados=row["julgados"] or "",
            rank=row["rank"],
        )
        for row in rows
    ]
