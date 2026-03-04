"""Serviço de busca textual via SQLite FTS5."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import aiosqlite

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Representa um fragmento recuperado pelo FTS5."""

    id: int
    tribunal: str
    numero_processo: str
    ementa: str
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
    safe_query = " OR ".join(tokens)  # OR para maximizar recall em buscas por linguagem natural
    logger.info("FTS5 query: '%s' | top_k=%d", safe_query, top_k)

    sql = f"""
        SELECT j.id, j.tribunal, j.numero_processo, j.ementa, f.rank
        FROM jurisprudencia_fts('{safe_query}') AS f
        JOIN jurisprudencia j ON j.id = f.rowid
        ORDER BY f.rank
        LIMIT ?
    """
    cursor = await conn.execute(sql, (top_k,))
    rows = await cursor.fetchall()
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
