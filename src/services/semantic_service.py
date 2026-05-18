"""
Busca semântica por similaridade cosseno + Reciprocal Rank Fusion (RRF).

Estratégia:
  1. Embeddings gerados offline (etl/generate_embeddings.py) e armazenados
     como BLOB float16 nas colunas `embedding` de jurisprudencia e teses_stj.
  2. Na primeira chamada, apenas id+embedding são carregados em memória como
     arrays NumPy (~121 MB para 165k acórdãos). Textos das ementas NÃO ficam
     em cache — são buscados por ID só para os top-k resultados (≤20 docs).
  3. A busca é uma multiplicação matricial (cosine similarity) — <1 ms.
  4. RRF funde o ranking lexical (FTS5) com o semântico em lista única.

Fallback: se nenhum embedding estiver no banco, retorna lista vazia e o
sistema continua funcionando apenas com FTS5.
"""

from __future__ import annotations

import logging
import struct
from typing import TypeVar

import aiosqlite
import numpy as np

from src.services.search_service import SearchResult, TesesResult, SumulaVinculanteResult

logger = logging.getLogger(__name__)

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Singletons — carregados na primeira chamada
_model = None
# Apenas id + vetor em cache; textos buscados por ID on-demand para top-k
_acordao_cache: tuple[list[int], np.ndarray] | None = None
_teses_cache: tuple[list[int], list[tuple], np.ndarray] | None = None
_sv_cache: tuple[list[int], list[tuple], np.ndarray] | None = None

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Carregando modelo de embeddings (%s)...", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("Modelo semântico pronto.")
    return _model


def _embed_query(query: str) -> np.ndarray:
    return _get_model().encode(query, normalize_embeddings=True)


# ---------------------------------------------------------------------------
# Serialização
# ---------------------------------------------------------------------------


_EMBEDDING_DIMS = 384                         # paraphrase-multilingual-MiniLM-L12-v2
_BLOB_BYTES_F16 = _EMBEDDING_DIMS * 2         # float16 = 2 bytes (padrão atual)
_BLOB_BYTES_F32 = _EMBEDDING_DIMS * 4         # float32 = 4 bytes (legado)


def _deserialize(blob: bytes) -> np.ndarray:
    if len(blob) == _BLOB_BYTES_F16:
        return np.frombuffer(blob, dtype=np.float16).copy()
    if len(blob) == _BLOB_BYTES_F32:
        return np.frombuffer(blob, dtype=np.float32).copy()
    raise ValueError(
        f"Tamanho de embedding inválido: {len(blob)} bytes "
        f"(esperado {_BLOB_BYTES_F16} float16 ou {_BLOB_BYTES_F32} float32)."
    )


# ---------------------------------------------------------------------------
# Cache em memória
# ---------------------------------------------------------------------------



async def _load_acordao_cache(
    conn: aiosqlite.Connection,
) -> tuple[list[int], np.ndarray] | None:
    """Carrega apenas id+embedding em memória (sem textos) para economizar RAM."""
    global _acordao_cache
    if _acordao_cache is not None:
        return _acordao_cache

    cur = await conn.execute(
        "SELECT id, embedding FROM jurisprudencia WHERE embedding IS NOT NULL"
    )
    rows = await cur.fetchall()
    if not rows:
        logger.warning(
            "Nenhum embedding de acórdão encontrado. "
            "Execute: python -m etl.generate_embeddings"
        )
        return None

    ids = [r[0] for r in rows]
    vectors = np.stack([_deserialize(r[1]) for r in rows])  # (N, 384)

    _acordao_cache = (ids, vectors)
    logger.info("Cache semântico: %d acórdãos carregados (só vetores, sem texto).", len(ids))
    return _acordao_cache


async def _fetch_acordao_meta_batch(
    conn: aiosqlite.Connection,
    doc_ids: list[int],
) -> dict[int, tuple]:
    """Busca metadados completos (incluindo ementa) para uma lista de IDs."""
    if not doc_ids:
        return {}
    placeholders = ",".join("?" * len(doc_ids))
    cur = await conn.execute(
        f"SELECT id, tribunal, numero_processo, ementa, orgao_julgador, repercussao_geral, data_julgamento "
        f"FROM jurisprudencia WHERE id IN ({placeholders})",
        doc_ids,
    )
    rows = await cur.fetchall()
    return {
        r[0]: (r[1] or "", r[2] or "", r[3] or "", r[4] or "", bool(r[5]), r[6] or "")
        for r in rows
    }


async def _load_teses_cache(
    conn: aiosqlite.Connection,
) -> tuple[list[int], list[tuple], np.ndarray] | None:
    global _teses_cache
    if _teses_cache is not None:
        return _teses_cache

    cur = await conn.execute(
        "SELECT id, area, edicao_num, edicao_titulo, tese_num, tese_texto, julgados, embedding "
        "FROM teses_stj WHERE embedding IS NOT NULL"
    )
    rows = await cur.fetchall()
    if not rows:
        return None

    ids = [r[0] for r in rows]
    meta = [(r[1] or "", r[2] or 0, r[3] or "", r[4] or 0, r[5] or "", r[6] or "") for r in rows]
    vectors = np.stack([_deserialize(r[7]) for r in rows])  # (N, 384)

    _teses_cache = (ids, meta, vectors)
    logger.info("Cache semântico: %d teses/súmulas carregadas.", len(ids))
    return _teses_cache


async def _load_sv_cache(
    conn: aiosqlite.Connection,
) -> tuple[list[int], list[tuple], np.ndarray] | None:
    global _sv_cache
    if _sv_cache is not None:
        return _sv_cache

    try:
        cur = await conn.execute(
            "SELECT id, numero, enunciado, embedding "
            "FROM sumulas_vinculantes_stf WHERE embedding IS NOT NULL"
        )
    except Exception:
        return None

    rows = await cur.fetchall()
    if not rows:
        return None

    ids = [r[0] for r in rows]
    meta = [(r[1] or 0, r[2] or "") for r in rows]
    vectors = np.stack([_deserialize(r[3]) for r in rows])

    _sv_cache = (ids, meta, vectors)
    logger.info("Cache semântico: %d súmulas vinculantes carregadas.", len(ids))
    return _sv_cache


def clear_cache() -> None:
    """Invalida caches em memória (usar após re-execução do ETL)."""
    global _acordao_cache, _teses_cache, _sv_cache
    _acordao_cache = None
    _teses_cache = None
    _sv_cache = None
    logger.info("Caches semânticos invalidados.")


# ---------------------------------------------------------------------------
# Busca semântica
# ---------------------------------------------------------------------------


def _top_k_by_cosine(
    query_vec: np.ndarray,
    vectors: np.ndarray,
    top_k: int,
) -> list[tuple[int, float]]:
    """Retorna lista de (índice, score) dos top_k por similaridade cosseno."""
    scores = vectors @ query_vec  # vetores já normalizados → dot = cosine
    k = min(top_k, len(scores))
    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    return [(int(i), float(scores[i])) for i in top_idx]


async def search_semantic(
    conn: aiosqlite.Connection,
    query: str,
    top_k: int = 15,
) -> list[SearchResult]:
    """Retorna top_k acórdãos por similaridade semântica com a query."""
    cache = await _load_acordao_cache(conn)
    if cache is None:
        return []

    ids, vectors = cache
    query_vec = _embed_query(query)
    top = _top_k_by_cosine(query_vec, vectors, top_k)

    top_ids = [ids[idx] for idx, _ in top]
    scores = {ids[idx]: score for idx, score in top}
    meta_map = await _fetch_acordao_meta_batch(conn, top_ids)

    return [
        SearchResult(
            id=doc_id,
            tribunal=meta_map[doc_id][0],
            numero_processo=meta_map[doc_id][1],
            ementa=meta_map[doc_id][2],
            orgao_julgador=meta_map[doc_id][3],
            repercussao_geral=meta_map[doc_id][4],
            data_julgamento=meta_map[doc_id][5],
            rank=-scores[doc_id],
        )
        for doc_id in top_ids
        if doc_id in meta_map
    ]


async def search_teses_semantic(
    conn: aiosqlite.Connection,
    query: str,
    top_k: int = 10,
) -> list[TesesResult]:
    """Retorna top_k teses/súmulas por similaridade semântica com a query."""
    cache = await _load_teses_cache(conn)
    if cache is None:
        return []

    ids, meta, vectors = cache
    query_vec = _embed_query(query)
    top = _top_k_by_cosine(query_vec, vectors, top_k)

    return [
        TesesResult(
            id=ids[idx],
            area=meta[idx][0],
            edicao_num=meta[idx][1],
            edicao_titulo=meta[idx][2],
            tese_num=meta[idx][3],
            tese_texto=meta[idx][4],
            julgados=meta[idx][5],
            rank=-score,
        )
        for idx, score in top
    ]


async def search_sv_semantic(
    conn: aiosqlite.Connection,
    query: str,
    top_k: int = 5,
) -> list[SumulaVinculanteResult]:
    """Retorna top_k Súmulas Vinculantes por similaridade semântica com a query."""
    cache = await _load_sv_cache(conn)
    if cache is None:
        return []

    ids, meta, vectors = cache
    query_vec = _embed_query(query)
    top = _top_k_by_cosine(query_vec, vectors, top_k)

    return [
        SumulaVinculanteResult(
            id=ids[idx],
            numero=meta[idx][0],
            enunciado=meta[idx][1],
            rank=-score,
        )
        for idx, score in top
    ]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------


def rrf_acordaos(
    lexical: list[SearchResult],
    semantic: list[SearchResult],
    top_n: int,
    k: int = 60,
) -> list[SearchResult]:
    """
    Funde rankings lexical (FTS5) e semântico de acórdãos via RRF.

    score(d) = 1/(k + rank_lexical + 1) + 1/(k + rank_semantic + 1)

    Documentos presentes em apenas uma lista recebem contribuição parcial.
    """
    scores: dict[int, float] = {}
    doc_map: dict[int, SearchResult] = {}

    for rank, doc in enumerate(lexical):
        scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (k + rank + 1)
        doc_map[doc.id] = doc

    for rank, doc in enumerate(semantic):
        scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (k + rank + 1)
        if doc.id not in doc_map:
            doc_map[doc.id] = doc

    ranked = sorted(scores, key=lambda i: scores[i], reverse=True)
    return [doc_map[i] for i in ranked[:top_n]]


def rrf_teses(
    lexical: list[TesesResult],
    semantic: list[TesesResult],
    top_n: int,
    k: int = 60,
) -> list[TesesResult]:
    """Funde rankings lexical e semântico de teses via RRF."""
    scores: dict[int, float] = {}
    doc_map: dict[int, TesesResult] = {}

    for rank, doc in enumerate(lexical):
        scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (k + rank + 1)
        doc_map[doc.id] = doc

    for rank, doc in enumerate(semantic):
        scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (k + rank + 1)
        if doc.id not in doc_map:
            doc_map[doc.id] = doc

    ranked = sorted(scores, key=lambda i: scores[i], reverse=True)
    return [doc_map[i] for i in ranked[:top_n]]


def rrf_sv(
    lexical: list[SumulaVinculanteResult],
    semantic: list[SumulaVinculanteResult],
    top_n: int,
    k: int = 60,
) -> list[SumulaVinculanteResult]:
    """Funde rankings lexical e semântico de Súmulas Vinculantes via RRF."""
    scores: dict[int, float] = {}
    doc_map: dict[int, SumulaVinculanteResult] = {}

    for rank, doc in enumerate(lexical):
        scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (k + rank + 1)
        doc_map[doc.id] = doc

    for rank, doc in enumerate(semantic):
        scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (k + rank + 1)
        if doc.id not in doc_map:
            doc_map[doc.id] = doc

    ranked = sorted(scores, key=lambda i: scores[i], reverse=True)
    return [doc_map[i] for i in ranked[:top_n]]
