"""
ETL: Gera e armazena embeddings semânticos no banco SQLite.

Modelo: paraphrase-multilingual-MiniLM-L12-v2
  - 384 dimensões, multilíngue, roda em CPU
  - Armazenado como BLOB (struct float32) em colunas `embedding`

Documentos vetorizados:
  - jurisprudencia.ementa    (acórdãos STF)
  - teses_stj.tese_texto     (teses e súmulas STJ)

Execução:
    python -m etl.generate_embeddings           # só os sem embedding
    python -m etl.generate_embeddings --force   # regera todos
"""

from __future__ import annotations

import logging
import sqlite3
import struct
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_DB = "data/db/iajuris.db"
BATCH_SIZE = 64

_VALID_TABLES: frozenset[str] = frozenset({"jurisprudencia", "teses_stj"})
_VALID_COLUMNS: frozenset[str] = frozenset({"ementa", "tese_texto"})
# Trunca texto antes de codificar (modelo max=128 tokens ≈ ~400 chars PT)
MAX_CHARS_ACORDAO = 1000
MAX_CHARS_TESE = 500


def _serialize(vector: list[float]) -> bytes:
    """Converte lista de floats em bytes (little-endian float32)."""
    return struct.pack(f"{len(vector)}f", *vector)


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """Adiciona coluna `embedding BLOB` nas tabelas, se não existir."""
    cur = conn.cursor()
    for table in _VALID_TABLES:
        try:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN embedding BLOB")
            logger.info("Coluna `embedding` adicionada à tabela %s.", table)
        except sqlite3.OperationalError:
            pass  # já existe
    conn.commit()


def _embed_table(
    conn: sqlite3.Connection,
    model,
    table: str,
    text_col: str,
    max_chars: int,
    force: bool,
) -> int:
    """Gera embeddings para uma tabela inteira. Retorna nº de registros atualizados."""
    if table not in _VALID_TABLES:
        raise ValueError(f"Tabela não permitida: {table!r}")
    if text_col not in _VALID_COLUMNS:
        raise ValueError(f"Coluna não permitida: {text_col!r}")

    cur = conn.cursor()

    if force:
        cur.execute(f"SELECT id, {text_col} FROM {table} WHERE {text_col} IS NOT NULL")
    else:
        cur.execute(
            f"SELECT id, {text_col} FROM {table} "
            f"WHERE {text_col} IS NOT NULL AND embedding IS NULL"
        )

    rows = cur.fetchall()
    total = len(rows)
    if total == 0:
        logger.info("%s: nenhum registro pendente.", table)
        return 0

    logger.info("%s: %d registros para vetorizar.", table, total)

    for i in range(0, total, BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        ids = [r[0] for r in batch]
        texts = [r[1][:max_chars] for r in batch]

        vecs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

        cur.executemany(
            f"UPDATE {table} SET embedding = ? WHERE id = ?",
            [(_serialize(v.tolist()), doc_id) for v, doc_id in zip(vecs, ids)],
        )
        conn.commit()
        logger.info("%s: %d/%d", table, min(i + BATCH_SIZE, total), total)

    return total


def run(db_path: str = DEFAULT_DB, force: bool = False) -> None:
    """Gera embeddings para acórdãos STF e teses/súmulas STJ."""
    from sentence_transformers import SentenceTransformer

    if not Path(db_path).exists():
        logger.error("Banco não encontrado: %s", db_path)
        sys.exit(1)

    logger.info("Carregando modelo %s...", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    logger.info("Modelo carregado.")

    conn = sqlite3.connect(db_path)
    try:
        _ensure_columns(conn)
        n_acordaos = _embed_table(conn, model, "jurisprudencia", "ementa", MAX_CHARS_ACORDAO, force)
        n_teses = _embed_table(conn, model, "teses_stj", "tese_texto", MAX_CHARS_TESE, force)
        logger.info("✓ Concluído. Acórdãos: %d | Teses/Súmulas: %d", n_acordaos, n_teses)
    finally:
        conn.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    force = "--force" in sys.argv
    db = next((a for a in sys.argv[1:] if not a.startswith("--")), DEFAULT_DB)
    run(db_path=db, force=force)
