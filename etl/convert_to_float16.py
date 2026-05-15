"""
Converte embeddings float32 → float16 no banco SQLite.

Reduz o banco de ~1 GB para ~750 MB e o uso de RAM em produção de 243 MB → 121 MB.
Sem perda de qualidade para busca por similaridade cosseno.

Uso:
    python -m etl.convert_to_float16              # banco padrão
    python -m etl.convert_to_float16 data/db/iajuris.db
"""

from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DB = "data/db/iajuris.db"
BATCH_SIZE = 10_000

_DIMS = 384
_F32_BYTES = _DIMS * 4
_F16_BYTES = _DIMS * 2

_TABLES = [
    ("jurisprudencia",          "embedding"),
    ("teses_stj",               "embedding"),
    ("sumulas_vinculantes_stf", "embedding"),
]


def _convert_table(conn: sqlite3.Connection, table: str, col: str) -> int:
    cur = conn.cursor()

    # Verifica se a tabela existe
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
    if not cur.fetchone():
        logger.info("%s: tabela não existe, pulando.", table)
        return 0

    # Conta quantos ainda são float32
    cur.execute(
        f"SELECT COUNT(*) FROM {table} WHERE {col} IS NOT NULL AND length({col}) = ?",
        (_F32_BYTES,),
    )
    total = cur.fetchone()[0]
    if total == 0:
        logger.info("%s: nenhum embedding float32 encontrado (já convertido?).", table)
        return 0

    logger.info("%s: convertendo %d embeddings float32 → float16...", table, total)

    # Sem OFFSET: após converter um lote, esses registros somem do WHERE
    # (length deixa de ser _F32_BYTES), então sempre buscamos do início.
    converted = 0
    while True:
        cur.execute(
            f"SELECT id, {col} FROM {table} "
            f"WHERE {col} IS NOT NULL AND length({col}) = ? "
            f"LIMIT ?",
            (_F32_BYTES, BATCH_SIZE),
        )
        rows = cur.fetchall()
        if not rows:
            break

        updates = []
        for row_id, blob in rows:
            vec_f32 = np.frombuffer(blob, dtype=np.float32)
            new_blob = vec_f32.astype(np.float16).tobytes()
            updates.append((new_blob, row_id))

        cur.executemany(
            f"UPDATE {table} SET {col} = ? WHERE id = ?",
            updates,
        )
        conn.commit()
        converted += len(rows)
        logger.info("%s: %d/%d convertidos", table, converted, total)

    return converted


def run(db_path: str = DEFAULT_DB) -> None:
    if not Path(db_path).exists():
        logger.error("Banco não encontrado: %s", db_path)
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        total = 0
        for table, col in _TABLES:
            total += _convert_table(conn, table, col)
        logger.info("✓ Conversão concluída. Total convertido: %d embeddings.", total)
        # Compacta o banco após redução dos BLOBs
        logger.info("Executando VACUUM para recuperar espaço em disco...")
        conn.execute("VACUUM")
        logger.info("✓ VACUUM concluído.")
    finally:
        conn.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    db = next((a for a in sys.argv[1:] if not a.startswith("--")), DEFAULT_DB)
    run(db)
