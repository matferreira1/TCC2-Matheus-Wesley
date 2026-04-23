"""
ETL: Carrega Súmulas Vinculantes do STF no banco SQLite.

Fonte: data/stf/sumulas_vinculantes.txt
Destino: tabela sumulas_vinculantes_stf + índice FTS5 sumulas_vinculantes_stf_fts

Formato do arquivo fonte (um bloco por súmula):
  SÚMULA VINCULANTE N
  Texto do enunciado.

  SÚMULA VINCULANTE N+1
  Texto do próximo enunciado.

As súmulas vinculantes têm efeito vinculante constitucional obrigatório
para todo o Judiciário e a administração pública (art. 103-A CF + Lei 11.417/2006).
São o precedente de maior hierarquia no ordenamento jurídico brasileiro.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_RX_HEADER = re.compile(r'^\s*SÚMULA\s+VINCULANTE\s+(\d+)\s*$', re.IGNORECASE)

DEFAULT_TXT = "data/stf/sumulas_vinculantes.txt"
DEFAULT_DB  = "data/db/iajuris.db"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _parse(filepath: str) -> list[dict]:
    """Parseia o arquivo TXT e retorna lista de dicts (numero, enunciado)."""
    logger.info("Lendo arquivo: %s", filepath)
    with open(filepath, encoding="utf-8", errors="replace") as f:
        content = f.read()

    results: list[dict] = []
    current_num: int | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_num, current_lines
        if current_num is not None and current_lines:
            text = " ".join(" ".join(current_lines).split()).strip()
            if text:
                results.append({"numero": current_num, "enunciado": text})
        current_num = None
        current_lines = []

    for line in content.splitlines():
        m = _RX_HEADER.match(line)
        if m:
            flush()
            current_num = int(m.group(1))
            continue
        if current_num is None:
            continue
        stripped = line.strip()
        if stripped:
            current_lines.append(stripped)

    flush()
    logger.info("Súmulas vinculantes extraídas: %d", len(results))
    return results


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL_TABLE = """
CREATE TABLE IF NOT EXISTS sumulas_vinculantes_stf (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    numero    INTEGER NOT NULL UNIQUE,
    enunciado TEXT    NOT NULL,
    embedding BLOB,
    created_at TEXT DEFAULT (datetime('now'))
);
"""

_DDL_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS sumulas_vinculantes_stf_fts USING fts5(
    enunciado,
    content='sumulas_vinculantes_stf',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 1'
);
"""

_DDL_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS sv_stf_ai
    AFTER INSERT ON sumulas_vinculantes_stf BEGIN
        INSERT INTO sumulas_vinculantes_stf_fts(rowid, enunciado)
        VALUES (new.id, new.enunciado);
    END;

CREATE TRIGGER IF NOT EXISTS sv_stf_ad
    AFTER DELETE ON sumulas_vinculantes_stf BEGIN
        INSERT INTO sumulas_vinculantes_stf_fts(sumulas_vinculantes_stf_fts, rowid, enunciado)
        VALUES ('delete', old.id, old.enunciado);
    END;

CREATE TRIGGER IF NOT EXISTS sv_stf_au
    AFTER UPDATE ON sumulas_vinculantes_stf BEGIN
        INSERT INTO sumulas_vinculantes_stf_fts(sumulas_vinculantes_stf_fts, rowid, enunciado)
        VALUES ('delete', old.id, old.enunciado);
        INSERT INTO sumulas_vinculantes_stf_fts(rowid, enunciado)
        VALUES (new.id, new.enunciado);
    END;
"""


def _ensure_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(_DDL_TABLE)
    cur.execute(_DDL_FTS)
    cur.executescript(_DDL_TRIGGERS)
    conn.commit()
    logger.info("Schema sumulas_vinculantes_stf verificado/criado.")


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------


def load(txt_path: str, db_path: str, force: bool = False) -> int:
    """
    Parseia o TXT e insere as súmulas vinculantes no banco SQLite.

    Com force=True, apaga registros anteriores e recarrega.
    É idempotente sem --force: aborta se a tabela já tiver dados.
    Retorna o número de registros inseridos.
    """
    records = _parse(txt_path)
    if not records:
        logger.warning("Nenhuma súmula vinculante encontrada — verifique o formato.")
        return 0

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        _ensure_schema(conn)
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM sumulas_vinculantes_stf")
        existing = cur.fetchone()[0]

        if existing > 0 and not force:
            logger.info(
                "sumulas_vinculantes_stf já contém %d registros — use --force para recarregar.",
                existing,
            )
            return 0

        if existing > 0:
            cur.execute("DELETE FROM sumulas_vinculantes_stf")
            cur.execute("INSERT INTO sumulas_vinculantes_stf_fts(sumulas_vinculantes_stf_fts) VALUES('rebuild')")
            logger.info("Registros anteriores removidos.")

        cur.executemany(
            "INSERT INTO sumulas_vinculantes_stf (numero, enunciado) VALUES (:numero, :enunciado)",
            records,
        )
        conn.commit()
        logger.info("✓ %d súmulas vinculantes STF inseridas.", len(records))
        return len(records)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Entrypoint CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    force = "--force" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    txt = args[0] if args else DEFAULT_TXT
    db  = args[1] if len(args) > 1 else DEFAULT_DB

    if not Path(txt).exists():
        logger.error("Arquivo não encontrado: %s", txt)
        sys.exit(1)
    if not Path(db).exists():
        logger.error("Banco não encontrado: %s — rode primeiro o ETL principal.", db)
        sys.exit(1)

    n = load(txt, db, force=force)
    logger.info("Concluído. Registros inseridos: %d", n)
