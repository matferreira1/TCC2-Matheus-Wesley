"""
ETL: Carrega Enunciados das Súmulas do STJ no banco SQLite.

Fonte: data/stj/SelecaoSumulas.txt — exportação PDF→TXT do portal scon.stj.jus.br/SCON/sumstj/
Destino: tabela teses_stj (mesma tabela das teses, area='SÚMULAS STJ') + FTS5

Formato do arquivo fonte:
  ●   SÚMULA NNN                                      VEJA MAIS
  <linha em branco>
  <linha em branco>
      Texto do enunciado. (SEÇÃO, julgado em DD/MM/YYYY, DJe de DD/MM/YYYY)
  <linha em branco>
  ●   SÚMULA NNN-1                                    VEJA MAIS
  ...

Mapeamento para teses_stj:
  area          = 'SÚMULAS STJ'
  edicao_num    = número da súmula (ex: 528)
  edicao_titulo = 'ENUNCIADOS DAS SÚMULAS'
  tese_num      = 1 (uma entrada por súmula)
  tese_texto    = texto completo do enunciado
  julgados      = ''
"""

from __future__ import annotations

import logging
import re
import sqlite3
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_AREA = "SÚMULAS STJ"
_EDICAO_TITULO = "ENUNCIADOS DAS SÚMULAS"

_RX_SUMULA = re.compile(r"●\s+SÚMULA\s+(\d+)", re.IGNORECASE)
_RX_FOOTER = re.compile(r"scon\.stj\.jus\.br")


def _parse(filepath: str) -> list[dict]:
    """Parseia o arquivo de súmulas e retorna lista de dicts."""
    logger.info("Lendo arquivo: %s", filepath)
    with open(filepath, encoding="utf-8", errors="replace") as f:
        content = f.read()

    results: list[dict] = []
    current_num: int | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_num, current_lines
        if current_num is not None and current_lines:
            text = " ".join(" ".join(current_lines).split())
            if text:
                results.append({
                    "area": _AREA,
                    "edicao_num": current_num,
                    "edicao_titulo": _EDICAO_TITULO,
                    "tese_num": 1,
                    "tese_texto": text,
                    "julgados": "",
                })
        current_num = None
        current_lines = []

    for line in content.splitlines():
        if _RX_FOOTER.search(line):
            continue

        m = _RX_SUMULA.search(line)
        if m:
            flush()
            current_num = int(m.group(1))
            continue

        if current_num is None:
            continue

        stripped = line.strip()
        if not stripped or "VEJA MAIS" in stripped:
            continue

        current_lines.append(stripped)

    flush()
    logger.info("Súmulas extraídas: %d", len(results))
    return results


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Garante que o schema de teses_stj existe (criado pelo load_teses_stj)."""
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='teses_stj'")
    if not cur.fetchone():
        raise RuntimeError(
            "Tabela teses_stj não encontrada. "
            "Execute primeiro: python -m etl.load_teses_stj"
        )


def load(txt_path: str, db_path: str, force: bool = False) -> int:
    """
    Parseia o TXT e insere as súmulas na tabela teses_stj.

    Com force=True, remove súmulas existentes antes de inserir.
    Retorna o nº de registros inseridos.
    """
    records = _parse(txt_path)
    if not records:
        logger.warning("Nenhuma súmula encontrada — verifique o formato do arquivo.")
        return 0

    conn = sqlite3.connect(db_path)
    try:
        _ensure_schema(conn)
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM teses_stj WHERE area = ?", (_AREA,))
        existing = cur.fetchone()[0]

        if existing > 0 and not force:
            logger.info(
                "Já existem %d súmulas no banco — use --force para recarregar.", existing
            )
            return 0

        if existing > 0:
            cur.execute("DELETE FROM teses_stj WHERE area = ?", (_AREA,))
            cur.execute("INSERT INTO teses_stj_fts(teses_stj_fts) VALUES('rebuild')")
            logger.info("Súmulas anteriores removidas.")

        cur.executemany(
            """
            INSERT INTO teses_stj
                (area, edicao_num, edicao_titulo, tese_num, tese_texto, julgados)
            VALUES
                (:area, :edicao_num, :edicao_titulo, :tese_num, :tese_texto, :julgados)
            """,
            records,
        )
        conn.commit()
        logger.info("✓ %d súmulas STJ inseridas.", len(records))
        return len(records)
    finally:
        conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    force = "--force" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    txt = args[0] if args else "data/stj/SelecaoSumulas.txt"
    db = args[1] if len(args) > 1 else "data/db/iajuris.db"

    if not Path(txt).exists():
        logger.error("Arquivo não encontrado: %s", txt)
        sys.exit(1)
    if not Path(db).exists():
        logger.error("Banco não encontrado: %s", db)
        sys.exit(1)

    n = load(txt, db, force=force)
    logger.info("Concluído. Registros inseridos: %d", n)
