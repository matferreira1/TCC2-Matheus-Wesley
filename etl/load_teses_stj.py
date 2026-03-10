"""
ETL: Carrega Jurisprudência em Teses do STJ no banco SQLite.

Fonte: JTSelecao.txt — exportação PDF→TXT do portal scon.stj.jus.br/SCON/jt/
Destino: tabela teses_stj + índice FTS5 teses_stj_fts no banco iajuris.db

Estrutura do arquivo fonte:
  - Cada "página física" é delimitada por \\x0c (form feed).
  - Cabeçalho de página: "   AREA   EDIÇÃO N. X: TITULO"
  - Tese: "    N. Texto da tese..." (4 espaços + número + ponto)
  - Julgados: "          Julgados: REsp ..." (10 espaços + "Julgados:")
  - Rodapé: linha contendo "scon.stj.jus.br" (ignorada)
"""

from __future__ import annotations

import logging
import re
import sqlite3
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Padrões regex
# ---------------------------------------------------------------------------

# Cabeçalho de página com área e edição (após \x0c)
_RX_HEADER = re.compile(
    r'^\s*((?:DIREITO|ORIENTAÇÕES|Direito)[^\n]+?)\s{3,}'
    r'EDIÇÃO N\.\s*(\d+):\s*([^\n]+)',
)
# Edição sem área (primeira ocorrência no bloco do sumário/conteúdo)
_RX_EDICAO = re.compile(r'^\s*EDIÇÃO N\.\s*(\d+):\s*([^\n]+)')
# Início de tese: 4 espaços + número + ponto
_RX_TESE = re.compile(r'^    (\d+)\.\s+(.+)')
# Continuação de tese (4 espaços, não começa com dígito+ponto)
_RX_TESE_CONT = re.compile(r'^    (?!\d+\.)(.+)')
# Julgados: 10 espaços + "Julgados:"
_RX_JULGADOS = re.compile(r'^ {8,}Julgados?:\s*(.+)', re.IGNORECASE)
# Continuação dos julgados
_RX_JULGADOS_CONT = re.compile(r'^ {8,}(.+)')
# Rodapé de página
_RX_FOOTER = re.compile(r'scon\.stj\.jus\.br')


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _parse(filepath: str) -> list[dict]:
    """
    Parseia o arquivo TXT e retorna lista de dicts representando cada tese.

    Cada dict contém: area, edicao_num, edicao_titulo, tese_num,
    tese_texto, julgados.
    """
    logger.info("Lendo arquivo: %s", filepath)
    with open(filepath, encoding="utf-8", errors="replace") as f:
        content = f.read()

    pages = content.split("\x0c")
    logger.info("Total de páginas (form-feed): %d", len(pages))

    # --- Passo 1: constrói mapa edicao_num → area varrendo todos os
    #              cabeçalhos que têm area explícita ---
    edicao_area_map: dict[int, str] = {}
    for page in pages:
        for line in page.splitlines()[:4]:
            m = _RX_HEADER.match(line)
            if m:
                num = int(m.group(2))
                if num not in edicao_area_map:
                    edicao_area_map[num] = m.group(1).strip()
                break

    logger.info("Edições com área mapeada: %d", len(edicao_area_map))

    # --- Passo 2: parser linha a linha com máquina de estados ---
    results: list[dict] = []
    area_atual = "OUTROS"
    edicao_num = 0
    edicao_titulo = ""

    tese_num: int | None = None
    tese_linhas: list[str] = []
    julgados_linhas: list[str] = []
    in_julgados = False

    def flush() -> None:
        nonlocal tese_num, tese_linhas, julgados_linhas, in_julgados
        if tese_num is not None and tese_linhas:
            results.append({
                "area": area_atual,
                "edicao_num": edicao_num,
                "edicao_titulo": edicao_titulo,
                "tese_num": tese_num,
                "tese_texto": " ".join(" ".join(tese_linhas).split()),
                "julgados": " ".join(" ".join(julgados_linhas).split()),
            })
        tese_num = None
        tese_linhas = []
        julgados_linhas = []
        in_julgados = False

    for page in pages:
        lines = page.splitlines()

        # Atualiza área e edição a partir do cabeçalho desta página
        for line in lines[:4]:
            m = _RX_HEADER.match(line)
            if m:
                area_atual = m.group(1).strip()
                edicao_num = int(m.group(2))
                edicao_titulo = m.group(3).strip()
                break
            m2 = _RX_EDICAO.match(line)
            if m2:
                num = int(m2.group(1))
                edicao_num = num
                edicao_titulo = m2.group(2).strip()
                # Recupera área do mapa pré-construído; mantém atual se não encontrar
                area_atual = edicao_area_map.get(num, area_atual)
                break

        for line in lines:
            # Ignora rodapés
            if _RX_FOOTER.search(line):
                continue

            # Nova tese → flush da anterior
            m_tese = _RX_TESE.match(line)
            if m_tese:
                flush()
                tese_num = int(m_tese.group(1))
                tese_linhas = [m_tese.group(2)]
                in_julgados = False
                continue

            if tese_num is None:
                continue

            # Início do bloco de julgados
            m_julg = _RX_JULGADOS.match(line)
            if m_julg:
                in_julgados = True
                julgados_linhas.append(m_julg.group(1))
                continue

            if in_julgados:
                m_cont = _RX_JULGADOS_CONT.match(line)
                if m_cont and line.strip():
                    julgados_linhas.append(m_cont.group(1))
            else:
                # Continuação do texto da tese (4 espaços, não é novo número)
                m_cont = _RX_TESE_CONT.match(line)
                if m_cont and line.strip():
                    tese_linhas.append(m_cont.group(1))

    flush()  # Última tese da última página
    logger.info("Teses extraídas pelo parser: %d", len(results))
    return results


# ---------------------------------------------------------------------------
# Carga no banco
# ---------------------------------------------------------------------------


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Cria tabela teses_stj e índice FTS5 caso ainda não existam."""
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS teses_stj (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            area           TEXT,
            edicao_num     INTEGER,
            edicao_titulo  TEXT,
            tese_num       INTEGER,
            tese_texto     TEXT,
            julgados       TEXT,
            created_at     TEXT DEFAULT (datetime('now'))
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS teses_stj_fts USING fts5(
            tese_texto,
            area,
            edicao_titulo,
            content='teses_stj',
            content_rowid='id',
            tokenize='unicode61 remove_diacritics 1'
        );

        CREATE TRIGGER IF NOT EXISTS teses_stj_ai
            AFTER INSERT ON teses_stj BEGIN
                INSERT INTO teses_stj_fts(rowid, tese_texto, area, edicao_titulo)
                VALUES (new.id, new.tese_texto, new.area, new.edicao_titulo);
            END;

        CREATE TRIGGER IF NOT EXISTS teses_stj_ad
            AFTER DELETE ON teses_stj BEGIN
                INSERT INTO teses_stj_fts(teses_stj_fts, rowid, tese_texto, area, edicao_titulo)
                VALUES ('delete', old.id, old.tese_texto, old.area, old.edicao_titulo);
            END;

        CREATE TRIGGER IF NOT EXISTS teses_stj_au
            AFTER UPDATE ON teses_stj BEGIN
                INSERT INTO teses_stj_fts(teses_stj_fts, rowid, tese_texto, area, edicao_titulo)
                VALUES ('delete', old.id, old.tese_texto, old.area, old.edicao_titulo);
                INSERT INTO teses_stj_fts(rowid, tese_texto, area, edicao_titulo)
                VALUES (new.id, new.tese_texto, new.area, new.edicao_titulo);
            END;
    """)
    conn.commit()
    logger.info("Schema teses_stj verificado/criado.")


def load(txt_path: str, db_path: str) -> int:
    """
    Parseia o TXT e insere as teses no banco SQLite.

    Retorna o nº de registros inseridos.
    É idempotente: aborta sem inserção se a tabela já contiver dados.
    """
    records = _parse(txt_path)
    if not records:
        logger.warning("Nenhuma tese encontrada no arquivo — verifique o formato.")
        return 0

    conn = sqlite3.connect(db_path)
    try:
        _ensure_schema(conn)
        cur = conn.cursor()

        # Idempotência: verifica se já há dados
        cur.execute("SELECT COUNT(*) FROM teses_stj")
        existing = cur.fetchone()[0]
        if existing > 0:
            logger.info(
                "teses_stj já contém %d registros — carga abortada (use --force para recarregar).",
                existing,
            )
            return 0

        # Insere registros (triggers cuidam do FTS5)
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
        logger.info("✓ %d teses STJ inseridas com sucesso.", len(records))
        return len(records)
    finally:
        conn.close()


def force_reload(txt_path: str, db_path: str) -> int:
    """Apaga dados existentes e recarrega do TXT."""
    conn = sqlite3.connect(db_path)
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute("DELETE FROM teses_stj")
        cur.execute("INSERT INTO teses_stj_fts(teses_stj_fts) VALUES('rebuild')")
        conn.commit()
        logger.info("Tabela teses_stj limpa.")
    finally:
        conn.close()
    return load(txt_path, db_path)


# ---------------------------------------------------------------------------
# Entrypoint CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    force = "--force" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    txt = args[0] if len(args) > 0 else "JTSelecao.txt"
    db = args[1] if len(args) > 1 else "data/db/iajuris.db"

    if not Path(txt).exists():
        logger.error("Arquivo não encontrado: %s", txt)
        sys.exit(1)
    if not Path(db).exists():
        logger.error(
            "Banco não encontrado: %s — rode primeiro o ETL principal (etl/load.py).", db
        )
        sys.exit(1)

    n = force_reload(txt, db) if force else load(txt, db)
    logger.info("Concluído. Registros inseridos: %d", n)
