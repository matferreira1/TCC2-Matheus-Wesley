"""
Gerenciamento assíncrono da conexão com o SQLite (aiosqlite).

Responsabilidades:
- Abrir/fechar a conexão global da aplicação (startup/shutdown).
- Criar o schema (tabela de metadados + índice FTS5) na primeira execução.
- Expor get_db() como Dependency Injection do FastAPI.

Schema:
    jurisprudencia       — tabela relacional com todos os metadados.
    jurisprudencia_fts   — tabela virtual FTS5 (external content) que
                           indexa os campos 'ementa' e 'decisao' para
                           busca textual BM25.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

import aiosqlite

from src.config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conexão global (inicializada no startup do FastAPI)
# ---------------------------------------------------------------------------

_db: aiosqlite.Connection | None = None

# ---------------------------------------------------------------------------
# DDL — schema do banco de dados
# ---------------------------------------------------------------------------

_DDL_JURISPRUDENCIA = f"""
CREATE TABLE IF NOT EXISTS {settings.db_table_meta} (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    tribunal         TEXT    NOT NULL,
    numero_processo  TEXT,
    ementa           TEXT,
    decisao          TEXT,
    data_julgamento  TEXT,
    created_at       TEXT    DEFAULT (datetime('now'))
);
"""

# Tabela virtual FTS5 com external content apontando para a tabela principal.
# - content='jurisprudencia' → lê os campos da tabela relacional durante rebuild.
# - content_rowid='id'       → rowid do FTS5 corresponde ao id da tabela principal.
# - tokenize=unicode61       → suporte a caracteres acentuados do português.
# - remove_diacritics=1      → "jurídico" encontra "juridico" (busca insensível a acento).
_DDL_FTS = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS {settings.db_table_fts} USING fts5(
    ementa,
    decisao,
    content='{settings.db_table_meta}',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 1'
);
"""

# ---------------------------------------------------------------------------
# DDL — tabela teses_stj (Jurisprudência em Teses do STJ)
# ---------------------------------------------------------------------------

_DDL_TESES_STJ = """
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
"""

_DDL_TESES_STJ_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS teses_stj_fts USING fts5(
    tese_texto,
    area,
    edicao_titulo,
    content='teses_stj',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 1'
);
"""

_DDL_TESES_TRIGGER_INSERT = """
CREATE TRIGGER IF NOT EXISTS teses_stj_ai
    AFTER INSERT ON teses_stj BEGIN
        INSERT INTO teses_stj_fts(rowid, tese_texto, area, edicao_titulo)
        VALUES (new.id, new.tese_texto, new.area, new.edicao_titulo);
    END
"""

_DDL_TESES_TRIGGER_DELETE = """
CREATE TRIGGER IF NOT EXISTS teses_stj_ad
    AFTER DELETE ON teses_stj BEGIN
        INSERT INTO teses_stj_fts(teses_stj_fts, rowid, tese_texto, area, edicao_titulo)
        VALUES ('delete', old.id, old.tese_texto, old.area, old.edicao_titulo);
    END
"""

_DDL_TESES_TRIGGER_UPDATE = """
CREATE TRIGGER IF NOT EXISTS teses_stj_au
    AFTER UPDATE ON teses_stj BEGIN
        INSERT INTO teses_stj_fts(teses_stj_fts, rowid, tese_texto, area, edicao_titulo)
        VALUES ('delete', old.id, old.tese_texto, old.area, old.edicao_titulo);
        INSERT INTO teses_stj_fts(rowid, tese_texto, area, edicao_titulo)
        VALUES (new.id, new.tese_texto, new.area, new.edicao_titulo);
    END
"""

# ---------------------------------------------------------------------------
# Triggers jurisprudencia FTS5
# ---------------------------------------------------------------------------

_DDL_TRIGGER_INSERT = f"""
CREATE TRIGGER IF NOT EXISTS jurisprudencia_ai
    AFTER INSERT ON {settings.db_table_meta} BEGIN
        INSERT INTO {settings.db_table_fts}
            (rowid, ementa, decisao)
        VALUES
            (new.id, new.ementa, new.decisao);
    END
"""

_DDL_TRIGGER_DELETE = f"""
CREATE TRIGGER IF NOT EXISTS jurisprudencia_ad
    AFTER DELETE ON {settings.db_table_meta} BEGIN
        INSERT INTO {settings.db_table_fts}
            ({settings.db_table_fts}, rowid, ementa, decisao)
        VALUES
            ('delete', old.id, old.ementa, old.decisao);
    END
"""

_DDL_TRIGGER_UPDATE = f"""
CREATE TRIGGER IF NOT EXISTS jurisprudencia_au
    AFTER UPDATE ON {settings.db_table_meta} BEGIN
        INSERT INTO {settings.db_table_fts}
            ({settings.db_table_fts}, rowid, ementa, decisao)
        VALUES
            ('delete', old.id, old.ementa, old.decisao);
        INSERT INTO {settings.db_table_fts}
            (rowid, ementa, decisao)
        VALUES
            (new.id, new.ementa, new.decisao);
    END
"""


# ---------------------------------------------------------------------------
# Ciclo de vida da conexão
# ---------------------------------------------------------------------------


async def open_db() -> None:
    """
    Abre a conexão global com o SQLite e aplica PRAGMAs de performance.

    Deve ser chamado **uma única vez** no evento de startup do FastAPI.
    Configura:
    - WAL mode: permite leituras concorrentes enquanto há escrita em andamento.
    - cache_size: usa até ~64 MB de cache de páginas em memória.
    - foreign_keys: ativa integridade referencial.
    """
    global _db

    settings.ensure_db_dir()

    _db = await aiosqlite.connect(settings.database_url)
    _db.row_factory = aiosqlite.Row  # resultados acessíveis por nome de coluna

    await _db.execute("PRAGMA journal_mode=WAL;")
    await _db.execute("PRAGMA cache_size=-65536;")   # 64 MB
    await _db.execute("PRAGMA synchronous=NORMAL;")  # seguro com WAL
    await _db.execute("PRAGMA foreign_keys=ON;")
    await _db.commit()

    logger.info("Conexão com o banco de dados aberta: %s", settings.database_url)


async def close_db() -> None:
    """
    Fecha a conexão global com o SQLite.

    Deve ser chamado no evento de shutdown do FastAPI.
    """
    global _db

    if _db is not None:
        await _db.close()
        _db = None
        logger.info("Conexão com o banco de dados encerrada.")


async def init_db() -> None:
    """
    Cria as tabelas e o índice FTS5 caso ainda não existam.

    Deve ser chamado **após** open_db(), no startup do FastAPI.
    É idempotente: pode ser executado múltiplas vezes sem efeito colateral.

    Raises:
        RuntimeError: Se open_db() não tiver sido chamado antes.
    """
    if _db is None:
        raise RuntimeError(
            "Banco de dados não foi aberto. Chame open_db() antes de init_db()."
        )

    # Executa cada bloco DDL separadamente
    for ddl_block in (_DDL_JURISPRUDENCIA, _DDL_FTS, _DDL_TESES_STJ, _DDL_TESES_STJ_FTS):
        await _db.execute(ddl_block)

    # Triggers são criados individualmente para evitar problemas de parsing
    # (os blocos BEGIN...END contêm ";" internamente e não podem ser divididos)
    for trigger_ddl in (
        _DDL_TRIGGER_INSERT,
        _DDL_TRIGGER_DELETE,
        _DDL_TRIGGER_UPDATE,
        _DDL_TESES_TRIGGER_INSERT,
        _DDL_TESES_TRIGGER_DELETE,
        _DDL_TESES_TRIGGER_UPDATE,
    ):
        await _db.execute(trigger_ddl)

    await _db.commit()
    logger.info("Schema do banco de dados verificado/criado com sucesso.")


# ---------------------------------------------------------------------------
# Dependency Injection para o FastAPI
# ---------------------------------------------------------------------------


async def get_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    """
    Dependency Injection do FastAPI que fornece a conexão global.

    Uso nos endpoints::

        from fastapi import Depends
        from src.database.connection import get_db

        @router.post("/query")
        async def handle_query(
            payload: QueryRequest,
            conn: aiosqlite.Connection = Depends(get_db),
        ) -> QueryResponse:
            ...

    Raises:
        RuntimeError: Se o banco não tiver sido inicializado no startup.
    """
    if _db is None:
        raise RuntimeError(
            "Banco de dados não inicializado. "
            "Verifique o evento de startup do FastAPI."
        )
    yield _db
