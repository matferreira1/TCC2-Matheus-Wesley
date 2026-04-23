"""
Fixtures compartilhadas entre todos os módulos de teste.

Fornece:
  - ``db`` : banco SQLite em memória com schema completo e dados de exemplo,
             pronto para testes de search_service e rag_service.
"""

from __future__ import annotations

import pytest_asyncio
import aiosqlite

# ---------------------------------------------------------------------------
# DDL — espelho do schema de connection.py (hardcoded para isolamento total)
# ---------------------------------------------------------------------------

_DDL_JURISPRUDENCIA = """
CREATE TABLE IF NOT EXISTS jurisprudencia (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    tribunal         TEXT    NOT NULL,
    numero_processo  TEXT,
    ementa           TEXT,
    decisao          TEXT,
    data_julgamento  TEXT,
    embedding        BLOB,
    created_at       TEXT    DEFAULT (datetime('now'))
);
"""

_DDL_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS jurisprudencia_fts USING fts5(
    ementa,
    decisao,
    content='jurisprudencia',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 1'
);
"""

_DDL_TRIGGER_INSERT = """
CREATE TRIGGER IF NOT EXISTS jurisprudencia_ai
    AFTER INSERT ON jurisprudencia BEGIN
        INSERT INTO jurisprudencia_fts(rowid, ementa, decisao)
        VALUES (new.id, new.ementa, new.decisao);
    END
"""

_DDL_TRIGGER_DELETE = """
CREATE TRIGGER IF NOT EXISTS jurisprudencia_ad
    AFTER DELETE ON jurisprudencia BEGIN
        INSERT INTO jurisprudencia_fts(jurisprudencia_fts, rowid, ementa, decisao)
        VALUES ('delete', old.id, old.ementa, old.decisao);
    END
"""

_DDL_TRIGGER_UPDATE = """
CREATE TRIGGER IF NOT EXISTS jurisprudencia_au
    AFTER UPDATE ON jurisprudencia BEGIN
        INSERT INTO jurisprudencia_fts(jurisprudencia_fts, rowid, ementa, decisao)
        VALUES ('delete', old.id, old.ementa, old.decisao);
        INSERT INTO jurisprudencia_fts(rowid, ementa, decisao)
        VALUES (new.id, new.ementa, new.decisao);
    END
"""

_DDL_TESES_STJ = """
CREATE TABLE IF NOT EXISTS teses_stj (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    area           TEXT,
    edicao_num     INTEGER,
    edicao_titulo  TEXT,
    tese_num       INTEGER,
    tese_texto     TEXT,
    julgados       TEXT,
    embedding      BLOB,
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

_DDL_SV_STF = """
CREATE TABLE IF NOT EXISTS sumulas_vinculantes_stf (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    numero    INTEGER NOT NULL UNIQUE,
    enunciado TEXT    NOT NULL,
    embedding BLOB,
    created_at TEXT DEFAULT (datetime('now'))
);
"""

_DDL_SV_STF_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS sumulas_vinculantes_stf_fts USING fts5(
    enunciado,
    content='sumulas_vinculantes_stf',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 1'
);
"""

_DDL_SV_TRIGGER_INSERT = """
CREATE TRIGGER IF NOT EXISTS sv_stf_ai
    AFTER INSERT ON sumulas_vinculantes_stf BEGIN
        INSERT INTO sumulas_vinculantes_stf_fts(rowid, enunciado)
        VALUES (new.id, new.enunciado);
    END
"""

_DDL_SV_TRIGGER_DELETE = """
CREATE TRIGGER IF NOT EXISTS sv_stf_ad
    AFTER DELETE ON sumulas_vinculantes_stf BEGIN
        INSERT INTO sumulas_vinculantes_stf_fts(sumulas_vinculantes_stf_fts, rowid, enunciado)
        VALUES ('delete', old.id, old.enunciado);
    END
"""

_DDL_SV_TRIGGER_UPDATE = """
CREATE TRIGGER IF NOT EXISTS sv_stf_au
    AFTER UPDATE ON sumulas_vinculantes_stf BEGIN
        INSERT INTO sumulas_vinculantes_stf_fts(sumulas_vinculantes_stf_fts, rowid, enunciado)
        VALUES ('delete', old.id, old.enunciado);
        INSERT INTO sumulas_vinculantes_stf_fts(rowid, enunciado)
        VALUES (new.id, new.enunciado);
    END
"""

# ---------------------------------------------------------------------------
# Dados de exemplo
# ---------------------------------------------------------------------------

_SAMPLE_ACORDAOS = [
    (
        "STF",
        "HC 100001",
        "Habeas corpus. Prisão preventiva. Ausência de fundamentação adequada. "
        "Constrangimento ilegal configurado. III. RAZÕES DE DECIDIR: A decisão "
        "não apresenta fundamentação idônea para a manutenção da custódia cautelar. "
        "IV. DISPOSITIVO: Ordem concedida.",
        "",
        "2023-01-15",
    ),
    (
        "STF",
        "HC 100002",
        "Habeas corpus. Tráfico de drogas. Prisão preventiva mantida. "
        "Ordem pública. Periculosidade demonstrada. "
        "III. RAZÕES DE DECIDIR: Presença dos requisitos do art. 312 do CPP. "
        "IV. DISPOSITIVO: Ordem denegada.",
        "",
        "2023-02-20",
    ),
    (
        "STF",
        "ARE 100003",
        "Recurso extraordinário. Repercussão geral não reconhecida. "
        "Direito constitucional. Liberdade de expressão. "
        "III. RAZÕES DE DECIDIR: Ausência de questão constitucional. "
        "IV. DISPOSITIVO: Recurso não admitido.",
        "",
        "2023-03-10",
    ),
    (
        "STF",
        "RE 100004",
        "Direito administrativo. Servidor público. Estabilidade. "
        "Cargo efetivo. Demissão sem processo administrativo disciplinar. "
        "III. RAZÕES DE DECIDIR: Violação ao devido processo legal. "
        "IV. DISPOSITIVO: Recurso provido.",
        "",
        "2023-04-05",
    ),
    (
        "STF",
        "HC 100005",
        "Habeas corpus. Supressão de instância. Decisão monocrática. "
        "Inadmissibilidade. "
        "III. RAZÕES DE DECIDIR: Não se admite habeas corpus contra decisão "
        "monocrática de ministro de Tribunal Superior. "
        "IV. DISPOSITIVO: HC não conhecido.",
        "",
        "2023-05-12",
    ),
]

_SAMPLE_TESES = [
    (
        "DIREITO CIVIL",
        143,
        "PLANO DE SAÚDE - III",
        1,
        "O plano de saúde pode estabelecer as doenças com cobertura, "
        "mas não o tipo de tratamento a ser utilizado para a cura de cada uma.",
        "REsp 1733013/PR, REsp 1.378.165/RS",
    ),
    (
        "DIREITO DO CONSUMIDOR",
        99,
        "RESPONSABILIDADE DO FORNECEDOR",
        1,
        "O fornecedor de produtos ou serviços responde pelos danos causados "
        "a consumidores por defeitos decorrentes de projeto ou fabricação.",
        "REsp 1234567/SP",
    ),
    (
        "DIREITO PENAL",
        55,
        "PRISÃO PREVENTIVA",
        1,
        "A prisão preventiva exige fundamentação específica e atual sobre "
        "a necessidade da medida cautelar, nos termos do art. 312 do CPP.",
        "HC 567890/RJ",
    ),
    # Súmula STJ — exercita o branch especial em _build_prompt()
    (
        "SÚMULAS STJ",
        528,
        "ENUNCIADOS DAS SÚMULAS",
        1,
        "Compete ao juiz federal do local da apreensão da droga remetida do "
        "exterior pela via postal processar e julgar o crime de tráfico internacional.",
        "",
    ),
]


# ---------------------------------------------------------------------------
# Fixture principal
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db() -> aiosqlite.Connection:
    """
    Banco SQLite em memória com schema completo (tabelas + FTS5 + triggers)
    e dados de teste inseridos.

    Isolamento total: cada teste recebe uma conexão nova, sem efeito colateral.
    """
    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row

        # Schema
        for ddl in (
            _DDL_JURISPRUDENCIA,
            _DDL_FTS,
            _DDL_TRIGGER_INSERT,
            _DDL_TRIGGER_DELETE,
            _DDL_TRIGGER_UPDATE,
            _DDL_TESES_STJ,
            _DDL_TESES_STJ_FTS,
            _DDL_TESES_TRIGGER_INSERT,
            _DDL_TESES_TRIGGER_DELETE,
            _DDL_TESES_TRIGGER_UPDATE,
            _DDL_SV_STF,
            _DDL_SV_STF_FTS,
            _DDL_SV_TRIGGER_INSERT,
            _DDL_SV_TRIGGER_DELETE,
            _DDL_SV_TRIGGER_UPDATE,
        ):
            await conn.execute(ddl)
        await conn.commit()

        # Dados de acórdãos (trigger popula jurisprudencia_fts automaticamente)
        await conn.executemany(
            "INSERT INTO jurisprudencia (tribunal, numero_processo, ementa, decisao, data_julgamento) "
            "VALUES (?, ?, ?, ?, ?)",
            _SAMPLE_ACORDAOS,
        )

        # Dados de teses (trigger popula teses_stj_fts automaticamente)
        await conn.executemany(
            "INSERT INTO teses_stj (area, edicao_num, edicao_titulo, tese_num, tese_texto, julgados) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            _SAMPLE_TESES,
        )

        # Dados de SVs (trigger popula sumulas_vinculantes_stf_fts automaticamente)
        await conn.executemany(
            "INSERT INTO sumulas_vinculantes_stf (numero, enunciado) VALUES (?, ?)",
            [
                (
                    11,
                    "Só é lícito o uso de algemas em casos de resistência e de fundado receio de fuga "
                    "ou de perigo à integridade física própria ou alheia, por parte do preso ou de "
                    "terceiros, justificada a excepcionalidade por escrito, sob pena de responsabilidade "
                    "disciplinar, civil e penal do agente ou da autoridade e de nulidade da prisão ou do "
                    "ato processual a que se refere, sem prejuízo da responsabilidade civil do Estado.",
                ),
                (
                    25,
                    "É ilícita a prisão civil de depositário infiel, qualquer que seja a modalidade do depósito.",
                ),
                (
                    37,
                    "Não cabe ao Poder Judiciário, que não tem função legislativa, aumentar vencimentos "
                    "de servidores públicos sob o fundamento de isonomia.",
                ),
            ],
        )

        await conn.commit()
        yield conn
