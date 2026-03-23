"""ETL — Carga no SQLite FTS5. Execução: python -m etl.load"""

import asyncio
import re
from pathlib import Path

# Aceita apenas nomes no padrão esperado: letras, dígitos, hífens e underscores
_CSV_NAME_RE = re.compile(r'^resultados-de-acordaos[\w\-]*\.csv$')

import aiosqlite

from src.config.settings import get_settings
from etl.extract import extract
from etl.transform import transform

RAW_DIR = Path("data/raw")


def _collect_csvs() -> list[Path]:
    """Coleta todos os CSVs de acórdãos em data/raw, validando nomes."""
    all_csvs = sorted(
        p for p in RAW_DIR.glob("resultados-de-acordaos*.csv")
        if _CSV_NAME_RE.match(p.name)
    )
    if not all_csvs:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {RAW_DIR}")
    return all_csvs


async def load() -> None:
    """Executa o pipeline ETL completo: extract → transform → load."""
    settings = get_settings()
    settings.ensure_db_dir()

    csvs = _collect_csvs()
    print(f"CSVs selecionados: {[p.name for p in csvs]}")
    df = transform(extract(csvs))

    async with aiosqlite.connect(settings.db_path) as db:
        # Recria as tabelas zerando dados anteriores
        await db.execute("DROP TABLE IF EXISTS jurisprudencia_fts")
        await db.execute("DROP TABLE IF EXISTS jurisprudencia")
        await db.execute("""
            CREATE TABLE jurisprudencia (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                tribunal        TEXT NOT NULL,
                numero_processo TEXT,
                ementa          TEXT,
                decisao         TEXT,
                data_julgamento TEXT,
                created_at      TEXT DEFAULT (datetime('now'))
            )
        """)
        await db.execute("""
            CREATE VIRTUAL TABLE jurisprudencia_fts
            USING fts5(
                ementa,
                content='jurisprudencia',
                content_rowid='id',
                tokenize='unicode61 remove_diacritics 1'
            )
        """)

        rows = [
            ("STF", row.titulo, row.ementa, None, row.data_julgamento)
            for row in df.itertuples(index=False)
        ]
        await db.executemany(
            "INSERT INTO jurisprudencia (tribunal, numero_processo, ementa, decisao, data_julgamento) VALUES (?,?,?,?,?)",
            rows,
        )
        await db.execute("INSERT INTO jurisprudencia_fts(jurisprudencia_fts) VALUES('rebuild')")
        await db.commit()

    print(f"{len(rows)} registros carregados com sucesso.")


if __name__ == "__main__":
    asyncio.run(load())
