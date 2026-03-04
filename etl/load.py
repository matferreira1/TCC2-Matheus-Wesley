"""ETL — Carga no SQLite FTS5. Execução: python -m etl.load"""

import asyncio
from pathlib import Path

import aiosqlite

from src.config.settings import get_settings
from etl.extract import extract
from etl.transform import transform

RAW_DIR = Path("data/raw")


def _collect_csvs() -> list[Path]:
    """Coleta todos os CSVs de acórdãos em data/raw, priorizando o maior arquivo acumulativo."""
    all_csvs = sorted(RAW_DIR.glob("resultados-de-acordaos*.csv"))
    if not all_csvs:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {RAW_DIR}")

    # O portal exporta arquivos acumulativos numerados; basta o maior (mais registros)
    # mais o arquivo original sem número (caso seja um lote diferente)
    numbered = [p for p in all_csvs if "(" in p.name]
    unnumbered = [p for p in all_csvs if "(" not in p.name]

    selected: list[Path] = []
    if numbered:
        # Pega apenas o de maior número (inclui todos os anteriores)
        selected.append(max(numbered, key=lambda p: int(p.stem.split("(")[1].rstrip(")").strip())))
    selected.extend(unnumbered)
    return selected


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
