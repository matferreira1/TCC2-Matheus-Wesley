"""ETL — Extração dos CSVs de acórdãos do STF."""

from pathlib import Path
import pandas as pd

# Colunas obrigatórias no schema canônico (após normalização)
EXPECTED_COLUMNS = {"Titulo", "Relator", "Data de publicação", "Data de julgamento", "Órgão julgador", "Ementa"}

# Mapeamento de nomes alternativos → nome canônico
_COL_MAP = {
    "Título"    : "Titulo",
    "Relator(a)": "Relator",
}


def _ler_csv(csv_path: Path) -> pd.DataFrame:
    """
    Lê um CSV de acórdãos STF detectando automaticamente o separador
    (vírgula ou ponto-e-vírgula) e normalizando os nomes das colunas.
    """
    # Tenta detectar o separador lendo as primeiras linhas como texto
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        primeira = f.readline()
    sep = ";" if primeira.count(";") > primeira.count(",") else ","

    df = pd.read_csv(
        csv_path,
        encoding="utf-8",
        sep=sep,
        on_bad_lines="skip",
        engine="python",
    )

    # Normaliza nomes de colunas
    df = df.rename(columns=_COL_MAP)

    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em {csv_path.name}: {missing}")

    return df


def extract(csv_paths: list[Path]) -> pd.DataFrame:
    """Lê um ou mais CSVs, concatena e deduplica por Titulo."""
    frames = []
    for csv_path in csv_paths:
        df = _ler_csv(csv_path)
        print(f"  {len(df):>6} registros extraídos de {csv_path.name}")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset="Titulo", keep="last")
    print(f"{len(combined)} registros únicos após deduplicação ({before - len(combined)} duplicatas removidas)")
    return combined
