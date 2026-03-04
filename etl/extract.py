"""ETL — Extração dos CSVs de acórdãos do STF."""

from pathlib import Path
import pandas as pd

EXPECTED_COLUMNS = {"Titulo", "Relator", "Data de publicação", "Data de julgamento", "Órgão julgador", "Ementa"}


def extract(csv_paths: list[Path]) -> pd.DataFrame:
    """Lê um ou mais CSVs, concatena e deduplica por Titulo."""
    frames = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, encoding="utf-8", sep=",")
        missing = EXPECTED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Colunas ausentes em {csv_path.name}: {missing}")
        print(f"  {len(df):>5} registros extraídos de {csv_path.name}")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset="Titulo", keep="last")
    print(f"{len(combined)} registros únicos após deduplicação ({before - len(combined)} duplicatas removidas)")
    return combined
