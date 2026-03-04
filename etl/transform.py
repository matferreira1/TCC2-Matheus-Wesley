"""ETL — Limpeza e normalização dos dados extraídos."""

import re
import pandas as pd


def _clean(text: object) -> str:
    """Remove espaços extras e normaliza o texto."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa e normaliza o DataFrame, retornando apenas as colunas necessárias."""
    out = df.copy()
    out["titulo"]          = df["Titulo"].apply(_clean)
    out["ementa"]          = df["Ementa"].apply(_clean)
    out["data_julgamento"] = df["Data de julgamento"].apply(_clean)
    out = out[out["ementa"] != ""]
    print(f"{len(out)} registros após limpeza")
    return out[["titulo", "ementa", "data_julgamento"]]
