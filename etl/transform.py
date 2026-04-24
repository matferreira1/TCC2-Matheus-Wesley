"""ETL — Limpeza e normalização dos dados extraídos."""

import re
import pandas as pd

# Detecta menção explícita a repercussão geral no texto da ementa.
# Cobre tanto "repercussão geral" (com acento) quanto "repercussao geral" (sem).
_RG_RE = re.compile(r"repercuss[aã]o\s+geral", re.IGNORECASE)


def _clean(text: object) -> str:
    """Remove espaços extras e normaliza o texto."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa e normaliza o DataFrame, retornando apenas as colunas necessárias."""
    out = df.copy()
    out["titulo"]             = df["Titulo"].apply(_clean)
    out["ementa"]             = df["Ementa"].apply(_clean)
    out["data_julgamento"]    = df["Data de julgamento"].apply(_clean)
    out["orgao_julgador"]     = df["Órgão julgador"].apply(_clean)
    out["repercussao_geral"]  = out["ementa"].apply(
        lambda t: bool(_RG_RE.search(t)) if t else False
    )
    out = out[out["ementa"] != ""]
    n_rg = out["repercussao_geral"].sum()
    print(f"{len(out)} registros após limpeza ({n_rg} com repercussão geral detectada)")
    return out[["titulo", "ementa", "data_julgamento", "orgao_julgador", "repercussao_geral"]]
