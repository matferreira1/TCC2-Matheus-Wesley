"""
Testes para o pipeline ETL (extract + transform).

Estrategia: cria DataFrames/CSVs temporarios em memoria sem depender
de arquivos em disco reais.
"""

from __future__ import annotations

import io
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from etl.extract import extract
from etl.transform import transform


# ===========================================================================
# Helpers
# ===========================================================================


def _make_csv(rows: list[dict]) -> Path:
    """Cria um CSV temporario em disco a partir de uma lista de dicts."""
    df = pd.DataFrame(rows)
    tmp = Path("/tmp/test_acordaos.csv")
    df.to_csv(tmp, index=False, encoding="utf-8")
    return tmp


_BASE_ROW = {
    "Titulo": "HC 100001 AgR",
    "Relator": "MIN. FULANO",
    "Data de publicação": "2023-01-15",
    "Data de julgamento": "2023-01-10",
    "Órgão julgador": "PRIMEIRA TURMA",
    "Ementa": "Habeas corpus. Prisao preventiva. Ordem concedida.",
}


# ===========================================================================
# Testes de extract()
# ===========================================================================


def test_extract_reads_valid_csv() -> None:
    """extract() deve retornar um DataFrame nao vazio para CSV valido."""
    csv_path = _make_csv([_BASE_ROW])
    result = extract([csv_path])
    assert len(result) == 1


def test_extract_returns_dataframe() -> None:
    """extract() deve retornar um pd.DataFrame."""
    csv_path = _make_csv([_BASE_ROW])
    result = extract([csv_path])
    assert isinstance(result, pd.DataFrame)


def test_extract_multiple_files_concatenated() -> None:
    """extract() deve concatenar multiplos CSVs em um unico DataFrame."""
    row1 = {**_BASE_ROW, "Titulo": "HC 100001"}
    row2 = {**_BASE_ROW, "Titulo": "HC 100002"}
    csv1 = Path("/tmp/test1.csv")
    csv2 = Path("/tmp/test2.csv")
    pd.DataFrame([row1]).to_csv(csv1, index=False)
    pd.DataFrame([row2]).to_csv(csv2, index=False)
    result = extract([csv1, csv2])
    assert len(result) == 2


def test_extract_deduplicates_by_titulo() -> None:
    """extract() deve remover duplicatas pelo campo Titulo, mantendo a ultima."""
    row_old = {**_BASE_ROW, "Titulo": "HC 100001", "Ementa": "versao antiga"}
    row_new = {**_BASE_ROW, "Titulo": "HC 100001", "Ementa": "versao nova"}
    csv_path = _make_csv([row_old, row_new])
    result = extract([csv_path])
    assert len(result) == 1
    assert "versao nova" in result.iloc[0]["Ementa"]


def test_extract_missing_columns_raises_value_error() -> None:
    """extract() deve lancar ValueError se alguma coluna obrigatoria estiver ausente."""
    csv_path = _make_csv([{"Titulo": "HC 100001", "Ementa": "texto"}])
    with pytest.raises(ValueError, match="Colunas ausentes"):
        extract([csv_path])


def test_extract_preserves_all_expected_columns() -> None:
    """extract() deve preservar todas as colunas obrigatorias no resultado."""
    csv_path = _make_csv([_BASE_ROW])
    result = extract([csv_path])
    for col in ("Titulo", "Ementa", "Data de julgamento"):
        assert col in result.columns


# ===========================================================================
# Testes de transform()
# ===========================================================================


def _make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_transform_returns_dataframe() -> None:
    """transform() deve retornar um pd.DataFrame."""
    df = _make_df([_BASE_ROW])
    result = transform(df)
    assert isinstance(result, pd.DataFrame)


def test_transform_output_columns() -> None:
    """transform() deve retornar exatamente as colunas: titulo, ementa, data_julgamento."""
    df = _make_df([_BASE_ROW])
    result = transform(df)
    assert set(result.columns) == {"titulo", "ementa", "data_julgamento"}


def test_transform_removes_empty_ementa() -> None:
    """transform() deve remover registros com ementa vazia ou apenas espacos."""
    rows = [
        {**_BASE_ROW, "Titulo": "HC 001", "Ementa": "Texto valido"},
        {**_BASE_ROW, "Titulo": "HC 002", "Ementa": ""},
        {**_BASE_ROW, "Titulo": "HC 003", "Ementa": "   "},
    ]
    df = _make_df(rows)
    result = transform(df)
    # Apenas o registro com ementa valida deve sobrar
    assert len(result) == 1
    assert result.iloc[0]["titulo"] == "HC 001"


def test_transform_normalizes_whitespace() -> None:
    """transform() deve colapsar espacos multiplos e remover espacos no inicio/fim."""
    row = {**_BASE_ROW, "Ementa": "  Habeas   corpus.   Prisão  preventiva.  "}
    df = _make_df([row])
    result = transform(df)
    ementa = result.iloc[0]["ementa"]
    assert "  " not in ementa  # nao pode ter dois espacos consecutivos
    assert not ementa.startswith(" ")
    assert not ementa.endswith(" ")


def test_transform_normalizes_titulo() -> None:
    """transform() deve normalizar o campo titulo tambem."""
    row = {**_BASE_ROW, "Titulo": "  HC  100001  AgR  "}
    df = _make_df([row])
    result = transform(df)
    assert result.iloc[0]["titulo"] == "HC 100001 AgR"


def test_transform_preserves_valid_records_count() -> None:
    """transform() deve manter todos os registros com ementa preenchida."""
    rows = [{**_BASE_ROW, "Titulo": f"HC {i}", "Ementa": f"Ementa {i}"}
            for i in range(5)]
    df = _make_df(rows)
    result = transform(df)
    assert len(result) == 5
