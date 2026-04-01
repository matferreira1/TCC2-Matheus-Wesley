"""
Testes para rerank_service (cross-encoder).

Todos os testes usam mock do modelo para não carregar pesos em memória —
verificam apenas a lógica de ordenação, truncagem e fallback.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.services.rerank_service import _get_text, rerank
from src.services.search_service import SearchResult, TesesResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _acordao(id: int, ementa: str = "texto de ementa") -> SearchResult:
    return SearchResult(
        id=id,
        tribunal="STF",
        numero_processo=f"HC {id:06d}",
        ementa=ementa,
        rank=-1.0,
    )


def _tese(id: int, area: str = "DIREITO CIVIL", texto: str = "tese jurídica") -> TesesResult:
    return TesesResult(
        id=id,
        area=area,
        edicao_num=100,
        edicao_titulo="TITULO",
        tese_num=id,
        tese_texto=texto,
        julgados="REsp 1/SP",
        rank=-1.0,
    )


def _mock_model(scores: list[float]) -> MagicMock:
    """Retorna um mock de CrossEncoder cujo predict() retorna os scores dados."""
    m = MagicMock()
    m.predict.return_value = np.array(scores, dtype=np.float32)
    return m


# ---------------------------------------------------------------------------
# Testes de _get_text()
# ---------------------------------------------------------------------------


def test_get_text_acordao_returns_ementa() -> None:
    doc = _acordao(1, ementa="Habeas corpus. Prisão preventiva.")
    assert _get_text(doc) == "Habeas corpus. Prisão preventiva."


def test_get_text_tese_includes_area() -> None:
    doc = _tese(1, area="DIREITO PENAL", texto="Fundamentação exigida.")
    text = _get_text(doc)
    assert "DIREITO PENAL" in text
    assert "Fundamentação exigida." in text


def test_get_text_tese_sem_area_nao_quebra() -> None:
    doc = _tese(1, area="", texto="Texto sem área.")
    text = _get_text(doc)
    assert "Texto sem área." in text


def test_get_text_trunca_em_512_chars() -> None:
    doc = _acordao(1, ementa="x" * 1000)
    assert len(_get_text(doc)) == 512


# ---------------------------------------------------------------------------
# Testes de rerank() — lógica de ordenação
# ---------------------------------------------------------------------------


def test_rerank_ordena_por_score_descendente() -> None:
    docs = [_acordao(1), _acordao(2), _acordao(3)]
    # doc 2 recebe o maior score
    with patch("src.services.rerank_service._get_model", return_value=_mock_model([0.3, 0.9, 0.1])):
        result = rerank("habeas corpus", docs)

    assert [d.id for d in result] == [2, 1, 3]


def test_rerank_respeita_top_n() -> None:
    docs = [_acordao(i) for i in range(1, 6)]  # 5 docs
    with patch("src.services.rerank_service._get_model", return_value=_mock_model([0.1, 0.5, 0.9, 0.2, 0.7])):
        result = rerank("prisão", docs, top_n=2)

    assert len(result) == 2
    assert result[0].id == 3  # score 0.9
    assert result[1].id == 5  # score 0.7


def test_rerank_top_n_none_retorna_todos() -> None:
    docs = [_acordao(i) for i in range(1, 4)]
    with patch("src.services.rerank_service._get_model", return_value=_mock_model([0.2, 0.8, 0.5])):
        result = rerank("query", docs, top_n=None)

    assert len(result) == 3


def test_rerank_lista_vazia_retorna_vazia() -> None:
    result = rerank("query", [])
    assert result == []


def test_rerank_funciona_com_teses() -> None:
    docs = [_tese(1), _tese(2), _tese(3)]
    with patch("src.services.rerank_service._get_model", return_value=_mock_model([0.1, 0.8, 0.3])):
        result = rerank("plano de saúde", docs, top_n=2)

    assert result[0].id == 2
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Testes de fallback
# ---------------------------------------------------------------------------


def test_rerank_fallback_quando_modelo_falha() -> None:
    """Se o modelo lançar exceção, retorna a lista original sem erro."""
    docs = [_acordao(1), _acordao(2), _acordao(3)]
    with patch("src.services.rerank_service._get_model", side_effect=RuntimeError("modelo indisponível")):
        result = rerank("habeas corpus", docs, top_n=2)

    # Fallback: retorna os 2 primeiros na ordem original
    assert len(result) == 2
    assert result[0].id == 1
    assert result[1].id == 2


def test_rerank_fallback_sem_top_n_retorna_todos() -> None:
    docs = [_acordao(i) for i in range(1, 4)]
    with patch("src.services.rerank_service._get_model", side_effect=OSError("arquivo não encontrado")):
        result = rerank("query", docs)

    assert len(result) == 3


def test_rerank_fallback_predict_falha() -> None:
    """Se predict() lançar exceção, o fallback também é acionado."""
    docs = [_acordao(1), _acordao(2)]
    bad_model = MagicMock()
    bad_model.predict.side_effect = ValueError("predict falhou")

    with patch("src.services.rerank_service._get_model", return_value=bad_model):
        result = rerank("query", docs)

    assert [d.id for d in result] == [1, 2]
