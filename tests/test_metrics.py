"""
Testes unitários para as funções de métricas de avaliação (eval/metrics.py).

Cobre: recall_at_k, mrr, ndcg_at_k, precision_at_k, compute_all, aggregate.
"""

from __future__ import annotations

import math

import pytest

from eval.metrics import (
    aggregate,
    compute_all,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


# ===========================================================================
# recall_at_k
# ===========================================================================


def test_recall_all_relevant() -> None:
    """Todos os docs são relevantes → Recall@k = 1.0."""
    assert recall_at_k([True, True, True], k=3) == 1.0


def test_recall_none_relevant() -> None:
    """Nenhum relevante → 0.0."""
    assert recall_at_k([False, False, False], k=3) == 0.0


def test_recall_partial() -> None:
    """1 relevante recuperado de 2 relevantes totais → Recall@2 = 0.5."""
    # relevant = [True, False, True, False]: 2 relevantes no total
    # top-2 = [True, False]: 1 recuperado → 1/2 = 0.5
    assert recall_at_k([True, False, True, False], k=2) == pytest.approx(0.5)


def test_recall_cutoff_excludes_tail() -> None:
    """Relevantes além do top-k não contam."""
    # 1 relevante nos top-2, 2 relevantes no total da lista
    result = recall_at_k([True, False, True], k=2)
    assert result == pytest.approx(0.5)


# ===========================================================================
# mrr
# ===========================================================================


def test_mrr_first_position() -> None:
    """Primeiro doc relevante → MRR = 1.0."""
    assert mrr([True, False, False]) == 1.0


def test_mrr_second_position() -> None:
    assert mrr([False, True, False]) == pytest.approx(1 / 2)


def test_mrr_third_position() -> None:
    assert mrr([False, False, True]) == pytest.approx(1 / 3)


def test_mrr_no_relevant() -> None:
    assert mrr([False, False, False]) == 0.0


def test_mrr_multiple_relevant_uses_first() -> None:
    """Com múltiplos relevantes, usa apenas o primeiro."""
    assert mrr([False, True, True]) == pytest.approx(1 / 2)


# ===========================================================================
# ndcg_at_k
# ===========================================================================


def test_ndcg_perfect_ranking() -> None:
    """Ranking ideal → nDCG@k = 1.0."""
    assert ndcg_at_k([True, True, False], k=3) == pytest.approx(1.0)


def test_ndcg_no_relevant() -> None:
    assert ndcg_at_k([False, False, False], k=3) == 0.0


def test_ndcg_relevant_at_position_2_less_than_1() -> None:
    """Relevante na posição 2 deve ter nDCG menor que na posição 1."""
    ndcg_pos1 = ndcg_at_k([True, False, False], k=3)
    ndcg_pos2 = ndcg_at_k([False, True, False], k=3)
    assert ndcg_pos1 > ndcg_pos2


def test_ndcg_between_0_and_1() -> None:
    result = ndcg_at_k([False, True, True, False], k=4)
    assert 0.0 <= result <= 1.0


# ===========================================================================
# precision_at_k
# ===========================================================================


def test_precision_all_relevant() -> None:
    assert precision_at_k([True, True, True], k=3) == pytest.approx(1.0)


def test_precision_none_relevant() -> None:
    assert precision_at_k([False, False], k=2) == 0.0


def test_precision_half() -> None:
    assert precision_at_k([True, False, True, False], k=4) == pytest.approx(0.5)


def test_precision_k_zero_returns_zero() -> None:
    assert precision_at_k([True, True], k=0) == 0.0


# ===========================================================================
# compute_all
# ===========================================================================


def test_compute_all_returns_all_metrics() -> None:
    result = compute_all([True, False, True], k=3)
    assert "Recall@3" in result
    assert "MRR" in result
    assert "nDCG@3" in result
    assert "P@3" in result


def test_compute_all_values_are_floats() -> None:
    result = compute_all([True, False], k=2)
    assert all(isinstance(v, float) for v in result.values())


# ===========================================================================
# aggregate
# ===========================================================================


def test_aggregate_mean() -> None:
    results = [
        {"MRR": 1.0, "P@5": 0.8},
        {"MRR": 0.5, "P@5": 0.4},
    ]
    agg = aggregate(results)
    assert agg["MRR"] == pytest.approx(0.75)
    assert agg["P@5"] == pytest.approx(0.6)


def test_aggregate_empty_returns_empty() -> None:
    assert aggregate([]) == {}


def test_aggregate_single_entry() -> None:
    results = [{"MRR": 0.33}]
    agg = aggregate(results)
    assert agg["MRR"] == pytest.approx(0.33)
