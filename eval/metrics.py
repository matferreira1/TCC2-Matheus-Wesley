"""
Métricas de avaliação para sistemas RAG — seção 4.4.1 do TCC.

Implementa as três métricas de recuperação priorizadas:
  - Recall@k  : proporção de documentos relevantes nos top-k recuperados
  - MRR       : Mean Reciprocal Rank — posição do primeiro relevante
  - nDCG@k    : Normalized Discounted Cumulative Gain — qualidade do ranqueamento

E a métrica auxiliar:
  - P@k       : Precision@k (equivalente ao Recall@k quando total_relevant = k)

Todas as funções são puramente computacionais — sem dependências externas.
"""

from __future__ import annotations

import math


def recall_at_k(relevant: list[bool], k: int) -> float:
    """
    Recall@k: proporção de documentos relevantes recuperados nos top-k.

    Na literatura de IR clássica, Recall requer conhecer o total de documentos
    relevantes na coleção. Aqui, seguindo a prática comum em avaliação de RAG,
    o total relevante é estimado como o número de relevantes na lista retornada
    (assumindo que o sistema recuperou os candidatos plausíveis).

    Returns:
        Valor entre 0.0 e 1.0; 0.0 se nenhum relevante na lista.
    """
    total_relevant = sum(relevant)
    if total_relevant == 0:
        return 0.0
    return sum(relevant[:k]) / total_relevant


def mrr(relevant: list[bool]) -> float:
    """
    Reciprocal Rank por consulta — usado para calcular o MRR agregado.

    MRR = média de 1/rank do primeiro documento relevante sobre todas as consultas.

    Returns:
        1/(posição do primeiro relevante), ou 0.0 se nenhum relevante.
    """
    for i, r in enumerate(relevant):
        if r:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(relevant: list[bool], k: int) -> float:
    """
    nDCG@k: Normalized Discounted Cumulative Gain.

    Avalia a qualidade do ranqueamento: documentos relevantes em posições
    mais altas contribuem mais para a pontuação (desconto logarítmico).

    Returns:
        Valor entre 0.0 e 1.0; 0.0 se nenhum relevante.
    """
    def dcg(flags: list[bool]) -> float:
        return sum(
            (1.0 if f else 0.0) / math.log2(i + 2)
            for i, f in enumerate(flags[:k])
        )

    actual_dcg = dcg(relevant)
    # DCG ideal: todos os relevantes nas primeiras posições
    ideal_dcg = dcg(sorted(relevant[:k], reverse=True))
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def precision_at_k(relevant: list[bool], k: int) -> float:
    """P@k: proporção de documentos relevantes entre os top-k recuperados."""
    if k <= 0:
        return 0.0
    return sum(relevant[:k]) / k


def compute_all(relevant: list[bool], k: int = 5) -> dict[str, float]:
    """
    Calcula todas as métricas para uma única consulta.

    Args:
        relevant: lista de bool indicando relevância de cada doc recuperado.
        k: cutoff para as métricas @k.

    Returns:
        Dict com Recall@k, MRR, nDCG@k e P@k.
    """
    return {
        f"Recall@{k}": recall_at_k(relevant, k),
        "MRR": mrr(relevant),
        f"nDCG@{k}": ndcg_at_k(relevant, k),
        f"P@{k}": precision_at_k(relevant, k),
    }


def aggregate(results: list[dict[str, float]]) -> dict[str, float]:
    """
    Agrega métricas de múltiplas consultas por média aritmética.

    MRR agregado = Mean Reciprocal Rank (média dos RR individuais).

    Args:
        results: lista de dicts retornados por compute_all().

    Returns:
        Dict com a média de cada métrica sobre todas as consultas.
    """
    if not results:
        return {}
    keys = results[0].keys()
    return {key: sum(r[key] for r in results) / len(results) for key in keys}
