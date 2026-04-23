"""
Reranking cross-encoder — estágio final do pipeline RAG.

Aplica um cross-encoder sobre os candidatos pré-selecionados pelo RRF,
pontuando cada par (query, documento) diretamente. Mais preciso que o
bi-encoder semântico porque analisa query e documento juntos, não em
representações separadas.

Modelo padrão: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
  - Multilíngue (inclui PT), treinado em MS MARCO multilíngue
  - MiniLM 12 camadas × 384 hidden dims — rápido em CPU para ≤30 docs
  - ~120 MB em disco

Fallback gracioso: se o modelo não estiver disponível ou ocorrer qualquer
erro, retorna a lista original intacta (RRF order preservada).
"""

from __future__ import annotations

import logging
from typing import TypeVar

from src.services.search_service import SearchResult, TesesResult, SumulaVinculanteResult

logger = logging.getLogger(__name__)

MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# Singleton — carregado na primeira chamada
_model = None

T = TypeVar("T", SearchResult, TesesResult, SumulaVinculanteResult)

# Decision boundary do cross-encoder: scores >= 0.0 indicam relevância
_ANSWER_FILTER_THRESHOLD = 0.0


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder  # lazy import
        logger.info("Carregando cross-encoder (%s)...", MODEL_NAME)
        _model = CrossEncoder(MODEL_NAME)
        logger.info("Cross-encoder pronto.")
    return _model


def clear_cache() -> None:
    """Invalida o singleton do modelo (usar após troca de modelo em runtime)."""
    global _model
    _model = None
    logger.info("Cache do cross-encoder invalidado.")


# ---------------------------------------------------------------------------
# Extração de texto
# ---------------------------------------------------------------------------

_MAX_TEXT_CHARS = 512  # cross-encoder MiniLM max ≈ 128 tokens ≈ ~400 chars PT


def _get_text(doc: SearchResult | TesesResult | SumulaVinculanteResult) -> str:
    """Extrai o texto principal do documento para o cross-encoder."""
    if isinstance(doc, SearchResult):
        return doc.ementa[:_MAX_TEXT_CHARS]
    if isinstance(doc, SumulaVinculanteResult):
        return doc.enunciado[:_MAX_TEXT_CHARS]
    # TesesResult: antecede com área temática para enriquecer contexto
    prefix = f"{doc.area}: " if doc.area else ""
    return (prefix + doc.tese_texto)[:_MAX_TEXT_CHARS]


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------


def rerank(
    query: str,
    docs: list[T],
    top_n: int | None = None,
) -> list[T]:
    """
    Reordena ``docs`` por relevância usando o cross-encoder.

    Cada par (query, texto_do_doc) é pontuado diretamente pelo modelo,
    produzindo um ranking mais preciso que o RRF.

    Parâmetros
    ----------
    query:
        Pergunta original do usuário.
    docs:
        Candidatos pré-selecionados pelo RRF (aceita SearchResult ou TesesResult).
    top_n:
        Quantos documentos retornar após o reranking. ``None`` retorna todos.

    Retorno
    -------
    Lista reordenada (maior score primeiro), truncada em ``top_n``.
    Em caso de falha no modelo, retorna ``docs[:top_n]`` intacto.
    """
    if not docs:
        return docs

    limit = top_n if top_n is not None else len(docs)

    try:
        model = _get_model()
        pairs = [(query, _get_text(d)) for d in docs]
        scores: list[float] = model.predict(pairs).tolist()
    except Exception as exc:
        logger.warning(
            "Cross-encoder indisponível (%s) — usando ordem RRF.", exc
        )
        return docs[:limit]

    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    result = [d for _, d in ranked]
    logger.info(
        "Reranking: %d candidatos → top %d selecionados.", len(docs), min(limit, len(docs))
    )
    return result[:limit]


def filter_by_answer(
    answer: str,
    docs: list[T],
    threshold: float = _ANSWER_FILTER_THRESHOLD,
) -> list[T]:
    """
    Filtra documentos por relevância semântica em relação à resposta gerada pelo LLM.

    Usa o mesmo cross-encoder do reranking para pontuar pares (resposta, doc).
    Documentos com score abaixo de ``threshold`` são descartados, indicando que
    o conteúdo da fonte não foi refletido na resposta gerada.

    A ordem dos documentos retornados é preservada (sem re-ordenação).
    Fallback gracioso: retorna lista original se o modelo falhar.

    Parâmetros
    ----------
    answer:
        Texto completo da resposta gerada pelo LLM.
    docs:
        Fontes recuperadas pelo pipeline (SearchResult, TesesResult ou SumulaVinculanteResult).
    threshold:
        Score mínimo para manter a fonte. Padrão: 0.0 (decision boundary do cross-encoder).
    """
    if not docs:
        return docs

    try:
        model = _get_model()
        pairs = [(answer, _get_text(d)) for d in docs]
        scores: list[float] = model.predict(pairs).tolist()
    except Exception as exc:
        logger.warning(
            "filter_by_answer: cross-encoder falhou (%s) — retornando lista original.", exc
        )
        return docs

    kept = [d for score, d in zip(scores, docs) if score >= threshold]
    logger.info(
        "filter_by_answer: %d/%d fontes retidas (threshold=%.1f) | scores=%s",
        len(kept),
        len(docs),
        threshold,
        [f"{s:.2f}" for s in scores],
    )
    return kept
