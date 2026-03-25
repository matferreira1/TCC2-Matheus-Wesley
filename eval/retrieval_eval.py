"""
Avaliação do módulo de recuperação (Retrieval) — seção 4.4.1 do TCC.

Compara dois mecanismos de busca sobre a mesma base de dados:
  - FTS5/BM25 : sistema atual (ranqueamento probabilístico via BM25)
  - LIKE       : busca baseline sem ranqueamento (operador SQL LIKE)

Para cada consulta do dataset, verifica a relevância de cada documento
recuperado por julgamentos léxicos (must_contain + any_of) e calcula
Recall@k, MRR e nDCG@k conforme descrito na seção 4.4.1.

Uso:
    python -m eval.retrieval_eval [--top-k 5]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import unicodedata
from pathlib import Path

import aiosqlite

from eval.metrics import aggregate, compute_all
from src.config.settings import settings
from src.services.search_service import search, search_teses

logger = logging.getLogger(__name__)

_DATASET_PATH = Path(__file__).parent / "dataset.json"


# ---------------------------------------------------------------------------
# Utilitários de relevância
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """
    Minúsculas + remove diacríticos.

    Espelha o comportamento do tokenizador FTS5
    (tokenize='unicode61 remove_diacritics 1').
    """
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def is_relevant(doc_text: str, criteria: dict) -> bool:
    """
    Julgamento de relevância léxica para um documento.

    Um documento é considerado relevante se:
      1. Contém TODOS os termos em criteria['must_contain'], E
      2. Contém PELO MENOS UM termo em criteria['any_of']
         (se a lista for vazia, a condição 2 é ignorada).

    Args:
        doc_text: texto do documento (ementa ou tese_texto).
        criteria: dict com chaves 'must_contain' e 'any_of'.
    """
    text = _normalize(doc_text)

    for term in criteria.get("must_contain", []):
        if _normalize(term) not in text:
            return False

    any_of = criteria.get("any_of", [])
    if any_of:
        return any(_normalize(t) in text for t in any_of)

    return True


# ---------------------------------------------------------------------------
# Busca LIKE (baseline sem BM25)
# ---------------------------------------------------------------------------


async def _like_search(
    conn: aiosqlite.Connection,
    query: str,
    top_k: int,
) -> list[dict]:
    """
    Busca baseline via LIKE — sem ranqueamento BM25.

    Tokens extraídos da query são buscados com OR em ementa.
    Resultados retornados em ordem arbitrária (por id).
    """
    tokens = [
        t for t in re.sub(r"[^\w\s]", " ", query).lower().split()
        if len(t) > 2
    ]
    if not tokens:
        return []

    where_clause = " OR ".join("ementa LIKE ?" for _ in tokens)
    params: list = [f"%{t}%" for t in tokens] + [top_k]

    sql = f"""
        SELECT id, tribunal, numero_processo, ementa
        FROM jurisprudencia
        WHERE {where_clause}
        LIMIT ?
    """
    cursor = await conn.execute(sql, params)
    rows = await cursor.fetchall()
    return [
        {
            "id": row["id"],
            "tribunal": row["tribunal"],
            "numero_processo": row["numero_processo"] or "",
            "ementa": row["ementa"] or "",
        }
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Avaliação por consulta
# ---------------------------------------------------------------------------


async def _evaluate_question(
    conn: aiosqlite.Connection,
    question_data: dict,
    top_k: int,
) -> dict:
    """Avalia uma consulta com FTS5 e LIKE, retornando métricas de ambas."""
    question = question_data["question"]
    criteria = question_data["relevance"]

    # ── FTS5 (acórdãos STF + teses STJ em paralelo) ──────────────────────
    fts5_acordaos = await search(conn, question, top_k=top_k)
    fts5_teses = await search_teses(conn, question, top_k=3)

    fts5_relevant = (
        [is_relevant(r.ementa, criteria) for r in fts5_acordaos]
        + [is_relevant(r.tese_texto, criteria) for r in fts5_teses]
    )
    fts5_metrics = compute_all(fts5_relevant, k=top_k)

    # ── LIKE baseline (apenas acórdãos) ──────────────────────────────────
    like_results = await _like_search(conn, question, top_k)
    like_relevant = [is_relevant(r["ementa"], criteria) for r in like_results]
    like_metrics = compute_all(like_relevant, k=top_k)

    return {
        "id": question_data["id"],
        "question": question,
        "area": question_data["area"],
        "fts5": {
            "metrics": fts5_metrics,
            "retrieved": len(fts5_acordaos) + len(fts5_teses),
            "relevant_count": sum(fts5_relevant),
            "relevance_flags": fts5_relevant,
        },
        "like": {
            "metrics": like_metrics,
            "retrieved": len(like_results),
            "relevant_count": sum(like_relevant),
            "relevance_flags": like_relevant,
        },
    }


# ---------------------------------------------------------------------------
# Runner principal
# ---------------------------------------------------------------------------


async def run(top_k: int = 5) -> dict:
    """
    Executa avaliação de retrieval para todas as consultas do dataset.

    Returns:
        Dict com resultados por consulta e métricas agregadas (MRR, nDCG@k, Recall@k).
    """
    with open(_DATASET_PATH, encoding="utf-8") as f:
        questions = json.load(f)["questions"]

    async with aiosqlite.connect(settings.database_url) as conn:
        conn.row_factory = aiosqlite.Row

        cur = await conn.execute("SELECT COUNT(*) FROM jurisprudencia")
        count = (await cur.fetchone())[0]
        if count == 0:
            raise RuntimeError(
                "Base de dados vazia. Execute 'python -m etl.load' antes de avaliar."
            )

        logger.info("Iniciando avaliação de retrieval: %d consultas, top_k=%d", len(questions), top_k)

        results = []
        for q in questions:
            result = await _evaluate_question(conn, q, top_k)
            results.append(result)
            logger.info(
                "[%s] FTS5 nDCG@%d=%.3f  LIKE nDCG@%d=%.3f",
                q["id"], top_k,
                result["fts5"]["metrics"][f"nDCG@{top_k}"],
                top_k,
                result["like"]["metrics"][f"nDCG@{top_k}"],
            )

    fts5_agg = aggregate([r["fts5"]["metrics"] for r in results])
    like_agg = aggregate([r["like"]["metrics"] for r in results])

    return {
        "top_k": top_k,
        "n_questions": len(questions),
        "results": results,
        "aggregate": {
            "fts5": fts5_agg,
            "like": like_agg,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_table(aggregate_results: dict, top_k: int) -> None:
    """Exibe tabela comparativa de métricas no terminal."""
    fts5 = aggregate_results["fts5"]
    like = aggregate_results["like"]

    metrics = [f"Recall@{top_k}", "MRR", f"nDCG@{top_k}", f"P@{top_k}"]

    col_w = 14
    print("\n" + "─" * 56)
    print("  Avaliação de Retrieval — FTS5/BM25 vs LIKE baseline")
    print("─" * 56)
    print(f"  {'Métrica':<14} {'FTS5/BM25':>{col_w}} {'LIKE':>{col_w}}")
    print("─" * 56)
    for m in metrics:
        v_fts5 = fts5.get(m, 0.0)
        v_like = like.get(m, 0.0)
        delta = v_fts5 - v_like
        sign = "▲" if delta >= 0 else "▼"
        print(f"  {m:<14} {v_fts5:>{col_w}.4f} {v_like:>{col_w}.4f}  {sign}{abs(delta):.4f}")
    print("─" * 56 + "\n")


async def _main(top_k: int, output_dir: Path) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    results = await run(top_k=top_k)

    _print_table(results["aggregate"], top_k)

    output_path = output_dir / "retrieval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Resultados salvos em: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação de retrieval do IAJuris")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k documentos recuperados")
    parser.add_argument("--output-dir", type=Path, default=Path("eval/results"), help="Diretório de saída")
    args = parser.parse_args()
    asyncio.run(_main(args.top_k, args.output_dir))
