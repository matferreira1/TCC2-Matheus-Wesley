"""
Comparação de variantes — seção 4.4.2 do TCC.

Compara quatro variantes do pipeline conforme descrito no TCC:
  - Variante A | LLM sem RAG     : pergunta enviada diretamente ao LLM, sem recuperação
  - Variante B | LIKE + LLM      : busca LIKE simples (baseline) + geração LLM
  - Variante C | FTS5 + LLM      : BM25 via FTS5 + geração LLM (só lexical)
  - Variante D | Híbrido + LLM   : FTS5 + Semântica (RRF) + geração LLM

Para retrieval (Variantes B, C e D): Recall@k, MRR, nDCG@k
Para geração  (todas):              Groundedness, Relevância, Coerência, Fluência + Grounding

Comparação estatística: para análise não-paramétrica (seção 4.4.2), os
scores individuais por consulta são salvos no JSON de saída — permitindo
aplicar Wilcoxon signed-rank test ou Friedman test externamente.

Uso:
    python -m eval.compare_variants [--top-k 5] [--questions q01,q02,q03]
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

from eval.generation_eval import _judge, check_grounding
from eval.metrics import aggregate, compute_all
from eval.retrieval_eval import _like_search, is_relevant
from src.config.settings import settings
from src.services import groq_service, search_service, semantic_service
from src.services.rag_service import RagResponse, _build_prompt, answer

logger = logging.getLogger(__name__)

_DATASET_PATH = Path(__file__).parent / "dataset.json"


# ---------------------------------------------------------------------------
# Variante A — LLM sem RAG
# ---------------------------------------------------------------------------

_NO_RAG_PROMPT = """\
Você é um assistente jurídico especializado em jurisprudência brasileira.
Responda à seguinte pergunta com base no seu conhecimento geral sobre \
o direito brasileiro. Seja objetivo e direto.

### Pergunta:
{question}

### Resposta:"""


async def _run_no_rag(question: str) -> tuple[str, list]:
    """Variante A: pergunta direta ao LLM sem nenhuma recuperação."""
    prompt = _NO_RAG_PROMPT.format(question=question)
    text = await groq_service.generate(prompt)
    return text, []  # sem fontes


# ---------------------------------------------------------------------------
# Variante B — LIKE + LLM
# ---------------------------------------------------------------------------


async def _run_like_rag(
    conn: aiosqlite.Connection,
    question: str,
    top_k: int,
) -> tuple[str, list[dict]]:
    """Variante B: busca LIKE simples + geração LLM."""
    like_results = await _like_search(conn, question, top_k)

    context_parts = [
        f"[Acórdão STF {i + 1}] {r['numero_processo']}\n{r['ementa'][:800]}"
        for i, r in enumerate(like_results)
    ]
    context = "\n\n".join(context_parts) if context_parts else "Nenhum documento encontrado."

    prompt = (
        "Você é um assistente jurídico especializado em jurisprudência brasileira.\n\n"
        "REGRAS OBRIGATÓRIAS:\n"
        "1. Use APENAS os acórdãos abaixo. Não invente nem extrapole.\n"
        "2. Cite os acórdãos relevantes após cada afirmação.\n"
        "3. Se não houver evidência suficiente, responda: 'Não encontrei informação suficiente.'\n"
        "4. Responda em português, de forma objetiva.\n\n"
        f"### Documentos:\n{context}\n\n"
        f"### Pergunta:\n{question}\n\n"
        "### Resposta:"
    )

    text = await groq_service.generate(prompt)
    return text, like_results


# ---------------------------------------------------------------------------
# Variante C — FTS5 + LLM (só lexical, sem semântica)
# ---------------------------------------------------------------------------


async def _run_fts5_rag(
    conn: aiosqlite.Connection,
    question: str,
    top_k: int,
) -> tuple[str, list]:
    """Variante C: busca FTS5 (BM25 lexical) + prompt v5 + LLM — sem semântica."""
    sources = await search_service.search(conn, question, top_k=top_k)
    sources_teses = await search_service.search_teses(conn, question, top_k=max(1, top_k // 2))

    prompt = _build_prompt(question, sources, sources_teses)
    text = await groq_service.generate(prompt)
    all_sources = list(sources) + list(sources_teses)
    return text, all_sources


# ---------------------------------------------------------------------------
# Variante D — Híbrido FTS5 + Semântica + RRF + LLM
# ---------------------------------------------------------------------------


async def _run_hybrid_rag(
    conn: aiosqlite.Connection,
    question: str,
) -> tuple[str, list]:
    """Variante D: pipeline híbrido completo (FTS5 + semântica + RRF) + LLM."""
    rag_resp: RagResponse = await answer(conn, question)
    all_sources = list(rag_resp.sources) + list(rag_resp.sources_teses)
    return rag_resp.answer, all_sources


# ---------------------------------------------------------------------------
# Avaliação de uma consulta nas quatro variantes
# ---------------------------------------------------------------------------


async def _evaluate_question(
    conn: aiosqlite.Connection,
    question_data: dict,
    top_k: int,
) -> dict:
    """Avalia uma consulta nas quatro variantes e retorna métricas comparativas."""
    question = question_data["question"]
    criteria = question_data["relevance"]

    variants: dict[str, dict] = {}

    for variant_id, label in [("A", "no_rag"), ("B", "like"), ("C", "fts5"), ("D", "hybrid")]:
        try:
            if variant_id == "A":
                answer_text, sources = await _run_no_rag(question)
                retrieval_metrics = {}
            elif variant_id == "B":
                answer_text, sources = await _run_like_rag(conn, question, top_k)
                like_relevant = [
                    is_relevant(s.get("ementa", ""), criteria)
                    for s in sources
                ]
                retrieval_metrics = compute_all(like_relevant, k=top_k)
            elif variant_id == "C":
                answer_text, sources = await _run_fts5_rag(conn, question, top_k)
                fts5_relevant = [
                    is_relevant(
                        getattr(s, "ementa", None) or getattr(s, "tese_texto", ""),
                        criteria,
                    )
                    for s in sources
                ]
                retrieval_metrics = compute_all(fts5_relevant, k=top_k)
            else:  # D — híbrido
                answer_text, sources = await _run_hybrid_rag(conn, question)
                hybrid_relevant = [
                    is_relevant(
                        getattr(s, "ementa", None) or getattr(s, "tese_texto", ""),
                        criteria,
                    )
                    for s in sources
                ]
                retrieval_metrics = compute_all(hybrid_relevant, k=top_k)

            # Contexto para o judge
            context_parts = []
            for s in sources:
                if isinstance(s, dict):
                    context_parts.append(f"{s.get('numero_processo','')}: {s.get('ementa','')[:400]}")
                else:
                    text = getattr(s, "ementa", None) or getattr(s, "tese_texto", "")
                    num = getattr(s, "numero_processo", "")
                    context_parts.append(f"{num}: {text[:400]}")
            context = "\n\n".join(context_parts) or "Sem contexto recuperado."

            judge = await _judge(question, context, answer_text)
            grounding = check_grounding(
                answer_text,
                sources if not (sources and isinstance(sources[0], dict)) else [],
            )

            variants[label] = {
                "answer": answer_text,
                "sources_count": len(sources),
                "retrieval_metrics": retrieval_metrics,
                "judge_scores": judge,
                "grounding": grounding,
            }

            logger.info(
                "[%s] Variante %s (%s) → G=%.1f R=%.1f C=%.1f F=%.1f",
                question_data["id"], variant_id, label,
                judge["groundedness"], judge["relevancia"],
                judge["coerencia"], judge["fluencia"],
            )

        except Exception as exc:
            logger.error("[%s] Variante %s falhou: %s", question_data["id"], variant_id, exc)
            variants[label] = {"error": str(exc)}

    return {
        "id": question_data["id"],
        "question": question,
        "area": question_data["area"],
        "variants": variants,
    }


# ---------------------------------------------------------------------------
# Runner principal
# ---------------------------------------------------------------------------


def _mean_dim(results: list[dict], variant: str, dim: str) -> float:
    vals = [
        r["variants"][variant]["judge_scores"][dim]
        for r in results
        if variant in r["variants"]
        and "judge_scores" in r["variants"][variant]
        and r["variants"][variant]["judge_scores"].get(dim, -1) >= 0
    ]
    return round(sum(vals) / len(vals), 3) if vals else -1.0


async def run(question_ids: list[str] | None = None, top_k: int = 5) -> dict:
    """Executa comparação das quatro variantes sobre as consultas selecionadas."""
    with open(_DATASET_PATH, encoding="utf-8") as f:
        all_questions = json.load(f)["questions"]

    questions = (
        [q for q in all_questions if q["id"] in question_ids]
        if question_ids else all_questions
    )

    async with aiosqlite.connect(settings.database_url) as conn:
        conn.row_factory = aiosqlite.Row

        cur = await conn.execute("SELECT COUNT(*) FROM jurisprudencia")
        if (await cur.fetchone())[0] == 0:
            raise RuntimeError("Base de dados vazia. Execute o ETL antes de avaliar.")

        logger.info("Comparação de variantes: %d consultas, top_k=%d", len(questions), top_k)
        results = []
        for q in questions:
            r = await _evaluate_question(conn, q, top_k)
            results.append(r)

    aggregate_summary = {}
    for variant in ["no_rag", "like", "fts5", "hybrid"]:
        dims = {}
        for dim in ["groundedness", "relevancia", "coerencia", "fluencia"]:
            dims[dim] = _mean_dim(results, variant, dim)
        aggregate_summary[variant] = dims

    return {
        "top_k": top_k,
        "n_questions": len(questions),
        "results": results,
        "aggregate": aggregate_summary,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_comparison(aggregate: dict) -> None:
    dims = ["groundedness", "relevancia", "coerencia", "fluencia"]
    variants = [
        ("no_rag", "LLM sem RAG"),
        ("like", "LIKE + LLM"),
        ("fts5", "FTS5 + LLM"),
        ("hybrid", "Híbrido + LLM"),
    ]

    col_w = 14
    print("\n" + "─" * 82)
    print("  Comparação de Variantes — LLM-as-judge (0–5)")
    print("─" * 82)
    header = f"  {'Dimensão':<16}" + "".join(f" {lbl:>{col_w}}" for _, lbl in variants)
    print(header)
    print("─" * 82)
    for dim in dims:
        row = f"  {dim:<16}"
        for key, _ in variants:
            val = aggregate.get(key, {}).get(dim, -1.0)
            row += f" {val:>{col_w}.3f}"
        print(row)
    print("─" * 82 + "\n")


async def _main(question_ids: list[str] | None, top_k: int, output_dir: Path) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    results = await run(question_ids=question_ids, top_k=top_k)

    _print_comparison(results["aggregate"])

    output_path = output_dir / "variants_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Resultados salvos em: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparação de variantes do IAJuris")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--questions", type=str, default=None, help="IDs separados por vírgula")
    parser.add_argument("--output-dir", type=Path, default=Path("eval/results"))
    args = parser.parse_args()

    ids = args.questions.split(",") if args.questions else None
    asyncio.run(_main(ids, args.top_k, args.output_dir))
