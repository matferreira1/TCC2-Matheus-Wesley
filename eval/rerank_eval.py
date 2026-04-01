"""
Avaliação comparativa: RAG híbrido sem vs com cross-encoder reranking.

Executa o pipeline completo instrumentado por estágio para 2 perguntas
e reporta lado a lado:

  Desempenho:
    - t_retrieval  : FTS5 + semântico + RRF
    - t_rerank     : inferência do cross-encoder  (só na variante COM)
    - t_llm        : geração da resposta
    - t_total      : soma dos três

  Qualidade retrieval (IR):
    - Recall@k, MRR, nDCG@k, P@k  (com base nos critérios do dataset.json)

  Qualidade geração (LLM-as-judge):
    - Groundedness, Relevância, Coerência, Fluência  (0–5)
    - Grounding score: % de citações válidas na resposta

  Documentos selecionados:
    - Tabela com os documentos que entraram no prompt em cada variante
      para inspeção visual da diferença de ordenação.

Uso:
    python -m eval.rerank_eval
    python -m eval.rerank_eval --questions q01,q03
    python -m eval.rerank_eval --no-judge          # pula LLM-as-judge (mais rápido)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from textwrap import shorten

import aiosqlite

from eval.generation_eval import _judge, check_grounding
from eval.metrics import compute_all
from eval.retrieval_eval import is_relevant
from src.config.settings import settings
from src.services import groq_service, rerank_service, search_service, semantic_service
from src.services.rag_service import _build_prompt

logger = logging.getLogger(__name__)

_DATASET_PATH = Path(__file__).parent / "dataset.json"

# Espelha as constantes de rag_service.py
_FETCH = 15
_RRF_CANDIDATES = 20


# ---------------------------------------------------------------------------
# Execução de uma variante
# ---------------------------------------------------------------------------


async def _run_variant(
    conn: aiosqlite.Connection,
    question: str,
    criteria: dict,
    use_reranker: bool,
) -> dict:
    """
    Executa o pipeline RAG híbrido com ou sem cross-encoder, cronometrando
    cada estágio separadamente.

    Retorna dict com latência por estágio, métricas IR, resposta LLM e fontes.
    """
    top_k = settings.rag_top_k
    top_k_teses = settings.rag_top_k_teses

    # ── 1. Retrieval: FTS5 + semântico + RRF ─────────────────────────────
    t0 = time.perf_counter()
    (fts5_acord, fts5_teses, sem_acord, sem_teses) = await asyncio.gather(
        search_service.search(conn, question, top_k=_FETCH),
        search_service.search_teses(conn, question, top_k=_FETCH),
        semantic_service.search_semantic(conn, question, top_k=_FETCH),
        semantic_service.search_teses_semantic(conn, question, top_k=_FETCH),
    )
    candidates_acord = semantic_service.rrf_acordaos(fts5_acord, sem_acord, top_n=_RRF_CANDIDATES)
    candidates_teses = semantic_service.rrf_teses(fts5_teses, sem_teses, top_n=_RRF_CANDIDATES)
    t_retrieval = time.perf_counter() - t0

    # ── 2. Reranking (opcional) ───────────────────────────────────────────
    t_rerank = 0.0
    if use_reranker:
        t1 = time.perf_counter()
        sources = rerank_service.rerank(question, candidates_acord, top_n=top_k)
        sources_teses = rerank_service.rerank(question, candidates_teses, top_n=top_k_teses)
        t_rerank = time.perf_counter() - t1
    else:
        sources = candidates_acord[:top_k]
        sources_teses = candidates_teses[:top_k_teses]

    # ── 3. Geração LLM ────────────────────────────────────────────────────
    t2 = time.perf_counter()
    prompt = _build_prompt(question, sources, sources_teses)
    answer_text = await groq_service.generate(prompt)
    t_llm = time.perf_counter() - t2

    # ── 4. Métricas IR ────────────────────────────────────────────────────
    all_sources = list(sources) + list(sources_teses)
    relevant_flags = [
        is_relevant(
            getattr(s, "ementa", None) or getattr(s, "tese_texto", ""),
            criteria,
        )
        for s in all_sources
    ]
    ir_metrics = compute_all(relevant_flags, k=top_k)

    return {
        "answer": answer_text,
        "sources": all_sources,
        "latency": {
            "retrieval_s": round(t_retrieval, 3),
            "rerank_s": round(t_rerank, 3),
            "llm_s": round(t_llm, 3),
            "total_s": round(t_retrieval + t_rerank + t_llm, 3),
        },
        "ir_metrics": ir_metrics,
        "relevant_flags": relevant_flags,
    }


# ---------------------------------------------------------------------------
# Avaliação de uma pergunta nas duas variantes
# ---------------------------------------------------------------------------


async def _evaluate_question(
    conn: aiosqlite.Connection,
    q: dict,
    run_judge: bool,
) -> dict:
    question = q["question"]
    criteria = q["relevance"]

    logger.info("Pergunta: %s", question[:60])

    logger.info("  → variante SEM reranking...")
    no_rerank = await _run_variant(conn, question, criteria, use_reranker=False)

    logger.info("  → variante COM reranking...")
    with_rerank = await _run_variant(conn, question, criteria, use_reranker=True)

    result = {
        "id": q["id"],
        "question": question,
        "area": q.get("area", ""),
        "no_rerank": no_rerank,
        "with_rerank": with_rerank,
    }

    # ── LLM-as-judge (opcional) ───────────────────────────────────────────
    if run_judge:
        logger.info("  → judge SEM reranking...")
        ctx_no = "\n\n".join(
            f"{getattr(s,'numero_processo','')}: "
            f"{(getattr(s,'ementa',None) or getattr(s,'tese_texto',''))[:400]}"
            for s in no_rerank["sources"]
        )
        no_rerank["judge"] = await _judge(question, ctx_no, no_rerank["answer"])
        no_rerank["grounding"] = check_grounding(no_rerank["answer"], no_rerank["sources"])

        logger.info("  → judge COM reranking...")
        ctx_with = "\n\n".join(
            f"{getattr(s,'numero_processo','')}: "
            f"{(getattr(s,'ementa',None) or getattr(s,'tese_texto',''))[:400]}"
            for s in with_rerank["sources"]
        )
        with_rerank["judge"] = await _judge(question, ctx_with, with_rerank["answer"])
        with_rerank["grounding"] = check_grounding(with_rerank["answer"], with_rerank["sources"])

    return result


# ---------------------------------------------------------------------------
# Impressão de resultados
# ---------------------------------------------------------------------------

_SEP = "─" * 76


def _fmt(val: float) -> str:
    return f"{val:.4f}"


def _print_results(results: list[dict], run_judge: bool) -> None:
    print("\n" + "═" * 76)
    print("  Avaliação: RAG Híbrido — SEM vs COM cross-encoder reranking")
    print("═" * 76)

    for r in results:
        nr = r["no_rerank"]
        wr = r["with_rerank"]

        print(f"\n  Pergunta [{r['id']}]: {r['question'][:65]}...")
        print(f"  Área: {r['area']}")
        print(_SEP)

        # ── Latência ─────────────────────────────────────────────────────
        col = 14
        print(f"\n  {'Latência':^{col*2+4}}")
        print(f"  {'Estágio':<20} {'SEM rerank':>{col}} {'COM rerank':>{col}}  Delta")
        print(f"  {'-'*20} {'-'*col} {'-'*col}  -------")
        for label, key in [
            ("Retrieval (FTS5+RRF)", "retrieval_s"),
            ("Cross-encoder", "rerank_s"),
            ("LLM", "llm_s"),
            ("TOTAL", "total_s"),
        ]:
            v_no = nr["latency"][key]
            v_wi = wr["latency"][key]
            delta = v_wi - v_no
            sign = "▲" if delta >= 0 else "▼"
            print(f"  {label:<20} {v_no:>{col}.3f}s {v_wi:>{col}.3f}s  {sign}{abs(delta):.3f}s")

        # ── Métricas IR ───────────────────────────────────────────────────
        top_k = settings.rag_top_k
        print(f"\n  {'Qualidade Retrieval (IR)':^{col*2+4}}")
        print(f"  {'Métrica':<20} {'SEM rerank':>{col}} {'COM rerank':>{col}}  Delta")
        print(f"  {'-'*20} {'-'*col} {'-'*col}  -------")
        for metric in [f"Recall@{top_k}", "MRR", f"nDCG@{top_k}", f"P@{top_k}"]:
            v_no = nr["ir_metrics"].get(metric, 0.0)
            v_wi = wr["ir_metrics"].get(metric, 0.0)
            delta = v_wi - v_no
            sign = "▲" if delta >= 0 else "▼"
            print(f"  {metric:<20} {_fmt(v_no):>{col}} {_fmt(v_wi):>{col}}  {sign}{abs(delta):.4f}")

        print(f"\n  Relevância flags  SEM: {nr['relevant_flags']}")
        print(f"  Relevância flags  COM: {wr['relevant_flags']}")

        # ── LLM-as-judge ─────────────────────────────────────────────────
        if run_judge and "judge" in nr:
            dims = ["groundedness", "relevancia", "coerencia", "fluencia"]
            print(f"\n  {'Qualidade Geração (LLM-as-judge 0–5)':^{col*2+4}}")
            print(f"  {'Dimensão':<20} {'SEM rerank':>{col}} {'COM rerank':>{col}}  Delta")
            print(f"  {'-'*20} {'-'*col} {'-'*col}  -------")
            for dim in dims:
                v_no = nr["judge"].get(dim, -1.0)
                v_wi = wr["judge"].get(dim, -1.0)
                if v_no < 0 or v_wi < 0:
                    print(f"  {dim:<20} {'N/A':>{col}} {'N/A':>{col}}")
                    continue
                delta = v_wi - v_no
                sign = "▲" if delta >= 0 else "▼"
                print(f"  {dim:<20} {v_no:>{col}.1f} {v_wi:>{col}.1f}  {sign}{abs(delta):.1f}")

            g_no = nr.get("grounding", {})
            g_wi = wr.get("grounding", {})
            gs_no = g_no.get("grounding_score", -1.0)
            gs_wi = g_wi.get("grounding_score", -1.0)
            if gs_no >= 0 and gs_wi >= 0:
                delta = gs_wi - gs_no
                sign = "▲" if delta >= 0 else "▼"
                print(f"  {'Grounding score':<20} {gs_no:>{col}.2f} {gs_wi:>{col}.2f}  {sign}{abs(delta):.2f}")

            print(f"\n  Justificativa SEM: {nr['judge'].get('justificativa','')[:80]}")
            print(f"  Justificativa COM: {wr['judge'].get('justificativa','')[:80]}")

        # ── Respostas geradas ─────────────────────────────────────────────
        print(f"\n  ┌─ Resposta SEM reranking {'─'*49}")
        for line in nr["answer"].splitlines():
            print(f"  │ {line}")
        print(f"  └{'─'*74}")

        print(f"\n  ┌─ Resposta COM reranking {'─'*49}")
        for line in wr["answer"].splitlines():
            print(f"  │ {line}")
        print(f"  └{'─'*74}")

        # ── Documentos selecionados ───────────────────────────────────────
        print(f"\n  {'Documentos no prompt':}")
        print(f"  {'#':<3} {'ID / Processo':<26} {'SEM rerank':<30} {'COM rerank':<30}")
        print(f"  {'-'*3} {'-'*26} {'-'*30} {'-'*30}")

        no_ids = [
            getattr(s, "numero_processo", None) or f"Tese {getattr(s,'tese_num','?')}"
            for s in nr["sources"]
        ]
        wi_ids = [
            getattr(s, "numero_processo", None) or f"Tese {getattr(s,'tese_num','?')}"
            for s in wr["sources"]
        ]
        max_rows = max(len(no_ids), len(wi_ids))
        for i in range(max_rows):
            n_id = no_ids[i] if i < len(no_ids) else ""
            w_id = wi_ids[i] if i < len(wi_ids) else ""
            marker = "✓" if i < len(nr["relevant_flags"]) and nr["relevant_flags"][i] else " "
            marker_w = "✓" if i < len(wr["relevant_flags"]) and wr["relevant_flags"][i] else " "
            print(f"  {i+1:<3} {'':26} {marker}{n_id:<29} {marker_w}{w_id:<29}")

        print()

    print(_SEP)


# ---------------------------------------------------------------------------
# Runner principal
# ---------------------------------------------------------------------------


async def run(question_ids: list[str], run_judge: bool) -> dict:
    with open(_DATASET_PATH, encoding="utf-8") as f:
        all_questions = json.load(f)["questions"]

    questions = [q for q in all_questions if q["id"] in question_ids]
    if not questions:
        raise ValueError(f"IDs não encontrados no dataset: {question_ids}")

    # Aquece modelos (carrega pesos uma única vez antes de medir)
    logger.info("Aquecendo modelos (semântico + cross-encoder)...")
    _warm_up()
    logger.info("Modelos prontos.")

    results = []
    async with aiosqlite.connect(settings.database_url) as conn:
        conn.row_factory = aiosqlite.Row
        for q in questions:
            r = await _evaluate_question(conn, q, run_judge)
            results.append(r)

    return {"questions": question_ids, "results": results}


def _warm_up() -> None:
    """Força o carregamento dos modelos antes dos benchmarks."""
    try:
        rerank_service._get_model()
    except Exception as e:
        logger.warning("Cross-encoder não carregado: %s — reranking usará fallback.", e)
    try:
        from src.services.semantic_service import _get_model as _sem_model
        _sem_model()
    except Exception as e:
        logger.warning("Modelo semântico não carregado: %s", e)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def _main(question_ids: list[str], run_judge: bool, output_dir: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    results = await run(question_ids, run_judge)
    _print_results(results["results"], run_judge)

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "rerank_eval_results.json"
    # Remove objetos não-serializáveis (sources são dataclasses) antes de salvar
    def _serializable(r: dict) -> dict:
        r = dict(r)
        for variant in ("no_rerank", "with_rerank"):
            if variant in r:
                v = dict(r[variant])
                v["sources"] = [
                    {
                        "id": getattr(s, "id", None),
                        "processo": getattr(s, "numero_processo", None),
                        "tipo": "acordao" if hasattr(s, "ementa") else "tese",
                    }
                    for s in v.get("sources", [])
                ]
                r[variant] = v
        return r
    serializable_results = [_serializable(r) for r in results["results"]]
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"questions": results["questions"], "results": serializable_results},
                  f, ensure_ascii=False, indent=2)
    print(f"\nResultados salvos em: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compara RAG híbrido sem vs com cross-encoder reranking"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="q01,q02",
        help="IDs das perguntas separados por vírgula (padrão: q01,q02)",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Pula a avaliação LLM-as-judge (mais rápido)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/results"),
    )
    args = parser.parse_args()
    ids = [i.strip() for i in args.questions.split(",")]
    asyncio.run(_main(ids, not args.no_judge, args.output_dir))
