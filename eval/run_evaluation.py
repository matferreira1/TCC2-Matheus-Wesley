"""
Script principal de avaliação do IAJuris — seção 4.4 do TCC.

Orquestra todos os módulos de avaliação experimental:
  retrieval  → Recall@k, MRR, nDCG@k — FTS5 vs LIKE (seção 4.4.1)
  generation → Groundedness, Relevância, Coerência, Fluência + Grounding (seção 4.4.1)
  latency    → p95/p99, throughput, taxa de erro (seção 4.4.1)
  variants   → LLM sem RAG vs LIKE+LLM vs FTS5+LLM (seção 4.4.2)
  all        → executa todos os módulos em sequência

Uso:
    python -m eval.run_evaluation retrieval
    python -m eval.run_evaluation generation --questions q01,q02,q03
    python -m eval.run_evaluation latency --n-runs 10
    python -m eval.run_evaluation variants --questions q01,q02,q03,q04,q05
    python -m eval.run_evaluation all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

_OUTPUT_DIR = Path("eval/results")

logger = logging.getLogger(__name__)


async def _run_retrieval(top_k: int) -> None:
    from eval.retrieval_eval import _main as retrieval_main
    await retrieval_main(top_k=top_k, output_dir=_OUTPUT_DIR)


async def _run_generation(question_ids: list[str] | None) -> None:
    from eval.generation_eval import _main as generation_main
    await generation_main(question_ids=question_ids, output_dir=_OUTPUT_DIR)


async def _run_latency(n_runs: int) -> None:
    from eval.latency_eval import _main as latency_main
    await latency_main(n_runs=n_runs, output_dir=_OUTPUT_DIR)


async def _run_variants(question_ids: list[str] | None, top_k: int) -> None:
    from eval.compare_variants import _main as variants_main
    await variants_main(question_ids=question_ids, top_k=top_k, output_dir=_OUTPUT_DIR)


async def _run_all(top_k: int, n_runs: int) -> None:
    """Executa todos os módulos em sequência e gera relatório consolidado."""
    print("\n" + "═" * 60)
    print("  IAJuris — Avaliação Experimental Completa (seção 4.4)")
    print("═" * 60)

    consolidated: dict = {"top_k": top_k, "modules": {}}

    # 1. Retrieval
    print("\n[1/4] Avaliação de Retrieval...")
    from eval.retrieval_eval import run as retrieval_run
    r_results = await retrieval_run(top_k=top_k)
    consolidated["modules"]["retrieval"] = r_results["aggregate"]
    _save(r_results, "retrieval_results.json")

    # 2. Geração — subset das 10 primeiras para economizar tokens
    print("\n[2/4] Avaliação de Geração (10 consultas)...")
    from eval.generation_eval import run as generation_run
    q_ids = [f"q{i:02d}" for i in range(1, 11)]
    g_results = await generation_run(question_ids=q_ids)
    consolidated["modules"]["generation"] = g_results["aggregate"]
    _save(g_results, "generation_results.json")

    # 3. Latência
    print(f"\n[3/4] Avaliação de Latência ({n_runs} execuções)...")
    from eval.latency_eval import run as latency_run
    l_results = await latency_run(n_runs=n_runs)
    consolidated["modules"]["latency"] = l_results["latency_metrics"]
    consolidated["modules"]["operational"] = {
        "error_rate": l_results["error_rate"],
        "throughput_per_min": l_results["throughput_per_min"],
    }
    _save(l_results, "latency_results.json")

    # 4. Comparação de variantes — 5 consultas para economizar tokens
    print("\n[4/4] Comparação de Variantes (5 consultas)...")
    from eval.compare_variants import run as variants_run
    v_ids = ["q01", "q02", "q04", "q06", "q09"]
    v_results = await variants_run(question_ids=v_ids, top_k=top_k)
    consolidated["modules"]["variants"] = v_results["aggregate"]
    _save(v_results, "variants_results.json")

    # Relatório consolidado
    _save(consolidated, "consolidated_report.json")
    _print_consolidated(consolidated)


def _save(data: dict, filename: str) -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = _OUTPUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _print_consolidated(data: dict) -> None:
    print("\n" + "═" * 60)
    print("  RELATÓRIO CONSOLIDADO — IAJuris")
    print("═" * 60)

    ret = data["modules"].get("retrieval", {})
    if ret:
        print("\n  RECUPERAÇÃO")
        fts5 = ret.get("fts5", {})
        like = ret.get("like", {})
        top_k = data.get("top_k", 5)
        for metric in [f"Recall@{top_k}", "MRR", f"nDCG@{top_k}"]:
            v_f = fts5.get(metric, 0.0)
            v_l = like.get(metric, 0.0)
            print(f"    {metric:<12} FTS5={v_f:.4f}  LIKE={v_l:.4f}")

    gen = data["modules"].get("generation", {})
    if gen:
        print("\n  GERAÇÃO (LLM-as-judge, média)")
        judge = gen.get("judge_mean", {})
        for dim, val in judge.items():
            print(f"    {dim:<18}: {val:.3f}/5.0")
        print(f"    {'grounding_score':<18}: {gen.get('grounding_mean', 0):.3f}/1.0")

    lat = data["modules"].get("latency", {})
    if lat:
        print("\n  OPERAÇÃO")
        ops = data["modules"].get("operational", {})
        print(f"    Latência p95:    {lat.get('p95_s', '?')}s")
        print(f"    Latência p99:    {lat.get('p99_s', '?')}s")
        print(f"    Taxa de erro:    {ops.get('error_rate', 0):.1%}")
        print(f"    Throughput:      {ops.get('throughput_per_min', 0):.1f} req/min")

    var = data["modules"].get("variants", {})
    if var:
        print("\n  VARIANTES — Groundedness médio (0–5)")
        labels = {
            "no_rag": "LLM sem RAG",
            "like": "LIKE + LLM",
            "fts5": "FTS5 + LLM",
            "hybrid": "Híbrido + LLM",
        }
        for name, scores in var.items():
            label = labels.get(name, name)
            g = scores.get("groundedness", -1)
            r = scores.get("relevancia", -1)
            print(f"    {label:<16} G={g:.2f} R={r:.2f}")

    print("\n" + "═" * 60)
    print(f"  Resultados completos em: {_OUTPUT_DIR}/")
    print("═" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        prog="python -m eval.run_evaluation",
        description="Framework de avaliação experimental do IAJuris",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # retrieval
    p_ret = subparsers.add_parser("retrieval", help="Avaliação de recuperação (FTS5 vs LIKE)")
    p_ret.add_argument("--top-k", type=int, default=5)

    # generation
    p_gen = subparsers.add_parser("generation", help="Avaliação de geração (LLM-as-judge)")
    p_gen.add_argument("--questions", type=str, default=None, help="IDs separados por vírgula")

    # latency
    p_lat = subparsers.add_parser("latency", help="Avaliação de latência e throughput")
    p_lat.add_argument("--n-runs", type=int, default=20)

    # variants
    p_var = subparsers.add_parser("variants", help="Comparação de variantes (seção 4.4.2)")
    p_var.add_argument("--top-k", type=int, default=5)
    p_var.add_argument("--questions", type=str, default=None)

    # all
    p_all = subparsers.add_parser("all", help="Executa todos os módulos")
    p_all.add_argument("--top-k", type=int, default=5)
    p_all.add_argument("--n-runs", type=int, default=10)

    args = parser.parse_args()

    if args.command == "retrieval":
        asyncio.run(_run_retrieval(args.top_k))
    elif args.command == "generation":
        ids = args.questions.split(",") if args.questions else [f"q{i:02d}" for i in range(1, 11)]
        asyncio.run(_run_generation(ids))
    elif args.command == "latency":
        asyncio.run(_run_latency(args.n_runs))
    elif args.command == "variants":
        ids = args.questions.split(",") if args.questions else [f"q{i:02d}" for i in range(1, 11)]
        asyncio.run(_run_variants(ids, args.top_k))
    elif args.command == "all":
        asyncio.run(_run_all(args.top_k, args.n_runs))


if __name__ == "__main__":
    main()
