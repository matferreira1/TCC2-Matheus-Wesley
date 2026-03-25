"""
Avaliação operacional: latência e throughput — seção 4.4.1 do TCC.

Métricas implementadas (camada operacional):
  - Latência p50, p95, p99, média, mín, máx (em segundos)
  - Throughput: consultas por minuto
  - Taxa de erro: proporção de chamadas com falha técnica

A avaliação percorre todas as consultas do dataset e mede o tempo
de resposta do pipeline RAG completo (FTS5 + geração LLM), reportando
os percentis de latência percebidos pelo usuário.

Uso:
    python -m eval.latency_eval [--n-runs 20]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

import aiosqlite

from src.config.settings import settings
from src.services.rag_service import answer

logger = logging.getLogger(__name__)

_DATASET_PATH = Path(__file__).parent / "dataset.json"


def _percentile(values: list[float], p: float) -> float:
    """Calcula o percentil p (0–100) de uma lista ordenada de valores."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = (p / 100) * (len(sorted_v) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_v) - 1)
    return sorted_v[lo] + (idx - lo) * (sorted_v[hi] - sorted_v[lo])


async def run(n_runs: int = 20) -> dict:
    """
    Executa n_runs consultas do dataset e coleta métricas de latência.

    Args:
        n_runs: número de consultas a executar. Se maior que o dataset,
                as consultas são repetidas ciclicamente.

    Returns:
        Dict com métricas de latência e throughput.
    """
    with open(_DATASET_PATH, encoding="utf-8") as f:
        questions = json.load(f)["questions"]

    # Seleciona consultas ciclicamente
    selected = [questions[i % len(questions)] for i in range(n_runs)]

    latencies: list[float] = []
    errors: int = 0
    individual: list[dict] = []

    async with aiosqlite.connect(settings.database_url) as conn:
        conn.row_factory = aiosqlite.Row

        cur = await conn.execute("SELECT COUNT(*) FROM jurisprudencia")
        if (await cur.fetchone())[0] == 0:
            raise RuntimeError("Base de dados vazia. Execute o ETL antes de avaliar.")

        logger.info("Avaliação de latência: %d execuções...", n_runs)
        wall_start = time.perf_counter()

        for i, q in enumerate(selected):
            t0 = time.perf_counter()
            try:
                resp = await answer(conn, q["question"])
                elapsed = time.perf_counter() - t0
                latencies.append(elapsed)
                individual.append({
                    "run": i + 1,
                    "question_id": q["id"],
                    "latency_s": round(elapsed, 3),
                    "answer_chars": len(resp.answer),
                    "sources_count": len(resp.sources) + len(resp.sources_teses),
                    "error": None,
                })
                logger.info("[%d/%d] %s → %.2fs", i + 1, n_runs, q["id"], elapsed)
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                errors += 1
                individual.append({
                    "run": i + 1,
                    "question_id": q["id"],
                    "latency_s": round(elapsed, 3),
                    "answer_chars": 0,
                    "sources_count": 0,
                    "error": str(exc),
                })
                logger.warning("[%d/%d] ERRO: %s", i + 1, n_runs, exc)

    wall_elapsed = time.perf_counter() - wall_start

    metrics: dict = {}
    if latencies:
        metrics = {
            "p50_s": round(_percentile(latencies, 50), 3),
            "p95_s": round(_percentile(latencies, 95), 3),
            "p99_s": round(_percentile(latencies, 99), 3),
            "mean_s": round(sum(latencies) / len(latencies), 3),
            "min_s": round(min(latencies), 3),
            "max_s": round(max(latencies), 3),
        }

    return {
        "n_runs": n_runs,
        "n_success": len(latencies),
        "n_errors": errors,
        "error_rate": round(errors / n_runs, 4) if n_runs > 0 else 0.0,
        "throughput_per_min": round(len(latencies) / (wall_elapsed / 60), 2) if latencies else 0.0,
        "wall_elapsed_s": round(wall_elapsed, 2),
        "latency_metrics": metrics,
        "individual": individual,
    }


def _print_report(results: dict) -> None:
    m = results["latency_metrics"]
    print("\n" + "─" * 48)
    print("  Avaliação Operacional — Latência e Throughput")
    print("─" * 48)
    print(f"  Execuções:       {results['n_runs']}")
    print(f"  Sucesso:         {results['n_success']}")
    print(f"  Erros:           {results['n_errors']}")
    print(f"  Taxa de erro:    {results['error_rate']:.1%}")
    print(f"  Throughput:      {results['throughput_per_min']:.1f} req/min")
    print("─" * 48)
    if m:
        print(f"  Latência p50:    {m['p50_s']:.3f}s")
        print(f"  Latência p95:    {m['p95_s']:.3f}s")
        print(f"  Latência p99:    {m['p99_s']:.3f}s")
        print(f"  Latência média:  {m['mean_s']:.3f}s")
        print(f"  Latência mín:    {m['min_s']:.3f}s")
        print(f"  Latência máx:    {m['max_s']:.3f}s")
    print("─" * 48 + "\n")


async def _main(n_runs: int, output_dir: Path) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    results = await run(n_runs=n_runs)

    _print_report(results)

    output_path = output_dir / "latency_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Resultados salvos em: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação de latência do IAJuris")
    parser.add_argument("--n-runs", type=int, default=20, help="Número de execuções")
    parser.add_argument("--output-dir", type=Path, default=Path("eval/results"))
    args = parser.parse_args()
    asyncio.run(_main(args.n_runs, args.output_dir))
