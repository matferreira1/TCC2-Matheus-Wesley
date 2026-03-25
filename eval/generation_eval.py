"""
Avaliação do módulo de geração (Generation) — seção 4.4.1 do TCC.

Implementa duas estratégias de avaliação:

1. LLM-as-judge (automática): usa o Groq para pontuar cada resposta em
   quatro dimensões da rubrica estruturada definida na seção 4.4.1:
     - Groundedness/Fidelidade (0–5): resposta apoiada exclusivamente nas fontes
     - Relevância            (0–5): resposta aborda o problema jurídico
     - Coerência             (0–5): consistência interna, sem contradições
     - Fluência              (0–5): clareza e legibilidade em português jurídico

2. Grounding check (determinístico): verifica se os números de processo
   citados na resposta correspondem às fontes recuperadas ("Testes de
   grounding" descritos na seção 4.4).

Uso:
    python -m eval.generation_eval [--top-k 5] [--questions q01,q02]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from pathlib import Path

import aiosqlite

from eval.retrieval_eval import is_relevant
from src.config.settings import settings
from src.services import groq_service, search_service
from src.services.rag_service import RagResponse, _build_prompt, answer

logger = logging.getLogger(__name__)

_DATASET_PATH = Path(__file__).parent / "dataset.json"

# Prompt do LLM-as-judge — avalia as 4 dimensões da rubrica do TCC
_JUDGE_PROMPT = """\
Você é um avaliador especializado em sistemas jurídicos de IA.
Avalie a resposta gerada pelo sistema nas quatro dimensões abaixo.

## Pergunta do usuário:
{question}

## Passagens recuperadas (contexto fornecido ao sistema):
{context}

## Resposta gerada pelo sistema:
{answer}

## Dimensões de avaliação (escala 0–5):

- **Groundedness/Fidelidade**: A resposta está estritamente apoiada nas passagens recuperadas, sem introduzir conteúdo não suportado ou inventado?
  0 = totalmente inventada | 5 = 100% apoiada nas fontes

- **Relevância**: A resposta aborda adequadamente a pergunta e o problema jurídico proposto?
  0 = completamente irrelevante | 5 = responde diretamente com precisão

- **Coerência**: A resposta é internamente consistente, sem contradições ou inconsistências lógicas?
  0 = contraditória | 5 = totalmente coerente

- **Fluência**: A resposta é clara, gramaticalmente correta e legível em português jurídico?
  0 = incompreensível | 5 = excelente legibilidade

Responda APENAS com um objeto JSON válido, sem texto adicional:
{{
  "groundedness": <0-5>,
  "relevancia": <0-5>,
  "coerencia": <0-5>,
  "fluencia": <0-5>,
  "justificativa": "<breve justificativa em 1-2 frases>"
}}"""


# ---------------------------------------------------------------------------
# Grounding check
# ---------------------------------------------------------------------------


def check_grounding(answer_text: str, sources: list) -> dict:
    """
    Verifica se as citações na resposta correspondem às fontes recuperadas.

    Implementa o 'Teste de grounding' descrito na seção 4.4:
    'verificação sistemática de que trechos citados nas respostas aparecem
    nas passagens recuperadas, bem como de que não há referências a fontes
    inexistentes.'

    Args:
        answer_text: texto da resposta gerada.
        sources: lista de SearchResult ou TesesResult retornados pelo RAG.

    Returns:
        Dict com total de citações, válidas, alucinadas e grounding_score.
    """
    # Extrai grupos de citação respeitando um nível de parênteses aninhados.
    # Padrão: ( conteúdo-sem-parens  ( conteúdo-interno )  conteúdo-sem-parens )
    # Necessário para teses STJ cujo identificador contém "(Tese N)" ao final.
    citation_groups = re.findall(
        r"\(([^()]*(?:\([^()]*\)[^()]*)*)\)", answer_text
    )

    # Constrói conjunto de identificadores válidos das fontes recuperadas.
    # Acórdãos STF → número do processo (ex: "HC 263552 AgR").
    # Teses STJ    → identificador completo no formato usado pelo prompt
    #               (ex: "DIREITO PENAL — Ed. 32: PRISÃO PREVENTIVA (Tese 8)").
    valid_ids: set[str] = set()
    for s in sources:
        if getattr(s, "numero_processo", None):
            valid_ids.add(s.numero_processo.lower().strip())
        elif all(hasattr(s, a) for a in ("area", "edicao_num", "edicao_titulo", "tese_num")):
            if s.area == "SÚMULAS STJ":
                valid_ids.add(f"súmula {s.edicao_num}/stj")
                valid_ids.add(f"súmula n. {s.edicao_num}")
                valid_ids.add(f"súmula {s.edicao_num}")
            else:
                tese_id = f"{s.area} — Ed. {s.edicao_num}: {s.edicao_titulo} (Tese {s.tese_num})"
                valid_ids.add(tese_id.lower().strip())

    valid: list[str] = []
    hallucinated: list[str] = []

    for group in citation_groups:
        # Cada grupo pode conter múltiplas citações separadas por ";"
        for citation in (c.strip() for c in group.split(";")):
            if len(citation) < 3:
                continue
            c_lower = citation.lower()
            matched = any(src_id in c_lower or c_lower in src_id for src_id in valid_ids)
            if matched:
                valid.append(citation)
            else:
                hallucinated.append(citation)

    total = len(valid) + len(hallucinated)
    grounding_score = len(valid) / total if total > 0 else 1.0

    return {
        "citations_total": total,
        "citations_valid": len(valid),
        "citations_hallucinated": len(hallucinated),
        "hallucinated_list": hallucinated,
        "grounding_score": round(grounding_score, 4),
    }


# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------


async def _judge(question: str, context: str, answer_text: str) -> dict:
    """
    Avalia a resposta com LLM-as-judge (Groq).

    Retorna dict com as 4 pontuações e justificativa, ou scores -1
    em caso de falha de parsing.
    """
    prompt = _JUDGE_PROMPT.format(
        question=question,
        context=context[:3000],  # limita para não estourar contexto do judge
        answer=answer_text,
    )
    try:
        raw = await groq_service.generate(prompt)
        # Extrai JSON mesmo que o modelo adicione texto antes/depois
        match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
        if not match:
            raise ValueError("JSON não encontrado na resposta do judge")
        scores = json.loads(match.group())
        # Normaliza chaves
        return {
            "groundedness": float(scores.get("groundedness", -1)),
            "relevancia": float(scores.get("relevancia", -1)),
            "coerencia": float(scores.get("coerencia", -1)),
            "fluencia": float(scores.get("fluencia", -1)),
            "justificativa": scores.get("justificativa", ""),
            "judge_ok": True,
        }
    except Exception as exc:
        logger.warning("Judge falhou para esta consulta: %s", exc)
        return {
            "groundedness": -1,
            "relevancia": -1,
            "coerencia": -1,
            "fluencia": -1,
            "justificativa": f"ERRO: {exc}",
            "judge_ok": False,
        }


# ---------------------------------------------------------------------------
# Avaliação por consulta
# ---------------------------------------------------------------------------


async def _evaluate_question(
    conn: aiosqlite.Connection,
    question_data: dict,
) -> dict:
    """Executa o pipeline RAG e avalia a resposta gerada."""
    question = question_data["question"]

    # Executa pipeline RAG completo
    rag_resp: RagResponse = await answer(conn, question)

    # Monta contexto textual (igual ao que foi enviado ao LLM)
    context_parts = []
    for s in rag_resp.sources:
        context_parts.append(f"[STF] {s.numero_processo}: {s.ementa[:500]}")
    for t in rag_resp.sources_teses:
        context_parts.append(f"[STJ Ed.{t.edicao_num} T{t.tese_num}]: {t.tese_texto[:500]}")
    context = "\n\n".join(context_parts) if context_parts else "Nenhum documento recuperado."

    # Avaliação automática com LLM-as-judge
    judge_scores = await _judge(question, context, rag_resp.answer)

    # Grounding check determinístico
    all_sources = list(rag_resp.sources) + list(rag_resp.sources_teses)
    grounding = check_grounding(rag_resp.answer, all_sources)

    return {
        "id": question_data["id"],
        "question": question,
        "area": question_data["area"],
        "answer": rag_resp.answer,
        "sources_retrieved": len(rag_resp.sources) + len(rag_resp.sources_teses),
        "judge_scores": judge_scores,
        "grounding_check": grounding,
    }


# ---------------------------------------------------------------------------
# Runner principal
# ---------------------------------------------------------------------------


def _mean_judge_scores(results: list[dict]) -> dict:
    """Agrega scores do judge excluindo consultas com falha (-1)."""
    dims = ["groundedness", "relevancia", "coerencia", "fluencia"]
    agg = {}
    for dim in dims:
        valid = [r["judge_scores"][dim] for r in results if r["judge_scores"][dim] >= 0]
        agg[dim] = sum(valid) / len(valid) if valid else -1.0
    return agg


async def run(question_ids: list[str] | None = None) -> dict:
    """
    Executa avaliação de geração para as consultas do dataset.

    Args:
        question_ids: lista de IDs (ex: ['q01', 'q02']). None = todas.

    Returns:
        Dict com resultados por consulta e scores médios agregados.
    """
    with open(_DATASET_PATH, encoding="utf-8") as f:
        all_questions = json.load(f)["questions"]

    if question_ids:
        questions = [q for q in all_questions if q["id"] in question_ids]
    else:
        questions = all_questions

    async with aiosqlite.connect(settings.database_url) as conn:
        conn.row_factory = aiosqlite.Row

        cur = await conn.execute("SELECT COUNT(*) FROM jurisprudencia")
        if (await cur.fetchone())[0] == 0:
            raise RuntimeError("Base de dados vazia. Execute o ETL antes de avaliar.")

        logger.info("Avaliação de geração: %d consultas...", len(questions))

        results = []
        for q in questions:
            r = await _evaluate_question(conn, q)
            results.append(r)
            scores = r["judge_scores"]
            logger.info(
                "[%s] G=%.1f R=%.1f C=%.1f F=%.1f | grounding=%.2f",
                q["id"],
                scores["groundedness"], scores["relevancia"],
                scores["coerencia"], scores["fluencia"],
                r["grounding_check"]["grounding_score"],
            )

    mean_scores = _mean_judge_scores(results)
    mean_grounding = sum(r["grounding_check"]["grounding_score"] for r in results) / len(results)

    return {
        "n_questions": len(questions),
        "results": results,
        "aggregate": {
            "judge_mean": mean_scores,
            "grounding_mean": round(mean_grounding, 4),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_table(results: list[dict]) -> None:
    dims = ["groundedness", "relevancia", "coerencia", "fluencia"]
    print("\n" + "─" * 76)
    print("  Avaliação de Geração — LLM-as-judge (0–5) + Grounding check")
    print("─" * 76)
    header = f"  {'ID':<6} {'Ground':>8} {'Relev':>8} {'Coer':>8} {'Fluên':>8} {'Grounding':>10}"
    print(header)
    print("─" * 76)
    for r in results:
        s = r["judge_scores"]
        g = r["grounding_check"]["grounding_score"]
        print(
            f"  {r['id']:<6} "
            f"{s['groundedness']:>8.1f} "
            f"{s['relevancia']:>8.1f} "
            f"{s['coerencia']:>8.1f} "
            f"{s['fluencia']:>8.1f} "
            f"{g:>10.3f}"
        )
    print("─" * 76)


async def _main(question_ids: list[str] | None, output_dir: Path) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    results = await run(question_ids=question_ids)

    _print_table(results["results"])

    agg = results["aggregate"]
    print("\n  Médias agregadas:")
    for dim, val in agg["judge_mean"].items():
        print(f"    {dim:<16}: {val:.3f}/5.0")
    print(f"    {'grounding_score':<16}: {agg['grounding_mean']:.3f}/1.0\n")

    output_path = output_dir / "generation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Resultados salvos em: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação de geração do IAJuris")
    parser.add_argument(
        "--questions",
        type=str,
        default=None,
        help="IDs separados por vírgula (ex: q01,q02). Padrão: todas",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("eval/results"))
    args = parser.parse_args()

    ids = args.questions.split(",") if args.questions else None
    asyncio.run(_main(ids, args.output_dir))
