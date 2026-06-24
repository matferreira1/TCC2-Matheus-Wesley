"""
Coletor de métricas de operação e qualidade por consulta (telemetria ao vivo).

Cada consulta RAG bem-sucedida (ou com erro) gera um registro que é mantido em
um buffer em memória (janela das últimas N consultas) e persistido em
``data/metrics_log.jsonl`` para sobreviver a reinícios. O dashboard de métricas
(`/metrics`) consome o agregado via ``GET /api/v1/metrics``.

Métricas calculadas **ao vivo, sem LLM** (custo ~zero):
  - Operação:  latência (p50/p95/p99/média), throughput/min, taxa de erro.
  - Funil:     candidatos em cada estágio (FTS5/semântico → RRF → fontes finais).
  - Qualidade reference-free:
      * grounding por citação (check_grounding) — % de citações que batem com
        fontes reais recuperadas;
      * proxy de groundedness — similaridade de cosseno (MiniLM) entre a resposta
        e os textos das fontes;
      * proxy de relevância   — similaridade de cosseno entre pergunta e resposta.

Métricas que exigem gabarito (Recall@k, MRR, nDCG, P@k) ou LLM-judge ficam no
painel de avaliação offline (eval/results), não são calculadas por prompt.
"""

from __future__ import annotations

import json
import logging
import re
import statistics
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).resolve().parent.parent.parent
_LOG_PATH = _BASE_DIR / "data" / "metrics_log.jsonl"

_WINDOW = 500  # nº de consultas mantidas em memória para os agregados
_buffer: deque[dict[str, Any]] = deque(maxlen=_WINDOW)
_lifetime_count = 0


# ---------------------------------------------------------------------------
# Redação de dados pessoais (LGPD)
# ---------------------------------------------------------------------------
#
# A pergunta do usuário é a única entrada de texto livre da telemetria e pode
# conter dados pessoais (CPF, CNPJ, e-mail, telefone). Como o histórico é
# persistido em disco e exposto no dashboard público (/metrics), esses
# identificadores são mascarados ANTES de qualquer gravação — assim o dado
# sensível nunca chega ao log nem ao endpoint. O corpus jurídico em si é
# público (STF/STJ) e não é afetado por esta redação.

_PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"), "[CPF]"),
    (re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b"), "[CNPJ]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL]"),
    (re.compile(r"\b(?:\+?55\s?)?(?:\(?\d{2}\)?\s?)?9?\d{4}[-\s]?\d{4}\b"), "[TELEFONE]"),
]


def _redact_pii(text: str) -> str:
    """Mascara identificadores pessoais estruturados (CPF, CNPJ, e-mail, telefone)."""
    if not text:
        return text
    # CNPJ antes de CPF: o padrão de CPF é prefixo do de CNPJ e o capturaria parcialmente.
    for pattern, replacement in sorted(_PII_PATTERNS, key=lambda p: p[1] != "[CNPJ]"):
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# Persistência
# ---------------------------------------------------------------------------


def load_history() -> None:
    """Carrega o histórico persistido para o buffer (chamado no startup)."""
    global _lifetime_count
    if not _LOG_PATH.exists():
        return
    try:
        lines = _LOG_PATH.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.warning("Falha ao ler histórico de métricas: %s", exc)
        return
    _lifetime_count = len(lines)
    for line in lines[-_WINDOW:]:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Redige registros antigos gravados antes da máscara de PII.
        if isinstance(rec.get("question"), str):
            rec["question"] = _redact_pii(rec["question"])
        _buffer.append(rec)
    logger.info("Métricas: %d registros carregados (%d no total).", len(_buffer), _lifetime_count)


def _persist(record: dict[str, Any]) -> None:
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("Falha ao persistir métrica: %s", exc)


# ---------------------------------------------------------------------------
# Qualidade reference-free
# ---------------------------------------------------------------------------


def _check_grounding(answer_text: str, valid_ids: set[str]) -> dict[str, Any]:
    """% das citações entre parênteses que correspondem a fontes recuperadas."""
    citation_groups = re.findall(r"\(([^()]*(?:\([^()]*\)[^()]*)*)\)", answer_text)
    valid = hallucinated = 0
    for group in citation_groups:
        for citation in (c.strip() for c in group.split(";")):
            if len(citation) < 3:
                continue
            c_lower = citation.lower()
            if any(sid in c_lower or c_lower in sid for sid in valid_ids):
                valid += 1
            else:
                hallucinated += 1
    total = valid + hallucinated
    return {
        "citations_total": total,
        "citations_valid": valid,
        "citations_hallucinated": hallucinated,
        "grounding_score": round(valid / total, 4) if total else 1.0,
    }


def _embedding_proxies(question: str, answer_text: str, source_texts: list[str]) -> dict[str, float]:
    """Proxies de relevância e groundedness via cosseno (MiniLM já em memória)."""
    from src.services import semantic_service

    texts = [question, answer_text] + source_texts
    try:
        model = semantic_service._get_model()
        vecs = model.encode(texts, normalize_embeddings=True)
    except Exception as exc:  # modelo indisponível → proxies neutros
        logger.warning("Proxies de embedding indisponíveis: %s", exc)
        return {"relevance_proxy": -1.0, "groundedness_proxy": -1.0}

    q_vec, a_vec = vecs[0], vecs[1]
    relevance = float(q_vec @ a_vec)

    if len(vecs) > 2:
        sims = [float(a_vec @ s_vec) for s_vec in vecs[2:]]
        groundedness = sum(sims) / len(sims)
    else:
        groundedness = -1.0

    return {
        "relevance_proxy": round(relevance, 4),
        "groundedness_proxy": round(groundedness, 4),
    }


# ---------------------------------------------------------------------------
# Registro
# ---------------------------------------------------------------------------


async def record_query(question: str, resp: Any) -> None:
    """
    Calcula métricas de qualidade e registra uma consulta bem-sucedida.

    Pensado para rodar em background (FastAPI BackgroundTasks) — não bloqueia a
    resposta ao usuário. ``resp`` é o RagResponse retornado pelo pipeline.
    """
    import asyncio

    meta = getattr(resp, "meta", {}) or {}

    # IDs válidos das fontes para o cálculo de grounding por citação.
    valid_ids: set[str] = set()
    source_texts: list[str] = []
    for s in getattr(resp, "sources", []):
        if getattr(s, "numero_processo", None):
            valid_ids.add(s.numero_processo.lower().strip())
        if getattr(s, "ementa", None):
            source_texts.append(s.ementa)
    for t in getattr(resp, "sources_teses", []):
        tese_id = f"{t.area} — Ed. {t.edicao_num}: {t.edicao_titulo} (Tese {t.tese_num})"
        valid_ids.add(tese_id.lower().strip())
        if getattr(t, "tese_texto", None):
            source_texts.append(t.tese_texto)
    for sv in getattr(resp, "sources_sv", []):
        valid_ids.add(f"sv {sv.numero}".lower())
        valid_ids.add(f"súmula vinculante {sv.numero}".lower())
        if getattr(sv, "enunciado", None):
            source_texts.append(sv.enunciado)

    grounding = _check_grounding(resp.answer, valid_ids)
    # encode é CPU-bound → roda fora do event loop.
    proxies = await asyncio.to_thread(
        _embedding_proxies, question, resp.answer, source_texts[:8]
    )

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "success": True,
        "question": _redact_pii(question[:160]),
        "latency_s": meta.get("latency_s"),
        "provider": meta.get("provider"),
        "model": meta.get("model"),
        "funnel": meta.get("funnel"),
        "answer_chars": meta.get("answer_chars"),
        "prompt_chars": meta.get("prompt_chars"),
        "grounding": grounding,
        **proxies,
    }
    _append(record)


def record_error(question: str, error: str) -> None:
    """Registra uma consulta que falhou (chamado inline no handler de erro)."""
    _append(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "success": False,
            "question": _redact_pii(question[:160]),
            "error": error[:200],
        }
    )


def _append(record: dict[str, Any]) -> None:
    global _lifetime_count
    _buffer.append(record)
    _lifetime_count += 1
    _persist(record)


# ---------------------------------------------------------------------------
# Agregação (consumida pelo endpoint)
# ---------------------------------------------------------------------------


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = (len(s) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return round(s[lo] + (idx - lo) * (s[hi] - s[lo]), 3)


def _mean(values: list[float]) -> float:
    return round(statistics.mean(values), 4) if values else 0.0


def snapshot() -> dict[str, Any]:
    """Agrega as métricas da janela atual (últimas N consultas)."""
    records = list(_buffer)
    ok = [r for r in records if r.get("success")]
    errors = [r for r in records if not r.get("success")]

    latencies = [r["latency_s"] for r in ok if r.get("latency_s") is not None]
    groundings = [r["grounding"]["grounding_score"] for r in ok if r.get("grounding")]
    hallucinated = [r["grounding"]["citations_hallucinated"] for r in ok if r.get("grounding")]
    citations = [r["grounding"]["citations_total"] for r in ok if r.get("grounding")]
    relevances = [r["relevance_proxy"] for r in ok if r.get("relevance_proxy", -1) >= 0]
    groundeds = [r["groundedness_proxy"] for r in ok if r.get("groundedness_proxy", -1) >= 0]

    # Throughput: consultas no último minuto.
    now = time.time()
    last_min = 0
    for r in records:
        try:
            ts = datetime.fromisoformat(r["ts"]).timestamp()
            if now - ts <= 60:
                last_min += 1
        except (ValueError, KeyError):
            continue

    # Médias do funil de recuperação.
    def _avg_funnel(stage: str, kind: str) -> float:
        vals = [r["funnel"][stage][kind] for r in ok if r.get("funnel")]
        return round(_mean(vals), 1) if vals else 0.0

    providers = Counter(r.get("provider") for r in ok if r.get("provider"))

    return {
        "window_size": len(records),
        "lifetime_count": _lifetime_count,
        "operation": {
            "latency_p50_s": _percentile(latencies, 50),
            "latency_p95_s": _percentile(latencies, 95),
            "latency_p99_s": _percentile(latencies, 99),
            "latency_mean_s": _mean(latencies),
            "throughput_last_min": last_min,
            "error_rate": round(len(errors) / len(records), 4) if records else 0.0,
            "errors_total": len(errors),
            "providers": dict(providers),
        },
        "quality": {
            "grounding_mean": _mean(groundings),
            "citations_mean": _mean(citations),
            "hallucinated_mean": _mean(hallucinated),
            "relevance_proxy_mean": _mean(relevances),
            "groundedness_proxy_mean": _mean(groundeds),
        },
        "retrieval_funnel": {
            "fts5_acordaos": _avg_funnel("fts5", "acordaos"),
            "semantic_acordaos": _avg_funnel("semantic", "acordaos"),
            "rrf_acordaos": _avg_funnel("rrf", "acordaos"),
            "final_acordaos": _avg_funnel("final", "acordaos"),
            "final_teses": _avg_funnel("final", "teses"),
            "final_sv": _avg_funnel("final", "sv"),
        },
        "recent": [
            {
                "ts": r["ts"],
                "question": r.get("question", ""),
                "success": r.get("success"),
                "latency_s": r.get("latency_s"),
                "grounding_score": (r.get("grounding") or {}).get("grounding_score"),
                "relevance_proxy": r.get("relevance_proxy"),
                "groundedness_proxy": r.get("groundedness_proxy"),
            }
            for r in reversed(records[-12:])
        ],
    }
