"""Orquestrador do pipeline RAG."""

from __future__ import annotations

import asyncio
import logging
import re
import textwrap
import time
from dataclasses import dataclass, field

import aiosqlite

from src.config.settings import settings
from src.services import ollama_service, groq_service, search_service, semantic_service, rerank_service

logger = logging.getLogger(__name__)

_VALID_LLM_PROVIDERS = frozenset({"groq", "ollama"})
_MAX_PROMPT_CHARS = 32_000  # aviso se prompt superar este limite

# Padrões de prompt injection em conteúdo do banco ou na pergunta do usuário
_INJECTION_RE = re.compile(
    r'(?i)(ignore|disregard|forget|bypass)\s.{0,40}(instruction|prompt|rule|directive)'
    r'|(system|admin)\s*(prompt|command|mode)',
)


@dataclass
class RagResponse:
    """Resposta completa do pipeline RAG."""

    answer: str
    sources: list[search_service.SearchResult] = field(default_factory=list)
    sources_teses: list[search_service.TesesResult] = field(default_factory=list)
    sources_sv: list[search_service.SumulaVinculanteResult] = field(default_factory=list)


async def answer(conn: aiosqlite.Connection, question: str) -> RagResponse:
    """Executa o pipeline RAG híbrido: FTS5 + semântica → RRF → prompt → LLM."""
    logger.info("━━━━ Nova consulta RAG ━━━━")
    logger.info("Pergunta: %s", question)
    inicio = time.perf_counter()

    # Busca paralela: FTS5 (lexical) + semântica em acórdãos, teses e SVs
    _FETCH = 15           # candidatos de cada fonte antes do RRF
    _RRF_CANDIDATES = 20  # candidatos pós-RRF enviados ao cross-encoder
    _FETCH_SV = 8         # SVs candidatas antes do RRF (corpus pequeno: ~60 docs)
    (
        fts5_acordaos,
        fts5_teses,
        fts5_sv,
        sem_acordaos,
        sem_teses,
        sem_sv,
    ) = await asyncio.gather(
        search_service.search(conn, question, top_k=_FETCH),
        search_service.search_teses(conn, question, top_k=_FETCH),
        search_service.search_sumulas_vinculantes(conn, question, top_k=_FETCH_SV),
        semantic_service.search_semantic(conn, question, top_k=_FETCH),
        semantic_service.search_teses_semantic(conn, question, top_k=_FETCH),
        semantic_service.search_sv_semantic(conn, question, top_k=_FETCH_SV),
    )

    # RRF: funde lexical + semântico → pool de candidatos para o reranker
    candidates_acordaos = semantic_service.rrf_acordaos(
        fts5_acordaos, sem_acordaos, top_n=_RRF_CANDIDATES
    )
    candidates_teses = semantic_service.rrf_teses(
        fts5_teses, sem_teses, top_n=_RRF_CANDIDATES
    )
    candidates_sv = semantic_service.rrf_sv(
        fts5_sv, sem_sv, top_n=3
    )

    # Cross-encoder reranking: pontua cada par (query, doc) e seleciona top_k
    if settings.reranker_enabled:
        sources = rerank_service.rerank(question, candidates_acordaos, top_n=settings.rag_top_k)
        sources_teses = rerank_service.rerank(question, candidates_teses, top_n=settings.rag_top_k_teses)
        sources_sv = rerank_service.rerank(question, candidates_sv, top_n=2)
    else:
        sources = candidates_acordaos[:settings.rag_top_k]
        sources_teses = candidates_teses[:settings.rag_top_k_teses]
        sources_sv = candidates_sv[:2]

    logger.info(
        "Retrieval híbrido: %d acórdãos + %d teses + %d SVs (FTS5)"
        " | %d + %d + %d (semântico)"
        " → RRF %d + %d + %d → rerank → %d + %d + %d",
        len(fts5_acordaos), len(fts5_teses), len(fts5_sv),
        len(sem_acordaos), len(sem_teses), len(sem_sv),
        len(candidates_acordaos), len(candidates_teses), len(candidates_sv),
        len(sources), len(sources_teses), len(sources_sv),
    )

    if settings.llm_provider not in _VALID_LLM_PROVIDERS:
        raise ValueError(
            f"LLM_PROVIDER inválido: {settings.llm_provider!r}. "
            f"Valores aceitos: {', '.join(sorted(_VALID_LLM_PROVIDERS))}"
        )

    prompt = _build_prompt(question, sources, sources_teses, sources_sv)
    logger.debug("Prompt enviado à LLM:\n%s\n%s\n%s", "─" * 60, prompt, "─" * 60)

    if len(prompt) > _MAX_PROMPT_CHARS:
        logger.warning(
            "Prompt muito longo: %d chars (limite recomendado: %d).",
            len(prompt), _MAX_PROMPT_CHARS,
        )

    if settings.llm_provider == "groq":
        logger.info("Provedor LLM: Groq (%s)", settings.groq_model)
        text = await groq_service.generate(prompt)
    else:
        logger.info("Provedor LLM: Ollama (%s)", settings.ollama_model)
        text = await ollama_service.generate(prompt)

    # Descarta fontes cujo conteúdo não se reflete na resposta gerada
    if settings.reranker_enabled:
        sources, sources_teses, sources_sv = _filter_cited_sources(
            text, sources, sources_teses, sources_sv
        )

    elapsed = time.perf_counter() - inicio
    logger.info("Pipeline concluído em %.1fs", elapsed)
    return RagResponse(answer=text, sources=sources, sources_teses=sources_teses, sources_sv=sources_sv)


def _filter_cited_sources(
    answer: str,
    sources: list[search_service.SearchResult],
    sources_teses: list[search_service.TesesResult],
    sources_sv: list[search_service.SumulaVinculanteResult],
) -> tuple[
    list[search_service.SearchResult],
    list[search_service.TesesResult],
    list[search_service.SumulaVinculanteResult],
]:
    """
    Descarta fontes cujo conteúdo não se reflete na resposta gerada pelo LLM.

    Aplica o cross-encoder sobre pares (resposta, fonte) e retém apenas os
    documentos com score >= threshold. A ordem original (RRF + reranking) é
    preservada — sem nova ordenação.

    Fallback: se nenhuma fonte passar o threshold (ex.: resposta genérica ou
    "não encontrei informação"), retorna as listas originais para evitar que a
    UI mostre zero fontes.
    """
    filtered_s = rerank_service.filter_by_answer(answer, sources)
    filtered_t = rerank_service.filter_by_answer(answer, sources_teses)
    filtered_sv = rerank_service.filter_by_answer(answer, sources_sv)

    if not filtered_s and not filtered_t and not filtered_sv:
        logger.info(
            "_filter_cited_sources: nenhuma fonte passou o threshold — "
            "retornando listas originais (fallback)."
        )
        return sources, sources_teses, sources_sv

    return filtered_s, filtered_t, filtered_sv


def _extract_ementa_payload(ementa: str, max_chars: int = 1500) -> str:
    """
    Extrai as partes juridicamente relevantes de uma ementa STF.

    As ementas do STF seguem estrutura padronizada com seções em algarismos
    romanos: I. CASO EM EXAME → II. QUESTÃO EM DISCUSSÃO →
    III. RAZÕES DE DECIDIR → IV. DISPOSITIVO.

    Estratégia de extração:
    1. Se a ementa cabe inteira no limite, retorna completa.
    2. Caso contrário, prioriza: cabeçalho (antes da seção I) +
       III. RAZÕES DE DECIDIR + IV. DISPOSITIVO — as seções com o
       payload semântico juridicamente relevante.
    3. Fallback: cabeçalho + tail (últimos chars), marcando omissão.
    """
    if len(ementa) <= max_chars:
        return ementa

    # Localiza início de cada seção romana (I., II., III., IV., V.)
    section_re = re.compile(r'(?<![A-Za-z])(I{1,3}V?|VI?)\. ')
    positions: dict[str, int] = {}
    for m in section_re.finditer(ementa):
        key = m.group(1)
        if key not in positions:  # guarda apenas a primeira ocorrência
            positions[key] = m.start()

    parts: list[str] = []

    # Cabeçalho: texto antes da seção "I"
    header_end = positions.get('I', min(300, len(ementa)))
    parts.append(ementa[:header_end].strip())

    # Seção III — RAZÕES DE DECIDIR (payload principal)
    if 'III' in positions:
        start_iii = positions['III']
        end_iii = positions.get('IV', len(ementa))
        parts.append(ementa[start_iii:end_iii].strip())

    # Seção IV — DISPOSITIVO / CONCLUSÃO
    if 'IV' in positions:
        parts.append(ementa[positions['IV']:].strip())

    # Fallback: ementa sem estrutura de seções romanas — trunca com shorten
    if 'III' not in positions and 'IV' not in positions:
        return textwrap.shorten(ementa, width=max_chars, placeholder='...')

    extracted = ' [...] '.join(p for p in parts if p)

    # Se ainda exceder o limite (seção III muito longa), trunca preservando
    # o início do raciocínio e o dispositivo final
    if len(extracted) > max_chars and 'IV' in positions:
        dispositivo = ementa[positions['IV']:].strip()
        budget_iii = max_chars - len(parts[0]) - len(dispositivo) - 20
        if budget_iii > 100 and 'III' in positions:
            start_iii = positions['III']
            end_iii = positions.get('IV', len(ementa))
            razoes_trunc = textwrap.shorten(
                ementa[start_iii:end_iii], width=budget_iii, placeholder='...'
            )
            extracted = f"{parts[0]} [...] {razoes_trunc} [...] {dispositivo}"
        else:
            extracted = textwrap.shorten(extracted, width=max_chars, placeholder='...')

    return extracted


def _sanitize_doc_text(text: str) -> str:
    """Remove padrões de prompt injection de textos vindos do banco."""
    return _INJECTION_RE.sub('[CONTEÚDO REMOVIDO]', text)


def _build_prompt(
    question: str,
    sources: list[search_service.SearchResult],
    sources_teses: list[search_service.TesesResult],
    sources_sv: list[search_service.SumulaVinculanteResult] | None = None,
) -> str:
    """Monta o prompt RAG v7 com contexto jurídico e pergunta do usuário.

    v7 — adiciona Súmulas Vinculantes STF como fonte de maior hierarquia:
    - Bloco [Súmula Vinculante STF N] com linha "Efeito: vinculante constitucional"
      (art. 103-A CF) — acima das Teses STJ e acórdãos casuísticos.
    - Regra de citação específica: citar como 'SV N/STF'.
    - Nota de fontes atualizada com a hierarquia completa de 4 níveis.
    """
    if sources_sv is None:
        sources_sv = []

    context_parts: list[str] = []

    # SVs primeiro — hierarquia máxima
    for sv in sources_sv:
        enunciado = _sanitize_doc_text(sv.enunciado)
        context_parts.append(
            f"[Súmula Vinculante STF {sv.numero}]\n"
            f"Efeito: vinculante constitucional — obrigatória para todo o Judiciário e a "
            f"administração pública (art. 103-A CF + Lei 11.417/2006).\n"
            f"{enunciado}"
        )

    for i, s in enumerate(sources):
        payload = _sanitize_doc_text(
            _extract_ementa_payload(s.ementa, max_chars=settings.rag_max_ementa_chars)
        )
        context_parts.append(
            f"[Acórdão STF {i + 1}] {s.numero_processo}\n"
            f"Efeito: decisão casuística — persuasiva, salvo se firmada com repercussão geral.\n"
            f"{payload}"
        )

    for i, t in enumerate(sources_teses):
        tese_texto = _sanitize_doc_text(t.tese_texto)
        if t.area == "SÚMULAS STJ":
            context_parts.append(
                f"[Súmula STJ {t.edicao_num}]\n"
                f"Efeito: enunciado persuasivo — consolidado, mas não vinculante.\n"
                f"{tese_texto}"
            )
        else:
            context_parts.append(
                f"[Tese STJ {i + 1}] {t.area} — Ed. {t.edicao_num}: {t.edicao_titulo} (Tese {t.tese_num})\n"
                f"Efeito: precedente qualificado — deve ser observado por todos os tribunais (art. 927, III, CPC).\n"
                f"{tese_texto}"
            )

    if context_parts:
        context = "\n\n".join(context_parts)
        has_sv = bool(sources_sv)
        has_acordaos = bool(sources)
        has_teses = bool(sources_teses)
        partes: list[str] = []
        if has_sv:
            partes.append("súmulas vinculantes do STF")
        if has_acordaos:
            partes.append("acórdãos do STF")
        if has_teses:
            partes.append("teses consolidadas do STJ")
        fontes_desc = " e ".join(partes) if partes else "documentos disponíveis"
    else:
        context = "Nenhum documento relevante encontrado."
        fontes_desc = "documentos disponíveis"

    return (
        "Você é um assistente jurídico especializado em jurisprudência brasileira.\n\n"
        "REGRAS OBRIGATÓRIAS:\n"
        f"1. Use APENAS as informações dos {fontes_desc} abaixo. Não invente nem extrapole.\n"
        "2. Identifique os temas comuns e SINTETIZE-os em poucos pontos claros.\n"
        "   Não liste cada documento separadamente — agrupe os que tratam do mesmo tema.\n"
        "   DIVERGÊNCIA: se dois ou mais documentos sustentam entendimentos opostos sobre\n"
        "   o mesmo ponto, NÃO os sintetize como consenso — registre a divergência:\n"
        "   'Há divergência entre os documentos: [fonte A] entende X, enquanto [fonte B] entende Y.'\n"
        "3. Após cada ponto, cite TODOS os documentos que o sustentam entre parênteses, separados por ponto e vírgula.\n"
        "   - Súmula Vinculante STF: cite como 'SV N/STF', usando o número do rótulo [Súmula Vinculante STF N].\n"
        "     Exemplo: '(SV 11/STF; SV 14/STF)'\n"
        "   - Acórdão STF: cite SOMENTE o número do processo (ex: HC 263552 AgR), que está na linha\n"
        "     imediatamente após o rótulo [Acórdão STF N]. NUNCA use o rótulo [Acórdão STF N] como citação.\n"
        "     Exemplo: '(HC 263552 AgR; HC 264610 AgR)'\n"
        "   - Tese STJ: copie o identificador COMPLETO que aparece após o rótulo [Tese STJ N],\n"
        "     incluindo área, edição e número da tese. NUNCA use formas abreviadas.\n"
        "     Exemplo: '(DIREITO CIVIL — Ed. 143: PLANO DE SAÚDE - III (Tese 3))'\n"
        "   - Súmula STJ: cite como 'Súmula NNN/STJ', usando o número que aparece no rótulo [Súmula STJ NNN].\n"
        "     Exemplo: '(Súmula 528/STJ; Súmula 302/STJ)'\n"
        "4. Encerre a resposta com 'Nota sobre as fontes:' e descreva o peso jurídico\n"
        "   dos documentos utilizados, com base nas linhas 'Efeito:' de cada fonte:\n"
        "   - Súmulas Vinculantes STF (vinculante constitucional) são obrigatórias para todo o Judiciário\n"
        "     e a administração pública — o precedente de maior hierarquia (art. 103-A CF).\n"
        "   - Teses STJ (precedente qualificado) vinculam todos os tribunais (art. 927, III, CPC).\n"
        "   - Súmulas STJ (enunciado persuasivo) são consolidadas, mas não vinculantes.\n"
        "   - Acórdãos STF (decisão casuística) são persuasivos, salvo com repercussão geral.\n"
        "   Se a resposta se basear APENAS em acórdãos casuísticos, avise que o usuário\n"
        "   deve verificar se há tese consolidada ou súmula antes de usar em peças processuais.\n"
        "5. A frase 'Não encontrei informação suficiente nos documentos disponíveis.' deve ser usada SOMENTE "
        "como resposta única e completa, quando absolutamente nenhum documento contém informação relevante. "
        "NUNCA insira essa frase dentro de uma lista numerada.\n"
        "6. Responda em português, de forma objetiva e direta.\n\n"
        f"### Documentos:\n{context}\n\n"
        f"### Pergunta:\n{question}\n\n"
        "### Resposta:"
    )
