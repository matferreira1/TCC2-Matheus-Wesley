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
from src.services import ollama_service, groq_service, search_service

logger = logging.getLogger(__name__)


@dataclass
class RagResponse:
    """Resposta completa do pipeline RAG."""

    answer: str
    sources: list[search_service.SearchResult] = field(default_factory=list)
    sources_teses: list[search_service.TesesResult] = field(default_factory=list)


async def answer(conn: aiosqlite.Connection, question: str) -> RagResponse:
    """Executa o pipeline RAG: busca FTS5 (acórdãos + teses) → prompt → LLM."""
    logger.info("━━━━ Nova consulta RAG ━━━━")
    logger.info("Pergunta: %s", question)
    inicio = time.perf_counter()

    # Busca paralela nos dois índices FTS5
    sources, sources_teses = await asyncio.gather(
        search_service.search(conn, question, top_k=settings.rag_top_k),
        search_service.search_teses(conn, question, top_k=settings.rag_top_k_teses),
    )

    prompt = _build_prompt(question, sources, sources_teses)
    logger.info("Prompt enviado à LLM:\n%s\n%s\n%s", "─" * 60, prompt, "─" * 60)

    if settings.llm_provider == "groq":
        logger.info("Provedor LLM: Groq (%s)", settings.groq_model)
        text = await groq_service.generate(prompt)
    else:
        logger.info("Provedor LLM: Ollama (%s)", settings.ollama_model)
        text = await ollama_service.generate(prompt)

    elapsed = time.perf_counter() - inicio
    logger.info("Pipeline concluído em %.1fs", elapsed)
    return RagResponse(answer=text, sources=sources, sources_teses=sources_teses)


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


def _build_prompt(
    question: str,
    sources: list[search_service.SearchResult],
    sources_teses: list[search_service.TesesResult],
) -> str:
    """Monta o prompt RAG com contexto jurídico e pergunta do usuário."""
    context_parts: list[str] = []

    for i, s in enumerate(sources):
        payload = _extract_ementa_payload(s.ementa, max_chars=settings.rag_max_ementa_chars)
        context_parts.append(
            f"[Acórdão STF {i + 1}] {s.numero_processo}\n{payload}"
        )

    for i, t in enumerate(sources_teses):
        context_parts.append(
            f"[Tese STJ {i + 1}] {t.area} — Ed. {t.edicao_num}: {t.edicao_titulo} (Tese {t.tese_num})\n"
            f"{t.tese_texto}"
        )

    if context_parts:
        context = "\n\n".join(context_parts)
        has_acordaos = bool(sources)
        has_teses = bool(sources_teses)
        if has_acordaos and has_teses:
            fontes_desc = "acórdãos do STF e teses consolidadas do STJ"
        elif has_teses:
            fontes_desc = "teses consolidadas do STJ"
        else:
            fontes_desc = "acórdãos do STF"
    else:
        context = "Nenhum documento relevante encontrado."
        fontes_desc = "documentos disponíveis"

    return (
        "Você é um assistente jurídico especializado em jurisprudência brasileira.\n\n"
        "REGRAS OBRIGATÓRIAS:\n"
        f"1. Use APENAS as informações dos {fontes_desc} abaixo. Não invente nem extrapole.\n"
        "2. Identifique os temas comuns e SINTETIZE-os em poucos pontos claros.\n"
        "   Não liste cada documento separadamente — agrupe os que tratam do mesmo tema.\n"
        "3. Após cada ponto, cite TODOS os documentos que o sustentam entre parênteses, separados por ponto e vírgula.\n"
        "   Exemplo com acórdão: '(HC 263552 AgR; HC 264610 AgR)'\n"
        "   Exemplo com tese STJ: '(STJ Ed. 39 T3; STJ Ed. 42 T1)'\n"
        "   Exemplo misto: '(HC 263552 AgR; STJ Ed. 39 T3)'\n"
        "4. A frase 'Não encontrei informação suficiente nos documentos disponíveis.' deve ser usada SOMENTE "
        "como resposta única e completa, quando absolutamente nenhum documento contém informação relevante. "
        "NUNCA insira essa frase dentro de uma lista numerada.\n"
        "5. Responda em português, de forma objetiva e direta.\n\n"
        f"### Documentos:\n{context}\n\n"
        f"### Pergunta:\n{question}\n\n"
        "### Resposta:"
    )
