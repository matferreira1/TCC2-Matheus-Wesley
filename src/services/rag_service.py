"""Orquestrador do pipeline RAG."""

from __future__ import annotations

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


async def answer(conn: aiosqlite.Connection, question: str) -> RagResponse:
    """Executa o pipeline RAG: busca FTS5 → monta prompt → gera resposta."""
    logger.info("━━━━ Nova consulta RAG ━━━━")
    logger.info("Pergunta: %s", question)
    inicio = time.perf_counter()

    sources = await search_service.search(conn, question, top_k=settings.rag_top_k)
    prompt = _build_prompt(question, sources)
    logger.info("Prompt enviado à LLM:\n%s\n%s\n%s", "─" * 60, prompt, "─" * 60)

    if settings.llm_provider == "groq":
        logger.info("Provedor LLM: Groq (%s)", settings.groq_model)
        text = await groq_service.generate(prompt)
    else:
        logger.info("Provedor LLM: Ollama (%s)", settings.ollama_model)
        text = await ollama_service.generate(prompt)

    elapsed = time.perf_counter() - inicio
    logger.info("Pipeline concluído em %.1fs", elapsed)
    return RagResponse(answer=text, sources=sources)


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
) -> str:
    """Monta o prompt RAG com contexto jurídico e pergunta do usuário."""
    if sources:
        context_parts = [
            f"[Acórdão {i+1}] {s.numero_processo}\n"
            + _extract_ementa_payload(s.ementa, max_chars=settings.rag_max_ementa_chars)
            for i, s in enumerate(sources)
        ]
        context = "\n\n".join(context_parts)
    else:
        context = "Nenhum acórdão relevante encontrado."

    return (
        "Você é um assistente jurídico especializado em jurisprudência brasileira.\n\n"
        "REGRAS OBRIGATÓRIAS:\n"
        "1. Use APENAS as informações dos acórdãos abaixo. Não invente nem extrapole.\n"
        "2. Identifique os temas comuns entre os acórdãos e SINTETIZE-os em poucos pontos claros.\n"
        "   Não liste cada acórdão separadamente — agrupe os que tratam do mesmo tema.\n"
        "3. Após cada ponto, cite TODOS os acórdãos que o sustentam entre parênteses, separados por ponto e vírgula.\n"
        "   Exemplo correto: '1. O STF não admite HC contra decisão monocrática do STJ por supressão de instância "
        "(HC 263552 AgR; HC 264610 AgR; HC 263660 AgR).'\n"
        "4. A frase 'Não encontrei informação suficiente nos acórdãos disponíveis.' deve ser usada SOMENTE "
        "como resposta única e completa, quando absolutamente nenhum acórdão contém informação relevante. "
        "NUNCA insira essa frase dentro de uma lista numerada.\n"
        "5. Responda em português, de forma objetiva e direta.\n\n"
        f"### Acórdãos:\n{context}\n\n"
        f"### Pergunta:\n{question}\n\n"
        "### Resposta:"
    )
