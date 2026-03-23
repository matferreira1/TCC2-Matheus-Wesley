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
from src.services import ollama_service, groq_service, search_service, semantic_service

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


async def answer(conn: aiosqlite.Connection, question: str) -> RagResponse:
    """Executa o pipeline RAG híbrido: FTS5 + semântica → RRF → prompt → LLM."""
    logger.info("━━━━ Nova consulta RAG ━━━━")
    logger.info("Pergunta: %s", question)
    inicio = time.perf_counter()

    # Busca paralela: FTS5 (lexical) + semântica em acórdãos e teses
    _FETCH = 15  # candidatos de cada fonte antes do RRF
    (
        fts5_acordaos,
        fts5_teses,
        sem_acordaos,
        sem_teses,
    ) = await asyncio.gather(
        search_service.search(conn, question, top_k=_FETCH),
        search_service.search_teses(conn, question, top_k=_FETCH),
        semantic_service.search_semantic(conn, question, top_k=_FETCH),
        semantic_service.search_teses_semantic(conn, question, top_k=_FETCH),
    )

    # RRF: funde lexical + semântico e seleciona os melhores para o prompt
    sources = semantic_service.rrf_acordaos(fts5_acordaos, sem_acordaos, top_n=settings.rag_top_k)
    sources_teses = semantic_service.rrf_teses(fts5_teses, sem_teses, top_n=settings.rag_top_k_teses)

    logger.info(
        "Retrieval híbrido: %d acórdãos + %d teses (FTS5) | %d + %d (semântico) → RRF → %d + %d",
        len(fts5_acordaos), len(fts5_teses),
        len(sem_acordaos), len(sem_teses),
        len(sources), len(sources_teses),
    )

    if settings.llm_provider not in _VALID_LLM_PROVIDERS:
        raise ValueError(
            f"LLM_PROVIDER inválido: {settings.llm_provider!r}. "
            f"Valores aceitos: {', '.join(sorted(_VALID_LLM_PROVIDERS))}"
        )

    prompt = _build_prompt(question, sources, sources_teses)
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


def _sanitize_doc_text(text: str) -> str:
    """Remove padrões de prompt injection de textos vindos do banco."""
    return _INJECTION_RE.sub('[CONTEÚDO REMOVIDO]', text)


def _build_prompt(
    question: str,
    sources: list[search_service.SearchResult],
    sources_teses: list[search_service.TesesResult],
) -> str:
    """Monta o prompt RAG com contexto jurídico e pergunta do usuário."""
    context_parts: list[str] = []

    for i, s in enumerate(sources):
        payload = _sanitize_doc_text(
            _extract_ementa_payload(s.ementa, max_chars=settings.rag_max_ementa_chars)
        )
        context_parts.append(
            f"[Acórdão STF {i + 1}] {s.numero_processo}\n{payload}"
        )

    for i, t in enumerate(sources_teses):
        tese_texto = _sanitize_doc_text(t.tese_texto)
        if t.area == "SÚMULAS STJ":
            context_parts.append(
                f"[Súmula STJ {t.edicao_num}]\n{tese_texto}"
            )
        else:
            context_parts.append(
                f"[Tese STJ {i + 1}] {t.area} — Ed. {t.edicao_num}: {t.edicao_titulo} (Tese {t.tese_num})\n"
                f"{tese_texto}"
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
        "   - Acórdão STF: cite SOMENTE o número do processo (ex: HC 263552 AgR), que está na linha\n"
        "     imediatamente após o rótulo [Acórdão STF N]. NUNCA use o rótulo [Acórdão STF N] como citação.\n"
        "     Exemplo: '(HC 263552 AgR; HC 264610 AgR)'\n"
        "   - Tese STJ: copie o identificador COMPLETO que aparece após o rótulo [Tese STJ N],\n"
        "     incluindo área, edição e número da tese. NUNCA use formas abreviadas.\n"
        "     Exemplo: '(DIREITO CIVIL — Ed. 143: PLANO DE SAÚDE - III (Tese 3))'\n"
        "   - Súmula STJ: cite como 'Súmula NNN/STJ', usando o número que aparece no rótulo [Súmula STJ NNN].\n"
        "     Exemplo: '(Súmula 528/STJ; Súmula 302/STJ)'\n"
        "4. A frase 'Não encontrei informação suficiente nos documentos disponíveis.' deve ser usada SOMENTE "
        "como resposta única e completa, quando absolutamente nenhum documento contém informação relevante. "
        "NUNCA insira essa frase dentro de uma lista numerada.\n"
        "5. Responda em português, de forma objetiva e direta.\n\n"
        f"### Documentos:\n{context}\n\n"
        f"### Pergunta:\n{question}\n\n"
        "### Resposta:"
    )
