"""Orquestrador do pipeline RAG."""

from __future__ import annotations

import asyncio
import logging
import re
import textwrap
import time
import unicodedata
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


async def answer(
    conn: aiosqlite.Connection,
    question: str,
    required_terms: list[str] | None = None,
) -> RagResponse:
    """Executa o pipeline RAG híbrido: FTS5 + semântica → RRF → filtro → prompt → LLM."""
    logger.info("━━━━ Nova consulta RAG ━━━━")
    logger.info("Pergunta: %s", question)
    inicio = time.perf_counter()

    # Busca paralela: FTS5 (lexical) + semântica em acórdãos, teses e SVs
    _FETCH = 20           # candidatos de cada fonte antes do RRF
    _RRF_CANDIDATES = 15  # candidatos pós-RRF enviados ao cross-encoder
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

    # Bigram coverage re-ranking: promove candidatos que cobrem mais frases específicas
    # da pergunta, reduzindo docs de áreas distintas que só compartilham tokens genéricos.
    # Aplicado antes do cross-encoder para melhorar a qualidade do pool de entrada.
    candidates_acordaos = _rerank_by_bigram_coverage(question, candidates_acordaos)
    candidates_teses = _rerank_by_bigram_coverage(question, candidates_teses)

    # Filtro de termos obrigatórios (parâmetro API opcional): descarta candidatos
    # que não contêm todos os termos especificados explicitamente pelo chamador.
    if required_terms:
        candidates_acordaos = _apply_required_terms_filter(candidates_acordaos, required_terms)
        candidates_teses = _apply_required_terms_filter(candidates_teses, required_terms)
        candidates_sv = _apply_required_terms_filter(candidates_sv, required_terms)

    # Cross-encoder reranking: uma única chamada model.predict() para todos os grupos
    if settings.reranker_enabled:
        sources, sources_teses, sources_sv = rerank_service.rerank_multi(
            question,
            [
                (candidates_acordaos, settings.rag_top_k),
                (candidates_teses, settings.rag_top_k_teses),
                (candidates_sv, 2),
            ],
        )
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
    filtered_s, filtered_t, filtered_sv = rerank_service.filter_by_answer_multi(
        answer, [sources, sources_teses, sources_sv]
    )

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

    # Fallback: ementa sem estrutura de seções romanas (ex: acórdãos STJ).
    # Preserva quebras de linha — importante para o formato "Rel. X.\n\nEMENTA\n\nTESE JURÍDICA: ..."
    if 'III' not in positions and 'IV' not in positions:
        return ementa[:max_chars - 3] + "..." if len(ementa) > max_chars else ementa

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


def _normalize_text(text: str) -> str:
    """Lowercase + remove diacríticos para comparação lexical insensível a acentos."""
    return unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode().lower()


# Stopwords normalizadas (sem acentos) — usadas no tokenizador de bigramas
_BIGRAM_STOPWORDS = frozenset({
    "o", "a", "os", "as", "um", "uma", "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas", "por", "para", "com", "que", "se",
    "ao", "aos", "e", "ou", "mas", "mais", "nao", "isso", "isto", "aqui",
    "ali", "ja", "tambem", "segundo", "seu", "sua", "seus", "suas",
    "me", "te", "lhe", "eh", "ha", "ate", "ser", "ter",
})


def _tokenize_for_bigrams(question: str) -> list[str]:
    """Extrai tokens normalizados da pergunta para formação de bigramas."""
    normalized = _normalize_text(question)
    tokens = re.sub(r'[^\w\s]', ' ', normalized).split()
    return [t for t in tokens if t not in _BIGRAM_STOPWORDS and len(t) > 2]


def _get_doc_text(doc) -> str:
    """Retorna o texto principal de qualquer tipo de documento."""
    if isinstance(doc, search_service.SearchResult):
        return doc.ementa
    if isinstance(doc, search_service.TesesResult):
        return doc.tese_texto
    if isinstance(doc, search_service.SumulaVinculanteResult):
        return doc.enunciado
    return ""


def _rerank_by_bigram_coverage(question: str, docs: list) -> list:
    """
    Re-ordena candidatos pós-RRF usando cobertura de bigramas — Opção 3 (gate + soft rerank).

    Duas etapas:

    1. Gate hard (cobertura == 0): descarta candidatos que não contêm nenhum
       bigrama da pergunta — documentos de áreas completamente alheias ao tema
       (ex.: "prestação de contas" numa query sobre fraude bancária).
       Fallback: se todos tiverem cobertura zero, o gate é ignorado para não
       esvaziar o pool.

    2. Soft rerank: 40% posição RRF  +  60% cobertura de bigramas.
       O peso maior na cobertura permite que um documento de posição inferior
       com frases-chave específicas (ex.: "sem grave ameaça") ultrapasse um
       documento de posição superior com cobertura baixa.
       As posições originais do RRF são preservadas no cálculo (não re-indexadas
       após o gate), mantendo o sinal de qualidade do RRF.
    """
    if len(docs) <= 1:
        return docs

    tokens = _tokenize_for_bigrams(question)
    if len(tokens) < 2:
        return docs

    bigrams = [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]

    # Calcula cobertura preservando posição original do RRF
    with_coverage = [
        (pos, doc, sum(1 for bg in bigrams if bg in _normalize_text(_get_doc_text(doc))) / len(bigrams))
        for pos, doc in enumerate(docs)
    ]

    # Gate hard: remove cobertura == 0 (sem nenhum bigrama da query no texto)
    filtered = [(pos, doc, cov) for pos, doc, cov in with_coverage if cov > 0.0]
    removed = len(with_coverage) - len(filtered)
    if not filtered:
        filtered = with_coverage  # fallback: todos têm cobertura zero
        removed = 0
    if removed:
        logger.info(
            "_rerank_by_bigram_coverage: gate removeu %d doc(s) com cobertura 0%%", removed
        )

    # Soft rerank: 40% pos RRF + 60% cobertura (posições originais mantidas)
    scored = [
        (0.40 * (1.0 / (pos + 1)) + 0.60 * cov, pos, doc)
        for pos, doc, cov in filtered
    ]
    scored.sort(key=lambda x: (-x[0], x[1]))
    logger.debug(
        "_rerank_by_bigram_coverage: %d bigramas | gate=%d removidos | top-3 scores: %s",
        len(bigrams), removed,
        [f"{s:.3f}" for s, _, _ in scored[:3]],
    )
    return [d for _, _, d in scored]


def _apply_required_terms_filter(
    docs: list,
    required_terms: list[str],
) -> list:
    """
    Filtra candidatos que não contêm todos os termos obrigatórios.

    A comparação é feita após normalização (minúsculas, sem acentos), então
    "prisão domiciliar" casa com "prisao domiciliar" no texto do documento.

    Fallback: se nenhum documento passar (todos foram eliminados), retorna a
    lista original para evitar contexto vazio no prompt.
    """
    if not required_terms or not docs:
        return docs

    norm_terms = [_normalize_text(t) for t in required_terms if t.strip()]
    if not norm_terms:
        return docs

    def _doc_text(doc) -> str:
        if isinstance(doc, search_service.SearchResult):
            return _normalize_text(doc.ementa)
        if isinstance(doc, search_service.TesesResult):
            return _normalize_text(doc.tese_texto)
        if isinstance(doc, search_service.SumulaVinculanteResult):
            return _normalize_text(doc.enunciado)
        return ""

    kept = [d for d in docs if all(t in _doc_text(d) for t in norm_terms)]

    if not kept:
        logger.info(
            "_apply_required_terms_filter: nenhum doc passou — retornando lista original (fallback) | termos=%s",
            required_terms,
        )
        return docs

    logger.info(
        "_apply_required_terms_filter: %d/%d docs retidos | termos=%s",
        len(kept), len(docs), required_terms,
    )
    return kept


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
        orgao = f" | {s.orgao_julgador}" if s.orgao_julgador else ""
        if s.tribunal == "STJ":
            if s.repercussao_geral:
                efeito = (
                    "recurso repetitivo — vinculante para os tribunais de origem "
                    "(art. 927, III, CPC; Res. STJ 8/2008)."
                )
            else:
                efeito = "precedente persuasivo — consolida a interpretação da legislação federal."
        else:
            if s.repercussao_geral:
                efeito = (
                    "decisão com repercussão geral — vinculante para os demais órgãos do "
                    "Poder Judiciário (art. 927, III, CPC)."
                )
            else:
                efeito = "decisão casuística — persuasiva, não vinculante."
        context_parts.append(
            f"[Acórdão {s.tribunal} {i + 1}] {s.numero_processo}{orgao}\n"
            f"Efeito: {efeito}\n"
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
            tribunais_acordaos = {s.tribunal for s in sources}
            if "STF" in tribunais_acordaos and "STJ" in tribunais_acordaos:
                partes.append("acórdãos do STF e do STJ")
            elif "STJ" in tribunais_acordaos:
                partes.append("acórdãos do STJ")
            else:
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
        "     imediatamente após o rótulo [Acórdão STF N]. NUNCA use o rótulo como citação.\n"
        "     Exemplo: '(HC 263552 AgR; RE 1.234.567)'\n"
        "   - Acórdão STJ: cite SOMENTE o número do processo (ex: REsp 1.926.749 ou AgInt nos EREsp 1926749),\n"
        "     que está na linha imediatamente após o rótulo [Acórdão STJ N]. NUNCA use o rótulo como citação.\n"
        "     Exemplo: '(REsp 1.234.567; HC 800.000)'\n"
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
