"""
Testes para o pipeline RAG (rag_service).

Cobre:
  - answer()               : orquestracao completa com mocks de busca e LLM
  - _build_prompt()        : montagem do prompt a partir das fontes
  - _extract_ementa_payload(): extracao inteligente de secoes das ementas STF
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from src.services.rag_service import (
    RagResponse,
    _build_prompt,
    _extract_ementa_payload,
    answer,
)
from src.services.search_service import SearchResult, TesesResult, SumulaVinculanteResult


# --------------------------------------------------------------------------
# Helpers — dados de mock
# --------------------------------------------------------------------------

def _make_acordao(
    id: int = 1,
    tribunal: str = "STF",
    numero: str = "HC 100001",
    ementa: str = "Habeas corpus. Prisao preventiva. Ordem concedida.",
) -> SearchResult:
    return SearchResult(id=id, tribunal=tribunal, numero_processo=numero,
                        ementa=ementa, rank=-1.0)


def _make_tese(
    id: int = 1,
    area: str = "DIREITO CIVIL",
    edicao_num: int = 143,
    edicao_titulo: str = "PLANO DE SAUDE",
    tese_num: int = 1,
    texto: str = "O plano de saude pode estabelecer as doencas com cobertura.",
) -> TesesResult:
    return TesesResult(
        id=id, area=area, edicao_num=edicao_num, edicao_titulo=edicao_titulo,
        tese_num=tese_num, tese_texto=texto, julgados="REsp 123/SP", rank=-1.0
    )


def _make_sv(
    id: int = 1,
    numero: int = 11,
    enunciado: str = "So e licito o uso de algemas em casos de resistencia.",
) -> SumulaVinculanteResult:
    return SumulaVinculanteResult(id=id, numero=numero, enunciado=enunciado, rank=-1.0)


# ===========================================================================
# Testes de answer() — orquestracao RAG
# ===========================================================================


async def test_answer_returns_rag_response(db: aiosqlite.Connection) -> None:
    """answer() deve retornar um RagResponse com answer e sources."""
    mock_sources = [_make_acordao()]
    mock_teses   = [_make_tese()]

    with (
        patch("src.services.rag_service.search_service.search",
              new_callable=AsyncMock, return_value=mock_sources),
        patch("src.services.rag_service.search_service.search_teses",
              new_callable=AsyncMock, return_value=mock_teses),
        patch("src.services.rag_service.settings") as mock_settings,
        patch("src.services.rag_service.groq_service.generate",
              new_callable=AsyncMock, return_value="Resposta gerada."),
    ):
        mock_settings.rag_top_k = 5
        mock_settings.rag_top_k_teses = 3
        mock_settings.rag_max_ementa_chars = 1500
        mock_settings.llm_provider = "groq"
        mock_settings.reranker_enabled = False

        result = await answer(db, "Quais os fundamentos para negar habeas corpus?")

    assert isinstance(result, RagResponse)
    assert isinstance(result.answer, str)
    assert result.answer != ""


async def test_answer_populates_sources(db: aiosqlite.Connection) -> None:
    """answer() deve incluir as fontes retornadas pelo search_service."""
    mock_sources = [_make_acordao(id=1), _make_acordao(id=2, numero="HC 100002")]
    mock_teses   = [_make_tese()]

    with (
        patch("src.services.rag_service.search_service.search",
              new_callable=AsyncMock, return_value=mock_sources),
        patch("src.services.rag_service.search_service.search_teses",
              new_callable=AsyncMock, return_value=mock_teses),
        patch("src.services.rag_service.settings") as mock_settings,
        patch("src.services.rag_service.groq_service.generate",
              new_callable=AsyncMock, return_value="Resposta."),
    ):
        mock_settings.rag_top_k = 5
        mock_settings.rag_top_k_teses = 3
        mock_settings.rag_max_ementa_chars = 1500
        mock_settings.llm_provider = "groq"
        mock_settings.reranker_enabled = False

        result = await answer(db, "Quais os fundamentos para negar habeas corpus?")

    assert len(result.sources) == 2
    assert len(result.sources_teses) == 1


async def test_answer_sources_empty_when_no_match(db: aiosqlite.Connection) -> None:
    """Se a busca nao retorna documentos, sources deve ser lista vazia."""
    with (
        patch("src.services.rag_service.search_service.search",
              new_callable=AsyncMock, return_value=[]),
        patch("src.services.rag_service.search_service.search_teses",
              new_callable=AsyncMock, return_value=[]),
        patch("src.services.rag_service.settings") as mock_settings,
        patch("src.services.rag_service.groq_service.generate",
              new_callable=AsyncMock, return_value="Nao encontrei informacao."),
    ):
        mock_settings.rag_top_k = 5
        mock_settings.rag_top_k_teses = 3
        mock_settings.rag_max_ementa_chars = 1500
        mock_settings.llm_provider = "groq"
        mock_settings.reranker_enabled = False

        result = await answer(db, "Pergunta sem resultados na base de dados?")

    assert result.sources == []
    assert result.sources_teses == []


async def test_answer_propagates_llm_error(db: aiosqlite.Connection) -> None:
    """Se o LLM lancar excecao, ela deve ser propagada pelo pipeline."""
    with (
        patch("src.services.rag_service.search_service.search",
              new_callable=AsyncMock, return_value=[_make_acordao()]),
        patch("src.services.rag_service.search_service.search_teses",
              new_callable=AsyncMock, return_value=[]),
        patch("src.services.rag_service.settings") as mock_settings,
        patch("src.services.rag_service.groq_service.generate",
              new_callable=AsyncMock, side_effect=RuntimeError("LLM error")),
    ):
        mock_settings.rag_top_k = 5
        mock_settings.rag_top_k_teses = 3
        mock_settings.rag_max_ementa_chars = 1500
        mock_settings.llm_provider = "groq"
        mock_settings.reranker_enabled = False

        with pytest.raises(RuntimeError, match="LLM error"):
            await answer(db, "Pergunta qualquer com erro no LLM?")


async def test_answer_uses_ollama_when_configured(db: aiosqlite.Connection) -> None:
    """Quando llm_provider=ollama, deve chamar ollama_service.generate."""
    with (
        patch("src.services.rag_service.search_service.search",
              new_callable=AsyncMock, return_value=[]),
        patch("src.services.rag_service.search_service.search_teses",
              new_callable=AsyncMock, return_value=[]),
        patch("src.services.rag_service.settings") as mock_settings,
        patch("src.services.rag_service.ollama_service.generate",
              new_callable=AsyncMock, return_value="Resposta Ollama.") as mock_ollama,
    ):
        mock_settings.rag_top_k = 5
        mock_settings.rag_top_k_teses = 3
        mock_settings.rag_max_ementa_chars = 1500
        mock_settings.llm_provider = "ollama"
        mock_settings.reranker_enabled = False

        result = await answer(db, "Pergunta para Ollama?")

    mock_ollama.assert_awaited_once()
    assert result.answer == "Resposta Ollama."


# ===========================================================================
# Testes de _build_prompt()
# ===========================================================================


def test_build_prompt_contains_question() -> None:
    """O prompt deve conter a pergunta do usuario."""
    question = "Quais os requisitos da prisao preventiva?"
    prompt = _build_prompt(question, [], [])
    assert question in prompt


def test_build_prompt_contains_acordao_label() -> None:
    """O prompt deve conter o label [Acordao STF N] para cada acordao."""
    sources = [_make_acordao(id=1, numero="HC 100001")]
    prompt = _build_prompt("Pergunta?", sources, [])
    assert "Ac" in prompt  # [Acórdão STF 1] ou variante
    assert "STF 1" in prompt
    assert "HC 100001" in prompt


def test_build_prompt_contains_tese_label() -> None:
    """O prompt deve conter o label [Tese STJ N] para cada tese."""
    teses = [_make_tese(id=1, edicao_num=143, tese_num=1)]
    prompt = _build_prompt("Pergunta?", [], teses)
    assert "[Tese STJ 1]" in prompt


def test_build_prompt_sumula_usa_label_sumula() -> None:
    """Tese com area=SÚMULAS STJ deve gerar [Súmula STJ N], não [Tese STJ N]."""
    sumula = TesesResult(
        id=1,
        area="SÚMULAS STJ",
        edicao_num=528,
        edicao_titulo="ENUNCIADOS DAS SÚMULAS",
        tese_num=1,
        tese_texto="Compete ao juiz federal do local da apreensão da droga.",
        julgados="",
        rank=-1.0,
    )
    prompt = _build_prompt("Pergunta?", [], [sumula])
    assert "[Súmula STJ 528]" in prompt


def test_build_prompt_sumula_nao_usa_label_tese() -> None:
    """Súmula STJ não deve gerar o label [Tese STJ N] no prompt."""
    sumula = TesesResult(
        id=1,
        area="SÚMULAS STJ",
        edicao_num=528,
        edicao_titulo="ENUNCIADOS DAS SÚMULAS",
        tese_num=1,
        tese_texto="Compete ao juiz federal do local da apreensão da droga.",
        julgados="",
        rank=-1.0,
    )
    prompt = _build_prompt("Pergunta?", [], [sumula])
    assert "[Tese STJ 1]" not in prompt


def test_build_prompt_sumula_contem_texto() -> None:
    """O texto da súmula deve aparecer no prompt."""
    texto = "Compete ao juiz federal do local da apreensão da droga."
    sumula = TesesResult(
        id=1,
        area="SÚMULAS STJ",
        edicao_num=528,
        edicao_titulo="ENUNCIADOS DAS SÚMULAS",
        tese_num=1,
        tese_texto=texto,
        julgados="",
        rank=-1.0,
    )
    prompt = _build_prompt("Pergunta?", [], [sumula])
    assert texto in prompt


def test_build_prompt_empty_sources_uses_fallback() -> None:
    """Sem fontes, o prompt deve indicar nenhum documento encontrado."""
    prompt = _build_prompt("Pergunta?", [], [])
    assert "Nenhum documento relevante encontrado" in prompt


def test_build_prompt_multiple_acordaos_numbered_correctly() -> None:
    """Acordaos devem ser numerados sequencialmente no prompt."""
    sources = [
        _make_acordao(id=1, numero="HC 100001"),
        _make_acordao(id=2, numero="HC 100002"),
        _make_acordao(id=3, numero="HC 100003"),
    ]
    prompt = _build_prompt("Pergunta?", sources, [])
    assert "STF 1" in prompt
    assert "STF 2" in prompt
    assert "STF 3" in prompt


def test_build_prompt_contains_mandatory_rules() -> None:
    """O prompt deve conter as regras obrigatorias da versao atual."""
    prompt = _build_prompt("Pergunta?", [_make_acordao()], [])
    assert "REGRAS OBRIGATORIAS" in prompt or "OBRIGAT" in prompt


def test_build_prompt_mixed_sources_description() -> None:
    """Com fontes mistas, a descricao deve mencionar STF e STJ."""
    prompt = _build_prompt("Pergunta?", [_make_acordao()], [_make_tese()])
    assert "STF" in prompt
    assert "STJ" in prompt


# ===========================================================================
# Testes do prompt v6 — divergência e peso jurídico das fontes
# ===========================================================================


def test_build_prompt_acordao_tem_anotacao_efeito() -> None:
    """Bloco de acórdão STF deve conter linha 'Efeito:' indicando decisão casuística."""
    prompt = _build_prompt("Pergunta?", [_make_acordao()], [])
    assert "Efeito:" in prompt
    assert "casuística" in prompt


def test_build_prompt_tese_tem_anotacao_efeito_precedente() -> None:
    """Bloco de Tese STJ deve conter linha 'Efeito:' indicando precedente qualificado."""
    prompt = _build_prompt("Pergunta?", [], [_make_tese()])
    assert "Efeito:" in prompt
    assert "precedente qualificado" in prompt
    assert "art. 927" in prompt


def test_build_prompt_sumula_tem_anotacao_efeito_persuasivo() -> None:
    """Bloco de Súmula STJ deve conter linha 'Efeito:' indicando enunciado persuasivo."""
    sumula = TesesResult(
        id=1,
        area="SÚMULAS STJ",
        edicao_num=528,
        edicao_titulo="ENUNCIADOS DAS SÚMULAS",
        tese_num=1,
        tese_texto="Compete ao juiz federal do local da apreensão da droga.",
        julgados="",
        rank=-1.0,
    )
    prompt = _build_prompt("Pergunta?", [], [sumula])
    assert "Efeito:" in prompt
    assert "persuasivo" in prompt


def test_build_prompt_contem_regra_divergencia() -> None:
    """O prompt deve conter instrução explícita sobre como tratar divergência entre fontes."""
    prompt = _build_prompt("Pergunta?", [_make_acordao()], [])
    assert "DIVERGÊNCIA" in prompt or "divergência" in prompt.lower()
    assert "NÃO os sintetize como consenso" in prompt or "NÃO" in prompt


def test_build_prompt_contem_regra_nota_fontes() -> None:
    """O prompt deve conter instrução para encerrar com 'Nota sobre as fontes:'."""
    prompt = _build_prompt("Pergunta?", [_make_acordao()], [_make_tese()])
    assert "Nota sobre as fontes" in prompt


# ===========================================================================
# Testes de _extract_ementa_payload()
# ===========================================================================


def test_extract_ementa_short_returns_as_is() -> None:
    """Ementa menor que max_chars deve ser retornada integralmente."""
    ementa = "Direito penal. Habeas corpus denegado."
    result = _extract_ementa_payload(ementa, max_chars=500)
    assert result == ementa


def test_extract_ementa_at_exact_limit_returns_as_is() -> None:
    """Ementa exatamente igual ao limite deve ser retornada sem alteracao."""
    ementa = "x" * 100
    result = _extract_ementa_payload(ementa, max_chars=100)
    assert result == ementa


def test_extract_ementa_with_sections_extracts_iii_and_iv() -> None:
    """Ementa com secoes romanas deve extrair prioritariamente III e IV."""
    ementa = (
        "HABEAS CORPUS. PRISAO PREVENTIVA. "
        "I. CASO EM EXAME: O paciente foi preso em flagrante delito. "
        "II. QUESTAO EM DISCUSSAO: Cabimento do HC contra decisao monocratica. "
        "III. RAZOES DE DECIDIR: A fundamentacao da prisao e ideonea e atual. "
        "Presenca dos requisitos do art. 312 do CPP. "
        "IV. DISPOSITIVO: Ordem denegada."
    )
    result = _extract_ementa_payload(ementa, max_chars=200)
    # O resultado deve conter as secoes relevantes
    assert "RAZOES DE DECIDIR" in result or "DISPOSITIVO" in result


def test_extract_ementa_no_sections_uses_shorten_fallback() -> None:
    """Ementa sem secoes romanas usa textwrap.shorten como fallback."""
    ementa = "palavra " * 500  # sem secoes romanas, bem maior que 100 chars
    result = _extract_ementa_payload(ementa, max_chars=100)
    assert len(result) <= 110  # shorten pode adicionar alguns chars do placeholder


def test_extract_ementa_preserves_dispositivo() -> None:
    """IV. DISPOSITIVO deve estar presente no resultado extraido."""
    ementa = (
        "CABECALHO LONGO " * 20
        + "III. RAZOES DE DECIDIR: fundamento juridico extenso " * 10
        + "IV. DISPOSITIVO: Ordem concedida."
    )
    result = _extract_ementa_payload(ementa, max_chars=500)
    assert "DISPOSITIVO" in result


def test_extract_ementa_result_respects_max_chars() -> None:
    """O resultado nao deve ultrapassar significativamente o limite max_chars."""
    ementa = (
        "I. CASO EM EXAME: texto inicial " * 5
        + "III. RAZOES DE DECIDIR: " + "razao " * 200
        + "IV. DISPOSITIVO: Concedida."
    )
    result = _extract_ementa_payload(ementa, max_chars=300)
    # Pode ser ligeiramente maior devido as marcacoes [...], mas nao excessivo
    assert len(result) <= 600


# ===========================================================================
# Testes — Súmulas Vinculantes STF no prompt (v7)
# ===========================================================================


def test_build_prompt_sv_aparece_antes_dos_acordaos() -> None:
    """Bloco de SV deve preceder os blocos de acórdão no prompt."""
    sv = _make_sv(numero=11, enunciado="Uso de algemas restrito a casos excepcionais.")
    acordao = _make_acordao(numero="HC 100001")
    prompt = _build_prompt("pergunta", [acordao], [], [sv])
    pos_sv = prompt.index("[Súmula Vinculante STF 11]")
    pos_ac = prompt.index("[Acórdão STF 1]")
    assert pos_sv < pos_ac, "SV deve preceder acórdãos no contexto do prompt"


def test_build_prompt_sv_tem_efeito_vinculante_constitucional() -> None:
    """Linha Efeito: da SV deve mencionar 'vinculante constitucional' e art. 103-A."""
    sv = _make_sv(numero=25, enunciado="E ilicita a prisao civil de depositario infiel.")
    prompt = _build_prompt("pergunta", [], [], [sv])
    assert "vinculante constitucional" in prompt
    assert "103-A" in prompt


def test_build_prompt_sv_regra_citacao_presente() -> None:
    """Regra de citação de SV como 'SV N/STF' deve estar no prompt."""
    prompt = _build_prompt("pergunta", [], [], [_make_sv()])
    assert "SV N/STF" in prompt or "SV 11/STF" in prompt or "SV" in prompt


def test_build_prompt_sv_nota_fontes_menciona_hierarquia_maxima() -> None:
    """Nota sobre as fontes deve incluir referência à hierarquia máxima das SVs."""
    prompt = _build_prompt("pergunta", [], [], [_make_sv()])
    assert "Nota sobre as fontes:" in prompt
    assert "art. 103-A" in prompt


def test_build_prompt_sv_fontes_desc_inclui_sumulas_vinculantes() -> None:
    """Com SVs presentes, fontes_desc deve mencionar 'súmulas vinculantes'."""
    sv = _make_sv()
    prompt = _build_prompt("pergunta", [], [], [sv])
    assert "súmulas vinculantes" in prompt.lower()


def test_build_prompt_sem_sv_nao_insere_bloco_numerado_no_contexto() -> None:
    """Sem SVs, o contexto não deve conter bloco com número real (ex: [Súmula Vinculante STF 11])."""
    import re
    prompt = _build_prompt("pergunta", [_make_acordao()], [], [])
    # As instruções de citação mencionam "[Súmula Vinculante STF N]" (literal N),
    # mas não deve existir bloco com número real como [Súmula Vinculante STF 11].
    assert not re.search(r'\[Súmula Vinculante STF \d+\]', prompt)
