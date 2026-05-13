"""
Testes para o módulo de expansão de consulta com sinônimos jurídicos.
"""

from __future__ import annotations

import pytest

from src.services.query_expansion import expand_query, _strip_accents


# ===========================================================================
# _strip_accents
# ===========================================================================


def test_strip_accents_removes_cedilla() -> None:
    assert _strip_accents("indenização") == "indenizacao"


def test_strip_accents_removes_tilde() -> None:
    assert _strip_accents("prisão") == "prisao"


def test_strip_accents_keeps_plain_text() -> None:
    assert _strip_accents("habeas") == "habeas"


# ===========================================================================
# expand_query — entrada vazia / sem correspondência
# ===========================================================================


def test_expand_empty_list_returns_empty() -> None:
    assert expand_query([]) == []


def test_expand_unknown_term_returns_empty() -> None:
    assert expand_query(["xyzabcqwerty"]) == []


def test_expand_stopwords_only_returns_empty() -> None:
    # Stopwords já são filtradas antes de chegar aqui, mas caso passem
    assert expand_query(["de", "o", "a"]) == []


# ===========================================================================
# expand_query — chaves de token único (≥ 3 chars)
# ===========================================================================


def test_expand_single_token_adi() -> None:
    result = expand_query(["adi"])
    assert "inconstitucionalidade" in result
    assert "direta" in result


def test_expand_single_token_are() -> None:
    result = expand_query(["are"])
    assert "recurso" in result
    assert "extraordinario" in result


def test_expand_single_token_resp() -> None:
    result = expand_query(["resp"])
    assert "recurso" in result
    assert "especial" in result


def test_expand_single_token_iss() -> None:
    result = expand_query(["iss"])
    assert "imposto" in result
    assert "servicos" in result


def test_expand_single_token_prisao() -> None:
    result = expand_query(["prisao"])
    assert "custodia" in result
    assert "detencao" in result


def test_expand_single_token_indenizacao() -> None:
    result = expand_query(["indenizacao"])
    assert "reparacao" in result
    assert "ressarcimento" in result


def test_expand_single_token_liminar() -> None:
    result = expand_query(["liminar"])
    assert "cautelar" in result
    assert "urgencia" in result


# ===========================================================================
# expand_query — chaves multi-token (todos os tokens devem estar presentes)
# ===========================================================================


def test_expand_habeas_corpus_adds_hc() -> None:
    result = expand_query(["habeas", "corpus"])
    assert "hc" in result


def test_expand_prisao_preventiva() -> None:
    result = expand_query(["prisao", "preventiva"])
    assert "cautelar" in result
    assert "custodia" in result


def test_expand_dano_moral() -> None:
    result = expand_query(["dano", "moral"])
    assert "indenizacao" in result
    assert "reparacao" in result


def test_expand_plano_saude() -> None:
    result = expand_query(["plano", "saude"])
    assert "cobertura" in result
    assert "operadora" in result


def test_expand_servidor_publico() -> None:
    result = expand_query(["servidor", "publico"])
    assert "funcionario" in result
    assert "cargo" in result


def test_expand_recurso_extraordinario() -> None:
    result = expand_query(["recurso", "extraordinario"])
    assert "re" in result


def test_expand_recurso_especial() -> None:
    result = expand_query(["recurso", "especial"])
    assert "resp" in result


# ===========================================================================
# expand_query — correspondência parcial NÃO deve expandir
# ===========================================================================


def test_expand_partial_match_habeas_only() -> None:
    # "habeas corpus" como chave requer ambos os tokens; "habeas" sozinho
    # tem sua própria chave, mas não deve adicionar os sinônimos de "habeas corpus"
    result = expand_query(["habeas"])
    # deve adicionar hc e corpus (chave "habeas"), mas NÃO via "habeas corpus"
    assert "hc" in result
    assert "corpus" in result


def test_expand_partial_match_plano_only() -> None:
    # sem "saude" não deve expandir "plano saude"
    result = expand_query(["plano"])
    assert "operadora" not in result
    assert "cobertura" not in result


def test_expand_partial_match_dano_only() -> None:
    # "dano" sozinho expande via chave "dano", mas não via "dano moral"
    result = expand_query(["dano"])
    assert "indenizacao" in result   # de "dano"
    assert "extrapatrimonial" not in result  # só em "dano moral"


# ===========================================================================
# expand_query — entrada com acentos (normalização automática)
# ===========================================================================


def test_expand_accented_prisao() -> None:
    result = expand_query(["prisão"])  # acento → normalizado internamente
    assert "custodia" in result


def test_expand_accented_indenizacao() -> None:
    result = expand_query(["indenização"])
    assert "reparacao" in result


def test_expand_accented_habeas_corpus() -> None:
    result = expand_query(["habeas", "córpus"])
    # "córpus" normaliza para "corpus" → deve casar com a chave "habeas corpus"
    assert "hc" in result


# ===========================================================================
# expand_query — sem duplicatas
# ===========================================================================


def test_no_duplicates_in_result() -> None:
    result = expand_query(["prisao", "preventiva"])
    assert len(result) == len(set(result))


def test_original_tokens_not_in_result() -> None:
    tokens = ["prisao", "preventiva"]
    result = expand_query(tokens)
    for t in tokens:
        assert t not in result


def test_already_present_synonym_not_duplicated() -> None:
    # "cautelar" já está na query → não deve aparecer nos extras
    result = expand_query(["prisao", "preventiva", "cautelar"])
    assert "cautelar" not in result


# ===========================================================================
# expand_query — múltiplas chaves disparadas simultaneamente
# ===========================================================================


def test_multiple_keys_triggered() -> None:
    # "dano moral" + "indenizacao" — dispara "dano moral" e "indenizacao"
    result = expand_query(["dano", "moral", "indenizacao"])
    # De "dano moral": extrapatrimonial, reparacao
    # De "indenizacao": reparacao, ressarcimento, compensacao
    # De "dano": prejuizo, lesao, indenizacao (já presente)
    assert "extrapatrimonial" in result
    assert "ressarcimento" in result


# ===========================================================================
# expand_query — Direito previdenciário
# ===========================================================================


def test_expand_inss_adds_previdencia_e_beneficio() -> None:
    result = expand_query(["inss"])
    assert "previdencia" in result
    assert "beneficio" in result
    assert "segurado" in result


def test_expand_aposentadoria_adds_inss_e_beneficio() -> None:
    result = expand_query(["aposentadoria"])
    assert "inss" in result
    assert "beneficio" in result
    assert "previdencia" in result


def test_expand_aposentadoria_invalidez() -> None:
    result = expand_query(["aposentadoria", "invalidez"])
    assert "incapacidade" in result
    assert "permanente" in result
    assert "inss" in result


def test_expand_pensao_morte() -> None:
    result = expand_query(["pensao", "morte"])
    assert "dependente" in result
    assert "obito" in result
    assert "inss" in result


def test_expand_auxilio_doenca() -> None:
    result = expand_query(["auxilio", "doenca"])
    assert "incapacidade" in result
    assert "temporaria" in result
    assert "beneficio" in result


def test_expand_bpc_adds_loas_e_assistencia() -> None:
    result = expand_query(["bpc"])
    assert "loas" in result
    assert "assistencia" in result
    assert "beneficio" in result


def test_expand_loas_adds_bpc() -> None:
    result = expand_query(["loas"])
    assert "bpc" in result
    assert "beneficio" in result


def test_expand_carencia_adds_contribuicao() -> None:
    result = expand_query(["carencia"])
    assert "contribuicao" in result
    assert "inss" in result


def test_expand_segurado_adds_inss_e_beneficiario() -> None:
    result = expand_query(["segurado"])
    assert "inss" in result
    assert "beneficiario" in result


def test_expand_invalidez_adds_incapacidade() -> None:
    result = expand_query(["invalidez"])
    assert "incapacidade" in result
    assert "aposentadoria" in result


def test_expand_tempo_contribuicao() -> None:
    result = expand_query(["tempo", "contribuicao"])
    assert "aposentadoria" in result
    assert "carencia" in result
    assert "inss" in result


def test_expand_previdencia_social() -> None:
    result = expand_query(["previdencia", "social"])
    assert "inss" in result
    assert "aposentadoria" in result


def test_expand_seguridade_social() -> None:
    result = expand_query(["seguridade", "social"])
    assert "previdencia" in result
    assert "inss" in result


# ===========================================================================
# expand_query — Direito bancário / fraude financeira
# ===========================================================================


def test_expand_banco_adds_financeira_e_bancario() -> None:
    result = expand_query(["banco"])
    assert "financeira" in result
    assert "bancario" in result


def test_expand_fraude_adds_estelionato_e_golpe() -> None:
    result = expand_query(["fraude"])
    assert "estelionato" in result
    assert "golpe" in result


def test_expand_debito_adds_saque_e_movimentacao() -> None:
    result = expand_query(["debito"])
    assert "saque" in result
    assert "movimentacao" in result


def test_expand_senha_adds_autenticacao_e_credencial() -> None:
    result = expand_query(["senha"])
    assert "autenticacao" in result
    assert "credencial" in result


def test_expand_conta_bancaria_adds_deposito_e_bancario() -> None:
    result = expand_query(["conta", "bancaria"])
    assert "deposito" in result
    assert "bancario" in result


def test_expand_fraude_bancaria_adds_estelionato_e_clonagem() -> None:
    result = expand_query(["fraude", "bancaria"])
    assert "estelionato" in result
    assert "clonagem" in result


def test_expand_pix_adds_transferencia_e_bancaria() -> None:
    result = expand_query(["pix"])
    assert "transferencia" in result
    assert "bancaria" in result


def test_expand_estelionato_adds_fraude_e_golpe() -> None:
    result = expand_query(["estelionato"])
    assert "fraude" in result
    assert "golpe" in result


def test_expand_clonagem_adds_fraude_e_cartao() -> None:
    result = expand_query(["clonagem"])
    assert "fraude" in result
    assert "cartao" in result


def test_expand_responsabilidade_banco_adds_risco() -> None:
    result = expand_query(["responsabilidade", "banco"])
    assert "risco" in result
    assert "atividade" in result


def test_expand_fortuito_interno_adds_banco() -> None:
    result = expand_query(["fortuito", "interno"])
    assert "banco" in result
    assert "responsabilidade" in result
