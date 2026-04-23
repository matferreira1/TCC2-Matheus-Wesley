"""Expansão de consulta com sinônimos jurídicos para FTS5.

Estratégia:
  - Cada entrada de ``_SYNONYMS`` mapeia uma sequência de tokens normalizados
    (sem acentos, minúsculas) para um conjunto de termos extras.
  - ``expand_query`` recebe os tokens já normalizados pelo search_service e
    devolve tokens adicionais a incluir na busca OR, ampliando o recall sem
    custo de latência.

Limitação conhecida: tokens de 2 caracteres (ex.: "hc", "re", "ms") são
filtrados pelo tokenizador do search_service (``len(t) > 2``) antes de chegar
aqui, portanto não funcionam como chave de expansão; apenas como sinônimos
adicionados à query expandida.
"""

from __future__ import annotations

import unicodedata

# ---------------------------------------------------------------------------
# Dicionário de sinônimos jurídicos
# Chaves: tokens normalizados (sem acento, minúsculas) separados por espaço.
# Valores: tokens extras a adicionar na busca OR.
# ---------------------------------------------------------------------------
_SYNONYMS: dict[str, frozenset[str]] = {
    # ── Remédios constitucionais e siglas ────────────────────────────────────
    "habeas corpus": frozenset({"hc"}),
    "habeas": frozenset({"hc", "corpus"}),
    "mandado seguranca": frozenset({"ms"}),
    "recurso extraordinario": frozenset({"re", "are"}),
    "recurso especial": frozenset({"resp"}),
    "acao direta inconstitucionalidade": frozenset({"adi"}),
    "adi": frozenset({"inconstitucionalidade", "direta"}),
    "agravo regimental": frozenset({"agr", "agint"}),
    "agr": frozenset({"agravo", "regimental", "agint"}),
    "agravo interno": frozenset({"agint", "agr"}),
    "agint": frozenset({"agravo", "interno", "agr"}),
    "agravo instrumento": frozenset({"agravo"}),
    "are": frozenset({"recurso", "extraordinario", "re"}),
    "resp": frozenset({"recurso", "especial"}),
    # ── Direito penal / processual penal ─────────────────────────────────────
    "prisao": frozenset({"custodia", "detencao", "encarceramento"}),
    "custodia": frozenset({"prisao", "detencao"}),
    "detencao": frozenset({"prisao", "custodia"}),
    "prisao preventiva": frozenset({"custodia", "cautelar", "segregacao"}),
    "flagrante": frozenset({"prisao", "flagrancia"}),
    "liberdade provisoria": frozenset({"soltura", "livramento"}),
    "pena": frozenset({"sancao", "penalidade", "condenacao"}),
    "sancao": frozenset({"pena", "penalidade"}),
    "condenacao": frozenset({"pena", "sentenca"}),
    "absolvicao": frozenset({"inocencia", "impronuncia"}),
    "trafico": frozenset({"drogas", "entorpecentes"}),
    "trafico drogas": frozenset({"entorpecentes", "narcotrafico"}),
    "peculato": frozenset({"corrupcao", "desvio"}),
    "prescricao": frozenset({"extincao", "decadencia"}),
    "reincidencia": frozenset({"antecedentes", "criminalidade"}),
    # ── Direito civil / obrigações ────────────────────────────────────────────
    "indenizacao": frozenset({"reparacao", "ressarcimento", "compensacao"}),
    "reparacao": frozenset({"indenizacao", "ressarcimento"}),
    "ressarcimento": frozenset({"indenizacao", "reparacao"}),
    "dano": frozenset({"indenizacao", "prejuizo", "lesao"}),
    "dano moral": frozenset({"indenizacao", "extrapatrimonial", "reparacao"}),
    "dano material": frozenset({"indenizacao", "ressarcimento", "prejuizo"}),
    "contrato": frozenset({"negocio", "ajuste", "obrigacao"}),
    "rescisao": frozenset({"resolucao", "distrato", "cancelamento"}),
    "inadimplemento": frozenset({"mora", "inadimplencia", "descumprimento"}),
    "inadimplencia": frozenset({"mora", "inadimplemento", "descumprimento"}),
    "responsabilidade civil": frozenset({"indenizacao", "reparacao", "dano"}),
    # ── Direito do consumidor ─────────────────────────────────────────────────
    "consumidor": frozenset({"cdc", "fornecedor"}),
    "fornecedor": frozenset({"consumidor", "cdc"}),
    "cdc": frozenset({"consumidor", "fornecedor"}),
    "plano saude": frozenset({"convenio", "operadora", "cobertura", "assistencia"}),
    "convenio medico": frozenset({"plano", "saude", "cobertura"}),
    # ── Direito administrativo / constitucional ──────────────────────────────
    "servidor publico": frozenset({"funcionario", "agente", "cargo", "efetivo"}),
    "funcionario publico": frozenset({"servidor", "agente", "cargo"}),
    "estabilidade": frozenset({"efetividade", "cargo", "efetivo"}),
    "demissao": frozenset({"dispensa", "rescisao", "desligamento"}),
    "dispensa": frozenset({"demissao", "rescisao"}),
    "concurso publico": frozenset({"cargo", "aprovacao", "nomeacao"}),
    "licitacao": frozenset({"contrato", "administrativo", "pregao"}),
    "improbidade": frozenset({"corrupcao", "desvio", "ato"}),
    "corrupcao": frozenset({"improbidade", "peculato", "desvio"}),
    # ── Direito tributário ────────────────────────────────────────────────────
    "tributo": frozenset({"imposto", "taxa", "contribuicao"}),
    "imposto": frozenset({"tributo", "exacao", "tributacao"}),
    "icms": frozenset({"imposto", "circulacao", "mercadorias"}),
    "iss": frozenset({"imposto", "servicos"}),
    "iptu": frozenset({"imposto", "propriedade", "urbano"}),
    # ── Direito trabalhista ───────────────────────────────────────────────────
    "salario": frozenset({"remuneracao", "vencimento", "proventos"}),
    "remuneracao": frozenset({"salario", "vencimento", "proventos"}),
    "horas extras": frozenset({"adicional", "sobrejornada", "jornada"}),
    "aviso previo": frozenset({"rescisao", "indenizacao"}),
    "assedio moral": frozenset({"dano", "moral", "violencia", "humilhacao"}),
    # ── Processo / garantias fundamentais ────────────────────────────────────
    "competencia": frozenset({"jurisdicao", "atribuicao"}),
    "jurisdicao": frozenset({"competencia", "atribuicao"}),
    "legitimidade": frozenset({"parte", "legitimacao", "capacidade"}),
    "coisa julgada": frozenset({"transito", "julgado", "imutabilidade"}),
    "transito julgado": frozenset({"coisa", "julgada", "imutabilidade"}),
    "tutela antecipada": frozenset({"liminar", "cautelar", "urgencia"}),
    "liminar": frozenset({"tutela", "urgencia", "cautelar"}),
    "cautelar": frozenset({"liminar", "tutela", "urgencia"}),
    "devido processo legal": frozenset({"contraditorio", "ampla", "defesa"}),
    "contraditorio": frozenset({"ampla", "defesa", "devido", "processo"}),
    "sumula": frozenset({"enunciado", "jurisprudencia", "orientacao"}),
    "jurisprudencia": frozenset({"sumula", "precedente", "julgado"}),
    "precedente": frozenset({"jurisprudencia", "sumula", "orientacao"}),
    # ── Súmulas Vinculantes STF ───────────────────────────────────────────────
    "sumula vinculante": frozenset({"sv", "vinculante", "stf"}),
    "sumulas vinculantes": frozenset({"sv", "vinculante", "stf"}),
    "vinculante": frozenset({"sumula", "obrigatorio", "stf"}),
    "sv": frozenset({"sumula", "vinculante", "stf"}),
}


def _strip_accents(text: str) -> str:
    """Remove acentos diacríticos (á→a, ç→c, ã→a, etc.)."""
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def expand_query(tokens: list[str]) -> list[str]:
    """Retorna tokens extras para expansão da consulta FTS5 via sinônimos jurídicos.

    Recebe os tokens já normalizados pelo search_service (minúsculas, sem
    pontuação especial, sem stopwords) e devolve uma lista de tokens adicionais
    a incluir na busca OR — sem duplicar nenhum token já presente.

    Exemplo::

        >>> expand_query(["prisao", "preventiva"])
        ['custodia', 'cautelar', 'segregacao', 'detencao', 'encarceramento']

        >>> expand_query(["habeas", "corpus"])
        ['hc']
    """
    if not tokens:
        return []

    # Normaliza para lookup (sem acentos)
    norm = [_strip_accents(t) for t in tokens]
    norm_set = set(norm)
    extra: set[str] = set()

    for key, synonyms in _SYNONYMS.items():
        key_tokens = key.split()
        if all(t in norm_set for t in key_tokens):
            extra.update(synonyms)

    # Remove tokens que já estão na query original
    return sorted(extra - norm_set)
