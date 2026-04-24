"""
Schemas Pydantic para o endpoint de consulta RAG.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Payload enviado pelo cliente na consulta jurídica."""

    question: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        examples=["Qual o entendimento do STF sobre sigilo bancário?"],
        description="Pergunta jurídica em linguagem natural.",
    )
    date_from: str | None = Field(
        default=None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        examples=["2020-01-01"],
        description=(
            "Data de início do filtro temporal no formato YYYY-MM-DD. "
            "Aplica-se apenas a acórdãos STF — teses e súmulas são sempre incluídas."
        ),
    )
    date_to: str | None = Field(
        default=None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        examples=["2024-12-31"],
        description=(
            "Data de fim do filtro temporal no formato YYYY-MM-DD. "
            "Aplica-se apenas a acórdãos STF — teses e súmulas são sempre incluídas."
        ),
    )


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class SourceDocument(BaseModel):
    """Metadados de um fragmento de jurisprudência utilizado como fonte."""

    tribunal: str
    numero_processo: str
    ementa: str
    tipo: str = Field(default="acordao", description="'acordao' (STF) ou 'tese_stj' (STJ Em Teses)")
    orgao_julgador: str | None = Field(
        default=None,
        description="Órgão julgador do STF (ex: Primeira Turma, Pleno). Nulo para teses/súmulas.",
    )
    repercussao_geral: bool | None = Field(
        default=None,
        description="Indica se o acórdão STF foi decidido com repercussão geral. Nulo para teses/súmulas.",
    )
    data_julgamento: str | None = Field(
        default=None,
        description="Data de julgamento do acórdão STF. Nulo para teses e súmulas.",
    )


class QueryResponse(BaseModel):
    """Resposta estruturada do pipeline RAG."""

    answer: str = Field(..., description="Resposta gerada pelo LLM.")
    sources: list[SourceDocument] = Field(
        default_factory=list,
        description="Fragmentos de jurisprudência utilizados como contexto.",
    )
