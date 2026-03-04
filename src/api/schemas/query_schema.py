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


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class SourceDocument(BaseModel):
    """Metadados de um fragmento de jurisprudência utilizado como fonte."""

    tribunal: str
    numero_processo: str
    ementa: str


class QueryResponse(BaseModel):
    """Resposta estruturada do pipeline RAG."""

    answer: str = Field(..., description="Resposta gerada pelo LLM.")
    sources: list[SourceDocument] = Field(
        default_factory=list,
        description="Fragmentos de jurisprudência utilizados como contexto.",
    )
