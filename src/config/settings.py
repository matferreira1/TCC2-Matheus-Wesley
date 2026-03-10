"""
Configurações globais da aplicação via Pydantic-Settings.

Lê variáveis do arquivo .env ou do ambiente do sistema operacional.
Uso em qualquer módulo:
    from src.config.settings import settings
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centraliza todas as configurações do IAJuris."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # Aplicação
    # ------------------------------------------------------------------
    app_name: str = "IAJuris"
    debug: bool = False

    # ------------------------------------------------------------------
    # Banco de dados SQLite
    # ------------------------------------------------------------------
    database_url: str = "data/db/iajuris.db"
    # Nome da tabela FTS5 e da tabela de metadados
    db_table_meta: str = "jurisprudencia"
    db_table_fts: str = "jurisprudencia_fts"

    # ------------------------------------------------------------------
    # Provedor LLM ativo: "groq" ou "ollama"
    # ------------------------------------------------------------------
    llm_provider: str = "ollama"  # sobrescrito pelo .env: LLM_PROVIDER=groq

    # ------------------------------------------------------------------
    # Ollama (LLM local — fallback)
    # ------------------------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout_seconds: int = 600  # CPU sem GPU pode levar vários minutos

    # ------------------------------------------------------------------
    # Groq (API externa)
    # ------------------------------------------------------------------
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_timeout_seconds: int = 60

    # ------------------------------------------------------------------
    # RAG
    # ------------------------------------------------------------------
    rag_top_k: int = 5              # nº de acórdãos recuperados pelo FTS5
    rag_top_k_teses: int = 3        # nº de teses STJ recuperadas pelo FTS5
    rag_max_tokens: int = 2048      # limite de tokens na geração
    rag_max_ementa_chars: int = 1500  # limite por ementa na montagem do prompt

    # ------------------------------------------------------------------
    # Helpers (não lidos do .env)
    # ------------------------------------------------------------------
    @property
    def db_path(self) -> Path:
        """Retorna o caminho absoluto do arquivo SQLite como objeto Path."""
        return Path(self.database_url)

    def ensure_db_dir(self) -> None:
        """Cria o diretório do banco de dados caso não exista."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Retorna a instância singleton de Settings (cacheada).

    Usar diretamente ``settings`` na maioria dos casos.
    Usar ``get_settings()`` como Depends() em testes para permitir override.
    """
    return Settings()


# Instância singleton — importar de outros módulos:
#   from src.config.settings import settings
settings: Settings = get_settings()
