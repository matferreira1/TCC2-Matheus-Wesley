"""Cliente assíncrono para a API da Groq."""

from __future__ import annotations

import asyncio
import logging
import re
import time

from groq import AsyncGroq, RateLimitError

from src.config.settings import settings

logger = logging.getLogger(__name__)

_client: AsyncGroq | None = None


def _get_client() -> AsyncGroq:
    """Retorna (ou cria) o cliente Groq singleton."""
    global _client
    if _client is None:
        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY não configurada. "
                "Defina a variável no arquivo .env ou no ambiente."
            )
        if not re.match(r'^gsk_[A-Za-z0-9]+$', settings.groq_api_key):
            logger.warning(
                "GROQ_API_KEY tem formato inesperado (esperado: gsk_<alfanumérico>)."
            )
        _client = AsyncGroq(api_key=settings.groq_api_key)
    return _client


async def generate(prompt: str) -> str:
    """
    Envia ``prompt`` à API da Groq e retorna o texto gerado.

    Raises:
        ValueError: chave de API ausente.
        groq.APIStatusError: erro retornado pela API (4xx/5xx).
        groq.APITimeoutError: timeout na requisição.
    """
    client = _get_client()
    logger.info(
        "Enviando prompt | modelo=%s | chars=%d",
        settings.groq_model,
        len(prompt),
    )
    inicio = time.perf_counter()

    for attempt in range(3):
        try:
            completion = await client.chat.completions.create(
                model=settings.groq_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=settings.rag_max_tokens,
                temperature=0.2,
                timeout=settings.groq_timeout_seconds,
            )
            break
        except RateLimitError as exc:
            # Extrai o tempo de espera sugerido pela API (ex: "43m40s")
            m = re.search(r"try again in (\d+)m(\d+(?:\.\d+)?)s", str(exc))
            wait = (int(m.group(1)) * 60 + float(m.group(2)) + 5) if m else 60
            if attempt < 2:
                logger.warning("Rate limit Groq — aguardando %.0fs antes de tentar novamente...", wait)
                await asyncio.sleep(wait)
            else:
                raise

    resposta = completion.choices[0].message.content or ""
    elapsed = time.perf_counter() - inicio
    logger.info("Groq respondeu em %.1fs | chars=%d", elapsed, len(resposta))
    return resposta


async def health_check() -> bool:
    """Retorna True se a chave Groq estiver configurada e a API acessível."""
    try:
        client = _get_client()
        # Listagem de modelos é um endpoint leve, não gera tokens
        await client.models.list()
        return True
    except Exception:
        return False
