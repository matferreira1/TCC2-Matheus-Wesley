"""Cliente HTTP assíncrono para a API do Ollama."""

from __future__ import annotations

import json
import logging
import time

import httpx

from src.config.settings import settings

logger = logging.getLogger(__name__)


async def generate(prompt: str) -> str:
    """
    Envia ``prompt`` ao Ollama e retorna o texto gerado.

    Usa stream=True internamente para não travar o event loop
    enquanto o modelo gera tokens (inferência pode levar minutos).

    Raises:
        httpx.HTTPStatusError: resposta HTTP com erro.
        httpx.TimeoutException: Ollama não respondeu no prazo.
    """
    url = f"{settings.ollama_base_url}/api/generate"
    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": True,
    }
    # connect_timeout curto; read/write sem limite — inferência em CPU pura pode
    # levar muitos minutos antes do primeiro byte chegar (headers só são enviados
    # quando o Ollama começa a gerar, não imediatamente ao receber a requisição)
    timeout = httpx.Timeout(None, connect=10.0)
    parts: list[str] = []

    logger.info("Enviando prompt | modelo=%s | chars=%d", settings.ollama_model, len(prompt))
    inicio = time.perf_counter()

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                parts.append(chunk.get("response", ""))
                if chunk.get("done"):
                    break

    resposta = "".join(parts)
    elapsed = time.perf_counter() - inicio
    logger.info("Ollama respondeu em %.1fs | chars=%d", elapsed, len(resposta))
    return resposta


async def health_check() -> bool:
    """Retorna True se o Ollama estiver acessível."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(settings.ollama_base_url)
            return resp.status_code == 200
    except Exception:
        return False
