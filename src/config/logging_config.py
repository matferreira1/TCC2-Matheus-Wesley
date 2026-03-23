"""Configuração central de logging do IAJuris."""

from __future__ import annotations

import logging
import re
import sys

# Padrões de dados sensíveis que nunca devem aparecer nos logs
_REDACT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r'gsk_[A-Za-z0-9]{20,}'), '[GROQ_KEY_REDACTED]'),
]


class _SecretFilter(logging.Filter):
    """Remove segredos conhecidos das mensagens de log antes de emiti-las."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = str(record.msg)
        for pattern, replacement in _REDACT_PATTERNS:
            msg = pattern.sub(replacement, msg)
        record.msg = msg
        return True


def setup_logging(debug: bool = False) -> None:
    """Configura formato e nível de logging da aplicação."""
    level = logging.DEBUG if debug else logging.INFO

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.addFilter(_SecretFilter())

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    # silencia libs externas verbosas
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
