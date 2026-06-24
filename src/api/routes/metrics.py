"""
Endpoint da dashboard de métricas.

Rota:
  GET /api/v1/metrics — agregado ao vivo (operação + qualidade reference-free
                        por consulta) + snapshot da última avaliação offline.
"""

import json
import logging
from pathlib import Path

from fastapi import APIRouter

from src.services import metrics_service

logger = logging.getLogger(__name__)
router = APIRouter()

_BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
_OFFLINE_REPORT = _BASE_DIR / "eval" / "results" / "consolidated_report.json"


def _load_offline() -> dict | None:
    """Lê o relatório consolidado da avaliação offline (Recall@k, judge, etc.)."""
    if not _OFFLINE_REPORT.exists():
        return None
    try:
        data = json.loads(_OFFLINE_REPORT.read_text(encoding="utf-8"))
        return {
            "available": True,
            "generated_at_mtime": _OFFLINE_REPORT.stat().st_mtime,
            **data,
        }
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Falha ao ler relatório offline: %s", exc)
        return None


@router.get(
    "/metrics",
    summary="Métricas do sistema",
    description=(
        "Métricas ao vivo (operação e qualidade reference-free), agregadas por "
        "consulta, mais o snapshot da última avaliação offline (Recall@k, MRR, "
        "nDCG@k, P@k, LLM-judge)."
    ),
)
async def get_metrics() -> dict:
    return {
        "live": metrics_service.snapshot(),
        "offline": _load_offline() or {"available": False},
    }
