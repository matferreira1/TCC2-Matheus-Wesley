"""
download_models.py
------------------
Executado APENAS durante o Docker build (RUN python download_models.py).

Faz o download e cache dos modelos HuggingFace para que o container
não precise baixá-los em runtime, eliminando latência no cold start.

Paths de cache:
  - SENTENCE_TRANSFORMERS_HOME=/app/models_cache
  - HF_HOME=/app/models_cache
  - TRANSFORMERS_CACHE=/app/models_cache

Esses valores são idênticos aos definidos no Dockerfile e respeitam
o comportamento padrão do sentence-transformers, que resolve o diretório
de cache na seguinte ordem:
  1. argumento cache_folder (não usado nos serviços)
  2. variável SENTENCE_TRANSFORMERS_HOME
  3. variável HF_HOME / torch.hub

Como semantic_service.py instancia SentenceTransformer(MODEL_NAME) e
rerank_service.py instancia CrossEncoder(MODEL_NAME) — ambos SEM
cache_folder explícito — a variável de ambiente é o único mecanismo de
controle do caminho e DEVE ser idêntica no build e em runtime.
"""

from __future__ import annotations

import os
import sys

# Deve bater com SENTENCE_TRANSFORMERS_HOME do Dockerfile
CACHE_DIR = os.environ.get("SENTENCE_TRANSFORMERS_HOME", "/app/models_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Nomes exatos copiados de semantic_service.py e rerank_service.py
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def download_embedding_model() -> None:
    """
    Instancia SentenceTransformer sem cache_folder — o modelo é salvo em
    SENTENCE_TRANSFORMERS_HOME, exatamente como semantic_service._get_model() faz.
    """
    from sentence_transformers import SentenceTransformer

    print(f"  → SentenceTransformer: {EMBEDDING_MODEL}")
    SentenceTransformer(EMBEDDING_MODEL)
    print(f"  ✓ Salvo em {CACHE_DIR}")


def download_cross_encoder_model() -> None:
    """
    Instancia CrossEncoder sem cache_folder — mesmo comportamento de
    rerank_service._get_model().
    """
    from sentence_transformers import CrossEncoder

    print(f"  → CrossEncoder: {CROSS_ENCODER_MODEL}")
    CrossEncoder(CROSS_ENCODER_MODEL, max_length=512)
    print(f"  ✓ Salvo em {CACHE_DIR}")


def main() -> None:
    print("=" * 60)
    print("IAJuris — pré-download de modelos HuggingFace")
    print(f"SENTENCE_TRANSFORMERS_HOME = {CACHE_DIR}")
    print("=" * 60)

    steps = [
        ("Embeddings semânticos (busca híbrida)", download_embedding_model),
        ("Cross-encoder reranking", download_cross_encoder_model),
    ]

    for description, fn in steps:
        print(f"\n[{description}]")
        try:
            fn()
        except Exception as exc:
            print(f"  ✗ ERRO: {exc}", file=sys.stderr)
            sys.exit(1)

    print("\n" + "=" * 60)
    print("✓ Todos os modelos baixados com sucesso.")
    print("=" * 60)


if __name__ == "__main__":
    main()