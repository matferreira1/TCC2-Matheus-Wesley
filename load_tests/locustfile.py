"""
Teste de carga IAJuris — Locust
================================

Pré-requisito:
  pip install locust

Como executar
-------------

# 1. Interface web (abre http://localhost:8089):
locust -f load_tests/locustfile.py --host http://127.0.0.1:8000

# 2. Modo headless — cenário padrão (1 HealthUser + 1 QueryUser, 2 min):
locust -f load_tests/locustfile.py --host http://127.0.0.1:8000 \\
       --headless -u 2 -r 1 -t 120s --html load_tests/report.html

# 3. Teste de stress (REQUER rate limit elevado — veja abaixo):
RATE_LIMIT_PER_MINUTE=200 uvicorn main:app --host 127.0.0.1 --port 8000
locust -f load_tests/locustfile.py --host http://127.0.0.1:8000 \\
       --headless -u 10 -r 2 -t 60s --html load_tests/report_stress.html \\
       --tags stress

Rate limit — ATENÇÃO
---------------------
O endpoint /api/v1/query tem limite padrão de 10 req/min por IP.
Como todos os usuários Locust compartilham o IP 127.0.0.1, o limite é
global para o teste inteiro, não por usuário virtual.

Modo padrão (sem --tags stress):
  - QueryUser usa wait_time(8, 15) → ~4-7 req/min → seguro com até 1 usuário
  - Para 2+ QueryUsers → espere 429s ocasionais (comportamento correto)

Modo stress (--tags stress):
  - Reinicie o servidor com RATE_LIMIT_PER_MINUTE=200 no .env antes de testar
  - StressQueryUser usa wait_time(1, 3) → alta taxa de requisições

Métricas coletadas pelo Locust
-------------------------------
  - Latência: mediana (p50), p95, p99 (ms)
  - Throughput: requisições/s
  - Taxa de erro: % de falhas (inclui 429, 500, 504)
  - Distribuição de status HTTP
"""

import random

from locust import HttpUser, between, tag, task

# ---------------------------------------------------------------------------
# Corpus de perguntas jurídicas realistas (simula uso real do sistema)
# ---------------------------------------------------------------------------

_PERGUNTAS = [
    "Qual o entendimento do STF sobre sigilo bancário?",
    "Como o STJ decide casos de responsabilidade civil do Estado?",
    "Quais são os requisitos para concessão de habeas corpus?",
    "O que diz a jurisprudência sobre prescrição tributária?",
    "Qual o posicionamento do STF sobre prisão em segunda instância?",
    "Como são tratados os casos de dano moral no STJ?",
    "Quais são os direitos do consumidor segundo o STJ?",
    "O que diz o STF sobre interceptação telefônica?",
    "Como funciona o princípio da insignificância no STF?",
    "Quais são as súmulas do STJ sobre cobrança de juros bancários?",
    "O que diz a jurisprudência do STF sobre ação popular?",
    "Como o STF trata o direito à saúde como direito fundamental?",
    "Qual a posição do STJ sobre impenhorabilidade de bem de família?",
    "Como é tratada a prova ilícita no processo penal pelo STF?",
    "O que diz o STF sobre liberdade de expressão e discurso de ódio?",
    "Quais são os requisitos para decretação de prisão preventiva?",
    "Como o STJ decide sobre revisão de cláusulas abusivas em contratos?",
    "O que diz a jurisprudência sobre improbidade administrativa?",
    "Qual o entendimento do STF sobre imunidade tributária de templos?",
    "Como o STJ trata o abandono afetivo como dano moral?",
]


# ---------------------------------------------------------------------------
# HealthUser — monitora o endpoint /health continuamente
# Simula ferramenta de monitoramento externo (uptime checker).
# Não está sujeito ao rate limit do /query.
# ---------------------------------------------------------------------------


class HealthUser(HttpUser):
    """Usuário que verifica o health check com alta frequência."""

    wait_time = between(1, 3)
    weight = 3  # 60% dos usuários spawned

    @task
    def check_health(self) -> None:
        with self.client.get("/api/v1/health", catch_response=True) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") != "ok":
                    resp.failure(f"Status degradado: {data.get('status')} | db={data.get('database')}")
            else:
                resp.failure(f"HTTP {resp.status_code}")


# ---------------------------------------------------------------------------
# QueryUser — realiza consultas jurídicas respeitando o rate limit padrão
# wait_time(8, 15) → ~4-7 req/min → abaixo do limite de 10/min por IP.
# Para múltiplos usuários, o limite é compartilhado — espere 429s com 2+.
# ---------------------------------------------------------------------------


class QueryUser(HttpUser):
    """Usuário que faz consultas jurídicas com cadência conservadora."""

    wait_time = between(8, 15)
    weight = 2  # 40% dos usuários spawned

    @task(4)
    def query_juridica(self) -> None:
        pergunta = random.choice(_PERGUNTAS)
        with self.client.post(
            "/api/v1/query",
            json={"question": pergunta},
            catch_response=True,
            timeout=90,  # LLM local (Ollama) pode demorar em hardware sem GPU
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if not data.get("answer"):
                    resp.failure("Resposta sem campo 'answer'")
                elif len(data.get("sources", [])) == 0:
                    # Não é falha, mas pode indicar problema de recuperação
                    resp.success()
            elif resp.status_code == 429:
                resp.failure("Rate limit atingido (429) — reduza usuários ou aumente RATE_LIMIT_PER_MINUTE")
            elif resp.status_code == 504:
                resp.failure("LLM timeout (504) — verifique se o modelo está disponível")
            else:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")

    @task(1)
    def health_intercalado(self) -> None:
        """Health check intercalado para medir disponibilidade durante carga."""
        self.client.get("/api/v1/health")


# ---------------------------------------------------------------------------
# StressQueryUser — teste de stress agressivo
# REQUER: servidor iniciado com RATE_LIMIT_PER_MINUTE=200 no .env
# Execute apenas com: locust ... --tags stress
# ---------------------------------------------------------------------------


class StressQueryUser(HttpUser):
    """Usuário de stress — alta taxa de requisições. Requer rate limit elevado."""

    wait_time = between(1, 3)
    weight = 0  # excluído do spawn padrão; ative via --tags stress

    @tag("stress")
    @task
    def query_stress(self) -> None:
        pergunta = random.choice(_PERGUNTAS)
        with self.client.post(
            "/api/v1/query",
            json={"question": pergunta},
            catch_response=True,
            timeout=90,
        ) as resp:
            # No stress test, 429 também é falha (limite deveria estar elevado)
            if resp.status_code not in (200,):
                resp.failure(f"HTTP {resp.status_code}")
