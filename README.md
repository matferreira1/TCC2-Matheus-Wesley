# IAJuris

Sistema de recuperação e resposta de jurisprudência brasileira baseado em RAG (*Retrieval-Augmented Generation*). Combina busca textual sobre acórdãos do STF e teses do STJ com geração de respostas fundamentadas via LLM, expondo o pipeline como uma API REST.

---

## Início rápido

```bash
# 1. Clonar e entrar no diretório
git clone https://github.com/matferreira1/TCC2-Matheus-Wesley.git
cd TCC2-Matheus-Wesley

# 2. Criar e ativar o ambiente virtual
python3 -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

# 3. Instalar as dependências
pip install -r requirements.txt

# 4. Criar o arquivo de configuração
cp .env.example .env            # edite e preencha GROQ_API_KEY

# 5. Carregar os dados (ETL)
python -m etl.load                 # acórdãos STF (data/raw/*.csv)
python -m etl.load_teses_stj       # teses STJ (data/stj/JTSelecao.txt)
python -m etl.load_sumulas_stj     # súmulas STJ (data/stj/SelecaoSumulas.txt)
python -m etl.generate_embeddings  # vetoriza o corpus (~10 min na 1ª vez)

# 6. Iniciar o servidor
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Acesse a interface web em **http://localhost:8000**

---

## Como funciona

```
Pergunta do usuário
      │
      ├── FTS5 (BM25) + query expansion ── jurisprudencia (STF)  ┐
      │                               └── teses_stj    (STJ)     │  SQLite FTS5
      │                                                           │
      ├── Semântica (cosine) ─────────── jurisprudencia (STF)    │  NumPy + MiniLM
      │                             └── teses_stj    (STJ)       │  sentence-transformers
      │                                                           ┘
      ▼
Reciprocal Rank Fusion (RRF, k=60) — funde rankings lexical + semântico
      │
      ▼
Cross-Encoder reranking — pontua pares (query, documento) diretamente
      │
      ▼
Montagem do prompt — acórdãos STF + teses/súmulas STJ (seções III e IV extraídas)
      │
      ▼
LLM (Groq llama-3.3-70b-versatile  /  Ollama llama3.2:3b)
      │
      ▼
Resposta estruturada + fontes citadas
```

1. A query é expandida com sinônimos jurídicos (`query_expansion.py`) e submetida a **quatro buscas em paralelo** via `asyncio.gather`: FTS5/BM25 e busca semântica (embeddings) sobre acórdãos STF e teses/súmulas STJ.
2. O **RRF** (Reciprocal Rank Fusion) funde os quatro rankings em até 20 candidatos por tipo.
3. O **cross-encoder** (`mmarco-mMiniLMv2-L12-H384-v1`) reordena os candidatos pontuando cada par (query, documento) diretamente, selecionando os top-k finais.
4. As ementas são processadas por `_extract_ementa_payload()`, que prioriza as seções **III. RAZÕES DE DECIDIR** e **IV. DISPOSITIVO**.
5. O **prompt v6** embute uma linha `Efeito:` em cada documento do contexto (casuístico / precedente qualificado / enunciado persuasivo), instrui a LLM a registrar divergências jurisprudenciais explicitamente em vez de sintetizá-las como consenso, e encerra a resposta com uma "Nota sobre as fontes:" indicando o peso vinculativo dos documentos utilizados.
6. A resposta é retornada via `POST /api/v1/query` com o texto e a lista de fontes.

---

## Base de conhecimento

| Fonte | Conteúdo | Documentos |
|---|---|---|
| STF — Portal de pesquisa | Acórdãos exportados em CSV (set/2025 – mar/2026) | 3.420 |
| STJ — Jurisprudência em Teses | Teses consolidadas por edição temática | 3.378 |
| STJ — Súmulas | Súmulas do STJ | 676 |
| **Total** | | **7.474** |

Todos os documentos possuem embedding vetorial (`paraphrase-multilingual-MiniLM-L12-v2`, 384 dims) armazenado como BLOB no SQLite, habilitando a busca semântica híbrida.

---

## Stack tecnológica

| Componente | Tecnologia |
|---|---|
| API REST | FastAPI 0.111 + Uvicorn |
| Banco de dados | SQLite com extensão FTS5 (via aiosqlite) |
| LLM primário | Groq API — `llama-3.3-70b-versatile` |
| LLM alternativo | Ollama local — `llama3.2:3b` |
| Busca semântica | sentence-transformers — `paraphrase-multilingual-MiniLM-L12-v2` |
| Reranking | sentence-transformers CrossEncoder — `mmarco-mMiniLMv2-L12-H384-v1` |
| Álgebra vetorial | NumPy |
| Rate limiting | slowapi 0.1.9 |
| ETL | pandas 2.2 |
| Testes | pytest 8.2 + pytest-asyncio |
| Python | 3.12 |

---

## Pré-requisitos

- Python 3.10 ou superior
- `pip` e `venv`
- [Conta Groq](https://console.groq.com) com chave de API (para o provedor padrão)
- Ollama instalado localmente (opcional, apenas se `LLM_PROVIDER=ollama`)

---

## Instalação

```bash
# 1. Clonar o repositório
git clone https://github.com/matferreira1/TCC2-Matheus-Wesley.git
cd TCC2-Matheus-Wesley

# 2. Criar e ativar o ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# 3. Instalar as dependências
pip install -r requirements.txt
```

---

## Configuração

Crie o arquivo `.env` na raiz do projeto:

```env
# Provedor LLM: "groq" (padrão) ou "ollama"
LLM_PROVIDER=groq

# Chave de API Groq (obrigatória se LLM_PROVIDER=groq)
GROQ_API_KEY=sua_chave_aqui

# Modelo Groq (opcional — padrão: llama-3.3-70b-versatile)
GROQ_MODEL=llama-3.3-70b-versatile

# Modo debug: habilita logs detalhados
DEBUG=false
```

> Para usar Ollama, defina `LLM_PROVIDER=ollama` e certifique-se de que o serviço está rodando em `http://localhost:11434` com o modelo `llama3.2:3b` disponível (`ollama pull llama3.2:3b`).

---

## Carga dos dados (ETL)

Os CSVs de acórdãos do STF devem estar em `data/raw/`. Os arquivos STJ (`JTSelecao.txt`, `SelecaoSumulas.txt`) devem estar em `data/stj/`.

```bash
# 1. Acórdãos STF
.venv/bin/python -m etl.load

# 2. Teses STJ (na primeira vez ou com --force para recarregar)
.venv/bin/python -m etl.load_teses_stj
.venv/bin/python -m etl.load_teses_stj --force  # recarga completa

# 3. Súmulas STJ
.venv/bin/python -m etl.load_sumulas_stj

# 4. Gerar embeddings (necessário para busca semântica)
.venv/bin/python -m etl.generate_embeddings
```

O ETL é idempotente: re-executar sem `--force` não gera duplicatas. O `generate_embeddings` pula documentos que já possuem embedding.

---

## Executando a aplicação

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

No startup, a suite de testes é executada automaticamente e os resultados são exibidos nos logs antes do servidor começar a aceitar requisições.

| Endereço | Descrição |
|---|---|
| `http://localhost:8000` | Interface web |
| `http://localhost:8000/docs` | Documentação interativa da API (Swagger) |
| `http://localhost:8000/api/v1/health` | Health check |

---

## Uso da API

### `POST /api/v1/query`

**Requisição:**
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Quais os fundamentos para negar seguimento a um habeas corpus?"}'
```

**Resposta:**
```json
{
  "answer": "1. O HC não pode ser utilizado como sucedâneo de revisão criminal (HC 265065 AgR; RHC 264563 AgR).\n2. A ausência de ilegalidade flagrante, abuso de poder ou teratologia é fundamento para negar seguimento (HC 266275 AgR).",
  "sources": [
    {
      "tribunal": "STF",
      "numero_processo": "HC 265065 AgR",
      "ementa": "HABEAS CORPUS. AGRAVO INTERNO...",
      "tipo": "acordao"
    },
    {
      "tribunal": "STJ",
      "numero_processo": "Ed. 39 — Tese 3",
      "ementa": "O habeas corpus não pode ser utilizado como sucedâneo de recurso ordinário...",
      "tipo": "tese_stj"
    }
  ]
}
```

**Validações:**
- Campo `question` obrigatório
- Mínimo de 10 caracteres, máximo de 1.000

**Documentação interativa:** `http://localhost:8000/docs`

---

## Testes

```bash
# Executar a suite completa
.venv/bin/python -m pytest tests/ -v

# Executar um módulo específico
.venv/bin/python -m pytest tests/test_search.py -v

# Com relatório de cobertura
.venv/bin/python -m pytest tests/ --tb=short
```

A suite possui **107 testes** cobrindo todos os módulos (ETL, busca, RAG, LLM, API, métricas). Todos os testes utilizam banco SQLite `:memory:` e mocks de LLM — nenhuma conexão de rede real é necessária para executá-los.

### Testes de carga (Locust)

```bash
# Instalar locust (já incluso no requirements.txt)
pip install locust

# Interface web — configure usuários e duração em http://localhost:8089
locust -f load_tests/locustfile.py --host http://127.0.0.1:8000

# Modo headless — 2 usuários, 2 minutos, relatório HTML
locust -f load_tests/locustfile.py --host http://127.0.0.1:8000 \
       --headless -u 2 -r 1 -t 120s --html load_tests/report.html
```

> **Rate limit:** o endpoint `/query` aceita 10 req/min por IP por padrão. Para testes de stress com mais usuários, defina `RATE_LIMIT_PER_MINUTE=200` no `.env` e reinicie o servidor antes de executar com `--tags stress`.

---

## Estrutura do projeto

```
iajuris/
├── main.py                   # Entrypoint FastAPI (lifespan, CORS, rate limit, security headers)
├── .env                      # Variáveis de ambiente (não versionado)
├── requirements.txt
├── pytest.ini
│
├── static/
│   └── index.html            # Interface web dark-themed (Tailwind + DM Sans + Playfair)
│
├── data/
│   ├── raw/                  # CSVs de acórdãos STF
│   ├── stj/                  # JTSelecao.txt, SelecaoSumulas.txt, jurispruConsumidor.txt
│   └── db/
│       └── iajuris.db        # Banco SQLite + FTS5 + embeddings BLOB
│
├── etl/
│   ├── extract.py            # Leitura e deduplicação dos CSVs
│   ├── transform.py          # Normalização e limpeza dos dados
│   ├── load.py               # Carga dos acórdãos STF no banco
│   ├── load_teses_stj.py     # Parser e carga das teses STJ (+ load_area())
│   ├── load_sumulas_stj.py   # Parser e carga das súmulas STJ
│   └── generate_embeddings.py# Vetorização do corpus (MiniLM → BLOB SQLite)
│
├── load_tests/
│   └── locustfile.py         # Cenários de carga: HealthUser, QueryUser, StressQueryUser
│
├── eval/                     # Framework de avaliação experimental
│   ├── dataset.json          # 20 perguntas jurídicas + 4 adversariais
│   ├── metrics.py            # Recall@k, MRR, nDCG@k, P@k
│   ├── retrieval_eval.py     # FTS5 vs LIKE sobre 20 perguntas
│   ├── generation_eval.py    # LLM-as-judge + grounding check
│   ├── latency_eval.py       # Benchmark p50/p95/p99
│   ├── compare_variants.py   # Variantes A/B/C/D
│   ├── run_evaluation.py     # CLI: python -m eval.run_evaluation all
│   └── results/              # JSON com resultados persistidos
│
├── src/
│   ├── config/
│   │   ├── settings.py       # Pydantic Settings (lê .env)
│   │   └── logging_config.py # Logging + _SecretFilter (redacta API keys)
│   ├── database/
│   │   └── connection.py     # Conexão aiosqlite + DDL (tabelas + FTS5 + triggers)
│   ├── services/
│   │   ├── search_service.py # Busca BM25 via FTS5 + query expansion
│   │   ├── semantic_service.py# Embeddings NumPy + cosine similarity + RRF
│   │   ├── rerank_service.py # Cross-encoder reranking (pós-RRF)
│   │   ├── query_expansion.py# Sinônimos jurídicos para FTS5
│   │   ├── rag_service.py    # Orquestrador: FTS5+semântica → RRF → rerank → LLM
│   │   ├── groq_service.py   # Cliente Groq API (singleton AsyncGroq)
│   │   └── ollama_service.py # Cliente Ollama (streaming HTTP)
│   └── api/
│       ├── limiter.py        # Instância slowapi.Limiter
│       ├── routes/
│       │   ├── query.py      # POST /api/v1/query (rate limit + injection check)
│       │   └── health.py     # GET /api/v1/health
│       └── schemas/          # QueryRequest, QueryResponse, SourceDocument
│
└── tests/
    ├── conftest.py            # Fixture db (SQLite :memory: + schema + dados de referência)
    ├── test_search.py         # 23 testes — search() e search_teses()
    ├── test_rag.py            # 26 testes — answer(), _build_prompt(), _extract_ementa_payload()
    ├── test_ollama.py         # 9 testes  — generate(), health_check()
    ├── test_groq.py           # 11 testes — _get_client(), generate(), health_check()
    ├── test_api.py            # 12 testes — validação, sucesso, erros HTTP
    ├── test_etl.py            # 12 testes — extract(), transform()
    └── test_metrics.py        # 22 testes — Recall@k, MRR, nDCG@k, P@k
```

---

## Variáveis de configuração disponíveis

| Variável | Padrão | Descrição |
|---|---|---|
| `LLM_PROVIDER` | `groq` | Provedor LLM: `groq` ou `ollama` |
| `GROQ_API_KEY` | — | Chave de API Groq (obrigatória para provedor Groq) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Modelo Groq |
| `OLLAMA_MODEL` | `llama3.2:3b` | Modelo Ollama |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL do servidor Ollama |
| `RAG_TOP_K` | `6` | Acórdãos STF retornados após reranking |
| `RAG_TOP_K_TESES` | `3` | Teses/súmulas STJ retornadas após reranking |
| `RAG_MAX_EMENTA_CHARS` | `1500` | Limite de caracteres por ementa (extração inteligente de seções) |
| `RERANKER_ENABLED` | `true` | Ativa cross-encoder pós-RRF (desativar para menor latência em CPU) |
| `RATE_LIMIT_PER_MINUTE` | `10` | Limite de requisições/min por IP em `/query` (aumentar para testes de carga) |
| `DEBUG` | `false` | Habilita logging detalhado |
