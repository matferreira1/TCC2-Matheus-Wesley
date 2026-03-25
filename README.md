# IAJuris

Sistema de recuperação e resposta de jurisprudência brasileira baseado em RAG (*Retrieval-Augmented Generation*). Combina busca textual sobre acórdãos do STF e teses do STJ com geração de respostas fundamentadas via LLM, expondo o pipeline como uma API REST.

---

## Como funciona

```
Pergunta do usuário
      │
      ▼
Busca FTS5 (BM25) ─── jurisprudencia (STF)   ┐
      │           └── teses_stj    (STJ)      │  SQLite FTS5
      │                                       ┘
      ▼
Montagem do prompt (v5)
  - Acórdãos STF relevantes (extração seções III e IV)
  - Teses STJ relevantes
      │
      ▼
LLM (Groq llama-3.3-70b-versatile  /  Ollama llama3.2:3b)
      │
      ▼
Resposta estruturada + fontes citadas
```

1. A pergunta é tokenizada e submetida a dois índices FTS5 em paralelo (`asyncio.gather`).
2. Os documentos recuperados têm suas ementas processadas por `_extract_ementa_payload()`, que prioriza as seções **III. RAZÕES DE DECIDIR** e **IV. DISPOSITIVO** das ementas estruturadas do STF.
3. O prompt v5 instrui o LLM a sintetizar os temas comuns, citar as fontes pelo número e usar o fallback quando não houver evidência suficiente.
4. A resposta é retornada via `POST /api/v1/query` com o texto e a lista de fontes usadas.

---

## Base de conhecimento

| Fonte | Conteúdo | Documentos |
|---|---|---|
| STF — Portal de pesquisa | Acórdãos exportados em CSV | 2.241 |
| STJ — Jurisprudência em Teses | Teses consolidadas por edição temática | 3.377 |
| **Total** | | **5.618** |

---

## Stack tecnológica

| Componente | Tecnologia |
|---|---|
| API REST | FastAPI 0.111 + Uvicorn |
| Banco de dados | SQLite com extensão FTS5 (via aiosqlite) |
| LLM primário | Groq API — `llama-3.3-70b-versatile` |
| LLM alternativo | Ollama local — `llama3.2:3b` |
| ETL | pandas 2.2 |
| Testes | pytest 8.2 + pytest-asyncio |
| Python | 3.10+ |

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

Os CSVs de acórdãos do STF devem estar em `data/raw/`. O arquivo de teses STJ (`JTSelecao.txt`) deve estar na raiz do projeto.

```bash
# Carregar acórdãos STF
.venv/bin/python -m etl.load

# Carregar teses STJ (na primeira vez ou com --force para recarregar)
.venv/bin/python -m etl.load_teses_stj
.venv/bin/python -m etl.load_teses_stj --force  # recarga completa
```

O ETL é idempotente: re-executá-lo sem `--force` não gera duplicatas.

---

## Executando a aplicação

```bash
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

No startup, a suite de testes é executada automaticamente e os resultados são exibidos nos logs antes do servidor começar a aceitar requisições.

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

A suite possui 85 testes cobrindo todos os módulos (ETL, busca, RAG, LLM, API). Todos os testes utilizam banco SQLite `:memory:` e mocks de LLM — nenhuma conexão de rede real é necessária para executá-los.

---

## Estrutura do projeto

```
iajuris/
├── main.py                   # Entrypoint FastAPI (lifespan, routers)
├── .env                      # Variáveis de ambiente (não versionado)
├── requirements.txt
├── pytest.ini
│
├── data/
│   ├── raw/                  # CSVs de acórdãos STF
│   └── db/
│       └── iajuris.db        # Banco SQLite com FTS5 (gerado pelo ETL)
│
├── etl/
│   ├── extract.py            # Leitura e deduplicação dos CSVs
│   ├── transform.py          # Normalização e limpeza dos dados
│   ├── load.py               # Carga dos acórdãos STF no banco
│   └── load_teses_stj.py     # Parser e carga das teses STJ
│
├── src/
│   ├── config/
│   │   ├── settings.py       # Pydantic Settings (lê .env)
│   │   └── logging_config.py # Configuração de logging
│   ├── database/
│   │   └── connection.py     # Conexão aiosqlite + DDL (tabelas + FTS5 + triggers)
│   ├── services/
│   │   ├── search_service.py # Busca BM25 via FTS5
│   │   ├── rag_service.py    # Pipeline RAG (busca → prompt → LLM)
│   │   ├── groq_service.py   # Cliente Groq API
│   │   └── ollama_service.py # Cliente Ollama (streaming HTTP)
│   └── api/
│       ├── routes/query.py   # Endpoint POST /api/v1/query
│       └── schemas/          # QueryRequest, QueryResponse, SourceDocument
│
└── tests/
    ├── conftest.py            # Fixture db (SQLite :memory: + schema + dados de referência)
    ├── test_search.py         # 23 testes — search() e search_teses()
    ├── test_rag.py            # 18 testes — answer(), _build_prompt(), _extract_ementa_payload()
    ├── test_ollama.py         # 9 testes  — generate(), health_check()
    ├── test_groq.py           # 11 testes — _get_client(), generate(), health_check()
    ├── test_api.py            # 12 testes — validação, sucesso, erros HTTP
    └── test_etl.py            # 12 testes — extract(), transform()
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
| `RAG_TOP_K` | `5` | Acórdãos STF recuperados por consulta |
| `RAG_TOP_K_TESES` | `3` | Teses STJ recuperadas por consulta |
| `RAG_MAX_EMENTA_CHARS` | `1500` | Limite de caracteres por ementa (extração inteligente de seções) |
| `DEBUG` | `false` | Habilita logging detalhado |
