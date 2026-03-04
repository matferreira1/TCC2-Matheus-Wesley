# Diário Técnico — IAJuris

> Registro cronológico de tudo que foi construído no projeto, escrito para subsidiar a redação do TCC.
> **Atualizado a cada mudança relevante.**

---

## Contexto do Projeto

**Nome:** IAJuris – Integração de IA Generativa com Bases Oficiais de Jurisprudência
**Autores:** Matheus Ferreira Diogo e Wesley Lira (UnB / FGA — Engenharia de Software)
**Objetivo:** Sistema RAG (Retrieval-Augmented Generation) que permite a advogados consultar jurisprudência brasileira em linguagem natural, recebendo respostas fundamentadas com citação dos acórdãos fonte.

**Decisões de projeto em relação ao TCC original:**
| Planejado no TCC | Implementado |
|---|---|
| Fonte: STJ Portal de Dados Abertos (JSON) | Fonte: STF (CSV exportado do portal) |
| LLM: Google Gemini | LLM: Ollama local (llama3.2:3b) |
| Busca vetorial | Busca BM25 via SQLite FTS5 |

A justificativa para FTS5 sobre vetores está no próprio TCC: *"busca jurídica depende de precisão de termos técnicos"*.

---

## Fase 1 — Configuração e Infraestrutura

### 1.1 Estrutura de diretórios

```
tcc/
├── data/
│   ├── raw/          # CSVs brutos do STF
│   └── db/           # iajuris.db (SQLite)
├── etl/              # Pipeline Extract → Transform → Load
├── src/
│   ├── api/
│   │   ├── routes/   # Endpoints FastAPI
│   │   └── schemas/  # Pydantic models de request/response
│   ├── config/       # Settings e logging
│   ├── database/     # Conexão aiosqlite
│   └── services/     # search, ollama, rag
├── tests/
├── main.py
└── requirements.txt
```

### 1.2 Configurações centralizadas (`src/config/settings.py`)

Foi usado **Pydantic-Settings** para centralizar todas as configurações em uma única classe `Settings`, com suporte a variáveis de ambiente via `.env`:

- `app_name`, `debug`
- `database_url` → `data/db/iajuris.db`
- `db_table_meta = "jurisprudencia"`, `db_table_fts = "jurisprudencia_fts"`
- `ollama_base_url = "http://localhost:11434"`, `ollama_model = "llama3.2:3b"`
- `rag_top_k = 5` — número de fragmentos recuperados pelo FTS5

A função `get_settings()` usa `@lru_cache` para que as configurações sejam carregadas apenas uma vez.

### 1.3 Conexão com o banco (`src/database/connection.py`)

- Usa `aiosqlite` para I/O assíncrono
- `row_factory = aiosqlite.Row` — permite acessar colunas por nome
- Cria as tabelas `jurisprudencia` e `jurisprudencia_fts` caso não existam
- Função `get_db()` usada como dependência do FastAPI via `Depends(get_db)`

### 1.4 Schema do banco de dados — SQLite FTS5

```sql
CREATE TABLE jurisprudencia (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    tribunal        TEXT NOT NULL,
    numero_processo TEXT,
    ementa          TEXT,
    decisao         TEXT,
    data_julgamento TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE jurisprudencia_fts
USING fts5(
    ementa,
    content='jurisprudencia',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 1'
);
```

**Decisão técnica — FTS5 como external content table:**
- `content='jurisprudencia'` faz a FTS5 apontar para a tabela principal em vez de duplicar os dados
- `tokenize='unicode61 remove_diacritics 1'` remove acentos na indexação e busca — essencial para jurisprudência em português (`habeas` bate com `Hábeas`)
- Após inserções, é necessário executar `INSERT INTO jurisprudencia_fts(jurisprudencia_fts) VALUES('rebuild')` para reconstruir o índice

---

## Fase 2 — ETL (Extract, Transform, Load)

### 2.1 Extract (`etl/extract.py`)

- Lê um ou mais CSVs do portal STF com `pandas.read_csv`
- Valida colunas obrigatórias: `Titulo`, `Relator`, `Data de publicação`, `Data de julgamento`, `Órgão julgador`, `Ementa`
- Concatena todos os DataFrames e **deduplica por `Titulo`** (keep="last")
- Motivo da deduplicação: o portal STF exporta arquivos acumulativos — o arquivo `(9).csv` já contém todos os registros dos arquivos `(1).csv` a `(8).csv`

**Lógica de seleção automática de CSVs (`_collect_csvs` em `load.py`):**
- Arquivos numerados `(1)` a `(N)` são acumulativos → seleciona apenas o de maior número
- O arquivo sem número (`resultados-de-acordaos.csv`) é tratado como lote independente
- Ambos são passados juntos para o `extract`, que deduplica

### 2.2 Transform (`etl/transform.py`)

- Remove espaços extras com `re.sub(r'\s+', ' ', text).strip()`
- Normaliza colunas para snake_case: `titulo`, `ementa`, `data_julgamento`
- Remove registros com ementa vazia

### 2.3 Load (`etl/load.py`)

- **Dropa e recria as tabelas a cada execução** — garante que não há dados duplicados entre runs
- Inserta todos os registros com `executemany`
- Reconstrói o índice FTS5 com `rebuild`

**Volumes:**
| Momento | Registros |
|---|---|
| Carga inicial | 250 acórdãos (1 CSV) |
| Após adição de 9 CSVs (2/mar/2026) | **2.241 acórdãos únicos** |

**Comando para reexecutar o ETL:**
```bash
cd /home/mat/trabalhos/tcc && .venv/bin/python -m etl.load
```

---

## Fase 3 — Serviços (Search, Ollama, RAG)

### 3.1 Serviço de busca (`src/services/search_service.py`)

**Algoritmo:**
1. Tokeniza a query do usuário
2. Filtra stopwords do português (`o`, `a`, `de`, `que`, etc.)
3. Constrói query FTS5 com operador `OR` entre os tokens
4. Executa a busca via BM25 (ranqueamento nativo do FTS5) e retorna os `top_k` resultados

**Problema crítico — FTS5 e parameterização:**
O SQLite FTS5 **não aceita `?` como placeholder** na cláusula `MATCH`. Tentativas de usar `cursor.execute("... MATCH ?", (query,))` geram erro silencioso ou resultado vazio. A solução foi usar **f-string com sanitização manual**:

```python
safe_query = " OR ".join(tokens)  # tokens já filtrados e sem caracteres especiais
sql = f"SELECT ... FROM jurisprudencia_fts('{safe_query}') AS f JOIN ..."
```

**Motivo do OR:** Em linguagem natural, `"requisitos para prisão preventiva"` ficaria como `requisitos OR prisão OR preventiva`, aumentando o recall. O BM25 penaliza automaticamente termos menos relevantes.

### 3.2 Serviço Ollama (`src/services/ollama_service.py`)

- Cliente HTTP assíncrono usando `httpx.AsyncClient`
- Usa **streaming** (`stream=True`) para não bloquear o event loop durante a geração
- Concatena os chunks de resposta até o chunk com `"done": true`

**Problema crítico — ReadTimeout:**
Inferência em CPU pura (sem GPU) pode levar de 20s a 300s. O `httpx` por padrão tem timeout de 5s, causando `ReadTimeout`. Solução:

```python
timeout = httpx.Timeout(None, connect=10.0)
# None = sem limite de leitura; 10s só para o handshake TCP inicial
```

### 3.3 Orquestrador RAG (`src/services/rag_service.py`)

Pipeline: `pergunta → FTS5 → contexto → prompt → Ollama → resposta`

**Truncamento de ementas (`_build_prompt`):**
Ementas juridicas são longas (média 1.300 chars). Com 5 acórdãos, o prompt chegava a ~6.500 chars, causando lentidão. Solução:

```python
textwrap.shorten(ementa, width=300, placeholder="...")
```

Resultado: 6.506 → 2.161 chars no prompt (~47% de redução), tempo: 177s → 94s.

**Prompt v3 (atual):**
```
Você é um assistente jurídico especializado em jurisprudência brasileira.
Use APENAS os acórdãos abaixo para responder. Se a resposta não estiver
nos acórdãos, diga: "Não encontrei informação suficiente nos acórdãos disponíveis."
Responda em português, em lista numerada quando houver múltiplos pontos.
OBRIGATÓRIO: após cada ponto, cite o número do acórdão entre parênteses.
Exemplo: '1. Ausência de ilegalidade flagrante (HC 266275 AgR).'

### Acórdãos:
[Acórdão 1] <numero_processo>
<ementa truncada a 300 chars>
...

### Pergunta:
<pergunta do usuário>

### Resposta:
```

**Evolução do prompt:**
- v1: instrução básica em português, sem regra de citação → LLM não citava acórdãos
- v2: adicionada regra "Cite o número do acórdão ao usar uma informação" → citações inconsistentes
- v3: `OBRIGATÓRIO` + exemplo concreto da citação → citações presentes em todos os itens

---

## Fase 4 — API REST (FastAPI)

### 4.1 Schemas (`src/api/schemas/query_schema.py`)

```python
class QueryRequest(BaseModel):
    question: str  # min_length=10, max_length=1000

class SourceDocument(BaseModel):
    tribunal: str
    numero_processo: str
    ementa: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
```

### 4.2 Endpoint (`src/api/routes/query.py`)

- `POST /api/v1/query` — recebe `QueryRequest`, retorna `QueryResponse`
- Usa `Depends(get_db)` — conexão gerenciada pelo FastAPI
- Tratamento de erros:
  - `httpx.TimeoutException` → HTTP 504 Gateway Timeout
  - `Exception` → HTTP 500 Internal Server Error

### 4.3 Aplicação principal (`main.py`)

- Usa `asynccontextmanager` para o lifespan da aplicação
- Configura logging na inicialização via `setup_logging(debug=...)`
- Router montado em `/api/v1`

**Comando para iniciar a API:**
```bash
cd /home/mat/trabalhos/tcc && .venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Exemplo de uso com curl:**
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Quais são os fundamentos para negar seguimento a um habeas corpus?"}'
```

---

## Sistema de Logging (`src/config/logging_config.py`)

Função `setup_logging(debug=False)`:
- Formato: `%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s`
- Silencia loggers ruidosos: `httpx`, `httpcore`, `uvicorn.access`
- Em modo `debug=True`, habilita `logging.DEBUG` em todos os módulos internos

**O que cada serviço loga:**
| Serviço | Log |
|---|---|
| `search_service` | query FTS5 montada + quantidade de resultados |
| `ollama_service` | modelo, tamanho do prompt em chars, tempo de resposta |
| `rag_service` | pergunta, prompt completo (entre separadores), tempo total do pipeline |

---

## Otimizações de Performance

### Threads do Ollama

Hardware: AMD Ryzen 5 2500U, 24 GB RAM, **sem GPU**.

Por padrão o Ollama usa todos os cores disponíveis (8 threads lógicos), o que saturava o sistema. Configuração via systemd override:

```ini
# /etc/systemd/system/ollama.service.d/threads.conf
[Service]
Environment="OLLAMA_NUM_THREADS=6"
```

```bash
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

Deixar 2 threads livres reduz jitter e melhora responsividade do SO durante a inferência.

### Seleção de modelo

| Modelo | Tempo (cold) | Qualidade |
|---|---|---|
| `deepseek-r1:1.5b` | ~30s | Incoerente para jurídico |
| `llama3.2:3b` | ~94s | Boa terminologia jurídica ✅ |
| `deepseek-r1:7b` | ~314s | Pior terminologia, muito lento |

**Modelo atual:** `llama3.2:3b`

---

## Problemas Encontrados e Soluções

| Problema | Causa | Solução |
|---|---|---|
| `httpx.ReadTimeout` | Inferência CPU demora minutos | `httpx.Timeout(None, connect=10.0)` |
| FTS5 retornando vazio | `MATCH ?` não funciona com placeholders | f-string com sanitização manual dos tokens |
| LLM não citava acórdãos | Instrução vaga no prompt | `OBRIGATÓRIO` + exemplo concreto no prompt v3 |
| Prompt muito longo (6.5k chars) | Ementas completas | `textwrap.shorten(width=300)` → 2.1k chars |
| `uvicorn` não encontrado | Não instalado no venv | `pip install uvicorn[standard]==0.41.0` |
| CPU saturada durante inferência | Ollama usando 8/8 threads | `OLLAMA_NUM_THREADS=6` via systemd |
| Banco com dados antigos ao recarregar | `CREATE TABLE IF NOT EXISTS` mantinha dados | `DROP TABLE` antes de criar no ETL |

---

## Estado Atual (2 de março de 2026)

- ✅ **Fase 1:** Configuração, schema do banco, conexão assíncrona
- ✅ **Fase 2:** ETL com suporte a múltiplos CSVs acumulativos — **2.241 acórdãos STF**
- ✅ **Fase 3:** Serviços de busca (FTS5 BM25), geração (Ollama streaming) e orquestração RAG
- ✅ **Fase 4:** API REST `POST /api/v1/query` com schemas Pydantic, tratamento de erros
- ✅ **Logging:** Todos os serviços + main.py instrumentados
- ✅ **Performance:** Truncamento de ementas + 6 threads Ollama
- ✅ **Qualidade:** Prompt v3 com citações obrigatórias + exemplo

**Próximos passos planejados:**
- [ ] Aumentar limite de truncamento de ementas (300 → 500 chars) — ver análise abaixo
- [ ] Phrase search para termos compostos (`"habeas corpus"` como unidade)
- [ ] Conjunto de avaliação: 5–10 perguntas de referência com respostas esperadas
- [ ] Métricas de recuperação: Recall@k, MRR, nDCG@k
- [ ] Métricas de geração: groundedness, relevância, coerência, fluência

---

## Testes com Base Ampliada (2 de março de 2026)

Após carregar 2.241 acórdãos, foram repetidas as 3 perguntas de referência.

### Resultados

| Pergunta | Fontes recuperadas | Resposta | Tempo |
|---|---|---|---|
| Fundamentos para negar HC | RHC 264563, HC 265955, HC 266275, RHC 265693, HC 265065 | 1 item com citação correta | 86,9s |
| Requisitos substituição prisão preventiva | ARE 1576893, HC 264048, HC 266764, Pet 14788, ARE 1528129 | 5 itens com citações | 143,8s |
| HC contra decisão monocrática do STJ | HC 263552, HC 264610, HC 263660, HC 264893, HC 264224 | "Não encontrei informação suficiente" | 105,0s |

### Análise

**Pergunta 1 — Parcialmente regressiva:**
A base maior ainda recupera os mesmos acórdãos relevantes. Porém a LLM gerou apenas 1 item, contra 3 que gerava antes. O contexto é o mesmo; variação de temperatura do modelo.

**Pergunta 2 — Melhora clara na recuperação:**
Com 250 acórdãos essa pergunta não trazia documentos específicos sobre medidas cautelares. Agora trouxe `Pet 14788` (lavagem de capitais, periculum libertatis) e `HC 266764` (descumprimento de medidas protetivas). A resposta tem 5 itens com citações, porém os itens 4 e 5 apresentam **alucinação** — artigos do Código Penal inventados (arts. 107, 144, 15, 20...) que não constam nas ementas.

**Pergunta 3 — Problema crítico identificado:**
A FTS5 recuperou **5 acórdãos perfeitamente alinhados** (todos tratam exatamente de "HC contra decisão monocrática do STJ, supressão de instância, inadmissibilidade"). Mesmo assim a LLM respondeu "Não encontrei informação suficiente".

**Causa provável:** o truncamento a 300 chars está cortando justamente a parte da ementa que contém a conclusão jurídica. Os 300 chars exibem apenas o cabeçalho padronizado dos acórdãos ("AGRAVO INTERNO... SUPRESSÃO DE INSTÂNCIA... AUSÊNCIA DE ILEGALIDADE EVIDENTE..."), mas o fundamento detalhado vem depois.

**Ação necessária:** aumentar o limite de truncamento de 300 para 500 chars e retestar.

**Problemas:** resposta ruim e tempo de resposta alto
**Possíveis soluções** Trocar llama3.2:3b por tinyllama ou phi3-mini	3–5× mais rápido
Quantização 4-bit (Q4_K_M no Ollama)	1.5–2× mais rápido
GPU (mesmo uma RTX 3060)	20–50× mais rápido
API externa (OpenAI, Groq, etc.)	< 5s por resposta

---

## Fase 5 — Substituição do Modelo LLM e Integração com API Groq (4 de março de 2026)

### 5.1 Motivação

A análise dos testes realizados em 2 de março de 2026 com o modelo `llama3.2:3b` executado localmente via Ollama evidenciou dois problemas interligados que comprometiam a viabilidade do ciclo de desenvolvimento:

1. **Latência de inferência proibitiva:** o tempo médio de resposta situava-se entre 86,9 s e 143,8 s por consulta (hardware: AMD Ryzen 5 2500U, 24 GB RAM, sem GPU). Esse intervalo tornava inviável qualquer processo iterativo de ajuste de prompt ou avaliação sistemática de qualidade, pois cada ciclo de edição-teste consumia vários minutos.

2. **Qualidade insuficiente do modelo:** o `llama3.2:3b`, por ser um modelo de 3 bilhões de parâmetros quantizado para execução em CPU, apresentou limitações estruturais para o domínio jurídico, incluindo alucinação de dispositivos legais inexistentes nas ementas e não conformidade com as instruções do prompt (e.g., inserção do fallback "Não encontrei informação suficiente" no meio de listas numeradas).

Constatou-se que aprimoramentos subsequentes — melhoria do prompt, avaliação formal, métricas de recuperação — não poderiam ser avaliados com confiabilidade enquanto o componente gerador permanecesse como fator limitante. Decidiu-se, portanto, substituir o modelo local por um modelo de maior capacidade hospedado em infraestrutura de nuvem, mantendo a arquitetura RAG inalterada.

### 5.2 Revisão do Prompt — Versão 4

Antes da troca do modelo, foram identificados três problemas específicos na versão 3 do prompt, diagnosticados a partir da saída da terceira pergunta de referência:

| Problema | Manifestação | Correção aplicada |
|---|---|---|
| Ausência de instrução de síntese | LLM listava cada acórdão separadamente em vez de agrupar por tema | Regra explícita: *"Identifique os temas comuns entre os acórdãos e SINTETIZE-os em poucos pontos claros"* |
| Uso incorreto do fallback | Frase "Não encontrei..." inserida dentro de lista numerada | Regra explícita: usar SOMENTE como resposta única e completa, nunca dentro de lista |
| Citação unitária | Exemplo de citação com um único acórdão por ponto | Exemplo reformulado com múltiplos acórdãos separados por `;` |
| Truncamento insuficiente | 500 chars cortavam o fundamento jurídico das ementas | Limite ampliado para 900 chars |

O prompt v4 introduziu uma seção estruturada de **REGRAS OBRIGATÓRIAS** (numeradas de 1 a 5) em substituição às instruções anteriores em prosa, com o objetivo de aumentar a aderência do modelo às restrições de formato e de grounding.

### 5.3 Integração com a API Groq

A Groq é uma plataforma de inferência que utiliza hardware proprietário (LPU — *Language Processing Unit*) para execução de modelos de linguagem com latência significativamente inferior à inferência em CPU convencional. O plano gratuito disponibiliza até 14.400 tokens por minuto, suficiente para os volumes de teste deste projeto.

**Modelo selecionado:** `llama-3.3-70b-versatile`
- 70 bilhões de parâmetros (vs. 3 bilhões do modelo anterior)
- Multilíngue com desempenho superior em português
- Disponível sem custo no tier gratuito da Groq

**Decisão arquitetural:** a integração foi implementada de forma a preservar toda a arquitetura existente. O sistema passou a suportar dois provedores LLM intercambiáveis mediante configuração, sem alteração no pipeline RAG ou nos endpoints da API.

**Arquivos modificados ou criados:**

| Arquivo | Alteração |
|---|---|
| `.env` | Criado com `LLM_PROVIDER`, `GROQ_API_KEY` e `GROQ_MODEL` |
| `src/config/settings.py` | Adicionados `llm_provider`, `groq_api_key`, `groq_model`, `groq_timeout_seconds` |
| `src/services/groq_service.py` | Novo serviço: cliente `AsyncGroq`, função `generate()`, `health_check()` |
| `src/services/rag_service.py` | Despacho condicional: `groq_service` ou `ollama_service` conforme `llm_provider` |
| `requirements.txt` | Adicionada dependência `groq>=0.11.0` |

**Parâmetros de geração configurados:**
```python
temperature=0.2   # baixa temperatura → respostas deterministas e factuais
max_tokens=2048
timeout=60        # segundos
```

A temperatura baixa (0,2) foi escolhida deliberadamente para o domínio jurídico, onde respostas deterministas e fundamentadas são preferíveis à criatividade generativa.

**Troca de provedor:** para alternar entre Groq e Ollama, basta modificar a variável `LLM_PROVIDER` no arquivo `.env`, sem qualquer alteração de código.

### 5.4 Avaliação Comparativa — Modelos llama3.2:3b (Ollama) vs. llama-3.3-70b-versatile (Groq)

As três perguntas de referência foram reaplicadas ao novo modelo, permitindo comparação direta.

#### 5.4.1 Resultados de Latência

| Pergunta | Ollama `llama3.2:3b` | Groq `llama-3.3-70b` | Redução |
|---|---|---|---|
| Fundamentos para negar HC | 86,9 s | 1,6 s | ~54× |
| Requisitos substituição prisão preventiva | 143,8 s | 1,0 s | ~144× |
| HC contra decisão monocrática do STJ | 105,0 s | 0,6 s | ~175× |

#### 5.4.2 Resultados de Qualidade

**Pergunta 1 — Fundamentos para negar habeas corpus**

Com o modelo Groq, a resposta passou de 1 item genérico para 3 pontos síntese, cada um com múltiplos acórdãos agrupados:

> 1. O habeas corpus não pode ser utilizado como sucedâneo de revisão criminal (RHC 264563 AgR; RHC 265693 AgR; HC 265065 AgR).
> 2. A ausência de ilegalidade flagrante, abuso de poder ou teratologia é fundamento para negar seguimento (HC 265955 AgR; HC 266275 AgR; RHC 265693 AgR).
> 3. A reiteração de argumentos sem novos elementos é motivo para negar provimento ao agravo (HC 265955 AgR; HC 266275 AgR; RHC 265693 AgR; HC 265065 AgR).

**Pergunta 2 — Requisitos para substituição da prisão preventiva**

A alucinação de artigos legais inexistentes, presente na resposta anterior do `llama3.2:3b`, foi eliminada. A resposta ficou integralmente fundamentada nos acórdãos recuperados, sem referência a dispositivos não mencionados nas ementas.

**Pergunta 3 — HC contra decisão monocrática do STJ**

Problema crítico corrigido: a resposta anterior ("Não encontrei informação suficiente") foi substituída por síntese correta dos 5 acórdãos em um único ponto, atribuindo a inadmissibilidade à configuração de supressão de instância — que é exatamente o fundamento presente em todos os acórdãos recuperados.

### 5.5 Testes com Perguntas de Domínio Cível

Foram testadas 5 perguntas adicionais abrangendo temas de direito civil e do consumidor, com o objetivo de avaliar o comportamento do sistema fora do domínio predominante da base (direito processual penal).

| Pergunta | Fontes recuperadas | Diagnóstico |
|---|---|---|
| Dano moral por inscrição indevida em cadastros | Rcl 86504 AgR, ARE 1571943 AgR, ARE 1573687 AgR | Resposta parcial — base contém AREs sobre repercussão geral, mas não acórdãos específicos do tema |
| Prisão após condenação em 2ª instância | HC 264497 AgR, RE 1546836 AgR-EDv-AgR, HC 265469 AgR | Fontes recuperadas potencialmente relevantes, mas ementas truncadas não chegam ao fundamento decisório |
| Revisão de contrato bancário por juros abusivos | Rcl 82751 AgR, ARE 1573140 AgR, Rcl 83111 AgR | "Não encontrei" — correto; base não cobre direito contratual bancário |
| Cumulação de dano moral e material em acidente de trânsito | ARE 1571943 AgR, ARE 1573687 AgR | "Não encontrei" — correto; tema ausente na base |
| Responsabilidade civil de plano de saúde | ARE 1579675 AgR, RE 1561570 AgR, Rcl 85605 AgR | Resposta desviada — recuperou acórdãos sobre responsabilidade solidária de entes federados (SUS), não plano de saúde privado |

#### Análise

Os resultados revelam uma limitação estrutural do sistema: a base de acórdãos coletada é predominantemente composta por matéria **criminal e processual penal** (habeas corpus, agravos internos, prisão preventiva, supressão de instância). Perguntas de direito civil, do consumidor ou contratual não encontram documentos relevantes na base, e o sistema responde corretamente com o fallback — com exceção da pergunta 5, em que houve recuperação espúria por sobreposição semântica parcial do termo "saúde".

Este comportamento é academicamente relevante, pois demonstra que:
1. O sistema é capaz de reconhecer ausência de evidência (comportamento de honestidade epistêmica do modelo);
2. A qualidade da recuperação é limitada pela cobertura temática da base, não pela capacidade do modelo gerador;
3. A expansão da base com acórdãos de outras matérias é condição necessária para ampliar o escopo de utilidade do sistema.

### 5.6 Estado Após as Mudanças de 4 de Março de 2026

- ✅ **Prompt v4:** regras estruturadas, síntese por tema, citação múltipla por ponto
- ✅ **Groq API:** integração com `llama-3.3-70b-versatile`, latência < 2 s por consulta
- ✅ **Provedores intercambiáveis:** `LLM_PROVIDER=groq|ollama` no `.env`
- ✅ **Eliminação de alucinações** observadas com o modelo anterior
- ⚠️ **Limitação identificada:** base de acórdãos restrita ao domínio criminal/processual penal

**Próximos passos:**
- [ ] Definir escopo: ampliar base ou documentar limitação e focar no domínio criminal
- [ ] Implementar conjunto de avaliação formal (10 perguntas com gabarito)
- [ ] Métricas de recuperação: Recall@5, MRR
- [ ] Métricas de geração: groundedness (taxa de respostas fundamentadas nos acórdãos)

---

## Fase 6 — Extração Semântica de Seções e Análise de Grounding (4 de março de 2026)

### 6.1 Diagnóstico: Loss of Semantic Payload

Após a integração com a Groq (Fase 5), foi identificado um problema residual de qualidade denominado *loss of semantic payload* na etapa de montagem do prompt. A descrição do problema:

> *"A FTS5 recuperou documentos potencialmente corretos. Mas o trecho fornecido ao LLM não continha a parte relevante. Resultado: o modelo não teve contexto suficiente. Isso não é falha do RAG — é limitação do chunking/truncamento."*

A investigação confirmou o diagnóstico. Foi realizada uma inspeção da distribuição de tamanho das ementas na base:

| Estatística | Valor |
|---|---|
| Mínimo | 178 chars |
| Mediana | 1.457 chars |
| p75 | 2.016 chars |
| p90 | 2.729 chars |
| p99 | 4.775 chars |
| Máximo | 9.821 chars |
| Ementas ≤ 900 chars | ~20% da base |
| Ementas ≤ 1.500 chars | ~50% da base |

O truncamento anterior de 900 chars cortava a maioria das ementas (mediana = 1.457 chars) antes das seções juridicamente decisivas. A inspeção manual das ementas do grupo HC revelou o padrão:

```
[0–870 chars]   Cabeçalho + I. CASO EM EXAME + II. QUESTÃO EM DISCUSSÃO
[870–1020 chars] III. RAZÕES DE DECIDIR + IV. DISPOSITIVO  ← cortado pelo truncamento
```

As seções **III. RAZÕES DE DECIDIR** e **IV. DISPOSITIVO** — que contêm o fundamento jurídico real da decisão — eram sistematicamente eliminadas do contexto enviado ao modelo.

### 6.2 Solução: Extrator de Seções Estruturais

A solução implementada foi a função `_extract_ementa_payload()` em `src/services/rag_service.py`, que substitui o `textwrap.shorten` ingênuo por extração inteligente baseada na estrutura padronizada das ementas do STF.

**Estrutura padronizada das ementas STF:**

```
[Cabeçalho]           Matéria e palavras-chave em caixa alta
I. CASO EM EXAME      Descrição dos fatos do processo
II. QUESTÃO EM DISCUSSÃO  Questão jurídica central
III. RAZÕES DE DECIDIR    Fundamentos jurídicos ← payload semântico principal
IV. DISPOSITIVO           Resultado do julgamento ← payload semântico principal
```

**Estratégia de extração (por ordem de prioridade):**

1. Se `len(ementa) ≤ max_chars` → retorna completa (sem perda alguma)
2. Se a ementa possui seções romanas detectadas → extrai `cabeçalho + III + IV`, truncando a seção III apenas se necessário para respeitar o limite
3. Fallback (ementa sem estrutura de seções) → `textwrap.shorten` preservando o início

O parâmetro `rag_max_ementa_chars = 1500` foi adicionado ao `settings.py`, configurável via `.env`.

**Resultado da extração nas ementas mais longas da base:**

| Acórdão | Original | Após extração | Estratégia usada |
|---|---|---|---|
| AP 2577 | 9.821 chars | 1.493 chars | Extração III + IV |
| MS 40336 ED-AgR | 8.538 chars | 1.492 chars | Extração III + IV |
| AP 2591 | 8.365 chars | 1.496 chars | Extração III + IV |
| HC 263552 AgR | 1.018 chars | 1.018 chars | Completa (dentro do limite) |

### 6.3 Impacto no Pipeline RAG

Com o extrator ativo, o pipeline foi reexecutado nas 3 perguntas de referência:

| Pergunta | Resultado anterior (trunc. 900) | Resultado com extrator (1500) |
|---|---|---|
| Fundamentos para negar HC | 3 pontos, citações corretas | 3 pontos, citações corretas |
| Requisitos substituição preventiva | 3 pontos, sem alucinação | 3 pontos + **art. 312 CPP e arts. 10; 14, §2º; 40 VII LEP** mencionados corretamente |
| HC contra decisão monocrática STJ | 1 ponto correto | 1 ponto correto |

O ganho mais visível foi na **Pergunta 2**: com as seções III e IV acessíveis, o modelo passou a citar dispositivos legais reais (`art. 312, caput e § 2º do CPP`; `arts. 10; 14, § 2º; e 40, VII da LEP`) presentes nas ementas, que antes estavam além do corte e eram alucinados ou omitidos.

### 6.4 Análise de Grounding — Verificação Manual das Respostas

Foi realizada uma análise de *grounding* — verificação ponto a ponto da fidelidade das respostas do modelo em relação aos acórdãos originais recuperados. Foram verificados os 15 acórdãos citados nas 3 perguntas de referência.

#### Critérios de avaliação

| Classificação | Critério |
|---|---|
| ✅ Correto | Afirmação diretamente sustentada pelo texto do acórdão citado |
| ⚠️ Parcial | Afirmação verdadeira, mas o acórdão citado para esse ponto específico contém apenas parte da evidência |
| ❌ Incorreto | Acórdão citado não contém evidência para a afirmação específica |

---

#### Pergunta 1 — Fundamentos para negar seguimento a um habeas corpus

**Ponto 1:** *"O HC não pode ser utilizado como sucedâneo de revisão criminal"* (RHC 264563 AgR; RHC 265693 AgR; HC 265065 AgR)

| Acórdão | Evidência no texto original | Resultado |
|---|---|---|
| RHC 264563 AgR | *"o habeas corpus não pode ser utilizado como sucedâneo de revisão criminal"* | ✅ Correto |
| RHC 265693 AgR | *"Inviabilidade de utilização do habeas corpus como sucedâneo de revisão criminal"* | ✅ Correto |
| HC 265065 AgR | *"Habeas corpus utilizado como sucedâneo de revisão criminal. Ausência de ilegalidade ou de teratologia."* | ✅ Correto |

**Ponto 2:** *"Ausência de ilegalidade flagrante, abuso de poder ou teratologia"* (RHC 264563 AgR; HC 266275 AgR; RHC 265693 AgR; HC 265065 AgR)

| Acórdão | Evidência no texto original | Resultado |
|---|---|---|
| RHC 264563 AgR | Menciona apenas *"ilegalidade flagrante"* — não menciona abuso de poder ou teratologia | ⚠️ Parcial |
| HC 266275 AgR | *"Ausência de ilegalidade flagrante, abuso de poder ou teratologia"* — os 3 elementos | ✅ Correto |
| RHC 265693 AgR | *"Ausência de flagrante ilegalidade, abuso de poder ou teratologia"* — os 3 elementos | ✅ Correto |
| HC 265065 AgR | *"Ausência de ilegalidade ou de teratologia"* — 2 de 3 elementos, sem menção a abuso de poder | ⚠️ Parcial |

**Ponto 3:** *"Reiteração de argumentos sem novos elementos torna o recurso inviável"* (HC 265955 AgR; HC 266275 AgR; RHC 265693 AgR; HC 265065 AgR)

| Acórdão | Evidência no texto original | Resultado |
|---|---|---|
| HC 265955 AgR | *"O recurso mostra-se inviável, na medida em que contém apenas a reiteração dos argumentos [...] sem, no entanto, revelar quaisquer elementos capazes de afastar as razões expressas na decisão agravada"* | ✅ Correto |
| HC 266275 AgR | Idem (fórmula padronizada) | ✅ Correto |
| RHC 265693 AgR | Idem | ✅ Correto |
| HC 265065 AgR | Idem | ✅ Correto |

---

#### Pergunta 2 — Requisitos para substituição da prisão preventiva

**Ponto 1:** *"Exige garantia da ordem pública e aplicação da lei penal"* (HC 262368 AgR; HC 264048 AgR; HC 265485 AgR; HC 266764 AgR)

| Acórdão | Evidência no texto original | Resultado |
|---|---|---|
| HC 262368 AgR | *"custódia cautelar foi assentada na necessidade da medida para garantia da ordem pública"* | ✅ Correto |
| HC 264048 AgR | Menciona art. 312 de passagem, mas o tema central é **supressão de instância** — pertence ao grupo da P3 | ⚠️ Parcial — erro de agrupamento temático |
| HC 265485 AgR | *"a gravidade do crime [...] nos termos do art. 312, caput e § 2°"* | ✅ Correto |
| HC 266764 AgR | *"devidamente motivada e fundamentada no receio de perigo gerado pelo estado de liberdade do paciente, nos exatos termos do art. 312"* | ✅ Correto |

**Ponto 2:** *"Substituição por prisão domiciliar depende de laudo médico analisado pelo juízo da execução"* (HC 262368 AgR; HC 265485 AgR; HC 266764 AgR)

| Acórdão | Evidência no texto original | Resultado |
|---|---|---|
| HC 262368 AgR | Trata de ausência de elementos de saúde, mas não menciona laudo médico analisado pelo juízo | ⚠️ Parcial |
| HC 265485 AgR | Sobre homicídio qualificado tentado. **Não trata de laudo médico nem de prisão domiciliar** | ❌ Citação não sustenta este ponto |
| HC 266764 AgR | *"o laudo médico colacionado a estes autos sequer foi analisado pelo juízo da execução"* + competência do juízo da execução | ✅ Correto — única fonte real para este ponto |

**Ponto 3:** *"Competência do juízo da execução, arts. 10; 14, §2º; e 40, VII da LEP"* (HC 266764 AgR)

| Acórdão | Evidência no texto original | Resultado |
|---|---|---|
| HC 266764 AgR | *"obrigatoriedade da observância, pela Administração Penitenciária e pelo Juízo da Execução, dos arts. 10; 14, § 2º; e 40, VII, todos da Lei de Execução Penal"* | ✅ Correto — transcrição literal |

---

#### Pergunta 3 — HC contra decisão monocrática do STJ

**Ponto 1:** *"O STF não admite HC contra decisão monocrática do STJ por supressão de instância"* (todos os 5 acórdãos)

| Acórdão | Evidência no texto original | Resultado |
|---|---|---|
| HC 263552 AgR | *"Não se admite habeas corpus contra decisão monocrática de ministro de Tribunal Superior, por ficar configurada indevida supressão de instância."* | ✅ Correto |
| HC 264610 AgR | Idem (variação: *"sob pena de ficar configurada"*) | ✅ Correto |
| HC 263660 AgR | Idem | ✅ Correto |
| HC 264893 AgR | Idem | ✅ Correto |
| HC 264224 AgR | Idem | ✅ Correto |

---

#### Placar de Grounding

| Pergunta | Pontos ✅ Corretos | Pontos ⚠️ Parciais | Pontos ❌ Incorretos | Total de citações verificadas |
|---|---|---|---|---|
| P1 | 7 | 2 | 0 | 11 |
| P2 | 5 | 2 | 1 | 10 |
| P3 | 5 | 0 | 0 | 5 |
| **Total** | **17** | **4** | **1** | **26** |

**Taxa de grounding:** 17/26 citações completamente sustentadas (65%); 21/26 sustentadas total ou parcialmente (81%); 1/26 incorreta (4%).

#### Interpretação

O único erro de grounding categórico (HC 265485 AgR atribuído ao ponto sobre laudo médico/prisão domiciliar) é um caso de **erro de agrupamento temático**: o modelo associou um acórdão sobre prisão preventiva em geral a um ponto específico sobre prisão domiciliar por condições de saúde. Não houve invenção de fatos, datas ou dispositivos legais inexistentes — todas as informações geradas estão presentes em algum acórdão da base.

Os casos parciais (⚠️) decorrem de o modelo ter generalizado para um acórdão uma característica que, naquele acórdão específico, aparece de forma incompleta, mas que é verdadeira para o conjunto. Este comportamento é esperado em modelos de linguagem operando sobre conjuntos de documentos com alta similaridade de conteúdo.

**Implicação técnica:** a principal causa de erros de atribuição não é falha do modelo gerador, mas sim a alta similaridade entre os documentos recuperados pelo FTS5. Acórdãos do mesmo tipo (AgR em HC) seguem fórmulas padronizadas quase idênticas, o que dificulta ao modelo distinguir precisamente qual afirmação provém de qual acórdão quando há sobreposição temática.

### 6.5 Estado Após a Fase 6

- ✅ **Extrator de seções:** `_extract_ementa_payload()` — prioriza III (Razões de Decidir) e IV (Dispositivo)
- ✅ **Configurável:** `rag_max_ementa_chars = 1500` em `settings.py`
- ✅ **Grounding verificado:** 81% de citações totalmente ou parcialmente sustentadas, 4% de erro categórico
- ✅ **Sem alucinação de conteúdo:** nenhum dispositivo legal, data ou fato inventado

**Próximos passos:**
- [ ] Definir escopo: ampliar base ou documentar limitação e focar no domínio criminal
- [ ] Implementar conjunto de avaliação formal (10 perguntas com gabarito)
- [ ] Métricas de recuperação: Recall@5, MRR
- [ ] Métricas de geração: groundedness automatizado (comparação resposta × ementas fonte)
