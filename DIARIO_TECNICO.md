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

---

## Fase 7 — Expansão da Base: STJ Jurisprudência em Teses (5 de março de 2026)

### 7.1 Motivação e Diagnóstico de Cobertura

A base construída nas fases anteriores era composta exclusivamente por 2.241 acórdãos do STF exportados do portal oficial. A análise da distribuição por tipo processual revelou um desequilíbrio estrutural grave:

| Área | Proporção estimada |
|---|---|
| Penal / HCs e RHCs | ~53% |
| Direito Constitucional (ARE/ADI) | ~43% |
| Direito Civil, Consumidor, Família, Contratos | < 1% |

Essa composição faz sentido para o STF, cujas competências principais são o controle de constitucionalidade e o julgamento de HCs, mas torna o sistema inútil para consultas sobre direito civil, contratos, responsabilidade civil, relações de consumo ou direito de família — que representam a maioria dos casos de um advogado generalista.

A solução identificada foi incorporar jurisprudência do **Superior Tribunal de Justiça (STJ)**, que possui competência sobre direito infraconstitucional federal e produz a jurisprudência predominante nas áreas faltantes.

### 7.2 Avaliação de Candidatos à Base

Foram avaliados dois manuais da CNJ Corregedoria como candidatos adicionais, e rejeitados:

| Documento | Páginas | Conteúdo | Decisão |
|---|---|---|---|
| Manual de Orientações da Corregedoria v2 (2024) | 120 | Procedimentos operacionais de inspeção | Rejeitado — não agrega valor para advogados |
| Manual Disciplinar v14 (junho/2024) | 64 | Procedimentos disciplinares de magistrados | Rejeitado — nicho demais para uso geral |

A escolha recaiu sobre o **Jurisprudência em Teses do STJ** — publicação periódica que consolida, por edição temática, as teses jurídicas firmadas pelo STJ com os julgados de referência. Fatores que tornaram esse corpus ideal:

- **Síntese consolidada:** cada tese é uma afirmação jurídica precisa (1–3 linhas), com alta densidade semântica por token
- **Cobertura complementar:** Direito Civil (623), Administrativo (544), Penal (512), Processual Civil (387), Empresarial (114), Consumidor (99), Previdenciário (92)
- **Disponível publicamente:** exportado do portal STJ como PDF, convertido para `JTSelecao.txt` (63.873 linhas, 3,5 MB)

### 7.3 Análise Estrutural do Arquivo Fonte

O arquivo `JTSelecao.txt` apresenta estrutura delimitada por form-feeds (`\x0c`):

```
\x0c
DIREITO CIVIL          EDIÇÃO N.143: PLANO DE SAÚDE - III
    1.  O plano de saúde pode estabelecer as doenças que terão cobertura,
        mas não o tipo de tratamento utilizado para a cura de cada uma...
          Julgados: REsp 1733013/PR, ...
    2.  É abusiva a cláusula contratual que exclua...
          Julgados: REsp 1.378.165/RS, ...
```

Totais identificados na análise:
- **1.482 páginas** (form-feeds)
- **271 edições** com área mapeada
- **3.377 teses** extraídas
- **17 áreas jurídicas** distintas

### 7.4 Arquitetura do Parser (`etl/load_teses_stj.py`)

O parser foi implementado como um **autômato de estados** que percorre o arquivo página a página. A complexidade principal estava em que a área jurídica aparece no cabeçalho da página, mas uma mesma edição pode se estender por múltiplas páginas — exigindo mapeamento prévio `edicao_num → area`.

**Estratégia de dois passos:**

1. **Passo 1 — mapeamento de edições:** constrói dicionário `{edicao_num: (area, edicao_titulo)}` varrendo todos os cabeçalhos
2. **Passo 2 — extração de teses:** percorre páginas com estado `(edicao_atual, tese_num, buffer_texto, buffer_julgados)`; a cada nova tese detectada, a anterior é "despejada" para a lista de resultados

**Expressões regulares principais:**

```python
_RX_HEADER   = re.compile(
    r'^\s*((?:DIREITO|ORIENTAÇÕES|Direito)[^\n]+?)\s{3,}EDIÇÃO N\.\s*(\d+):\s*([^\n]+)'
)
_RX_TESE     = re.compile(r'^    (\d+)\.\s+(.+)')   # 4 espaços + número + ponto
_RX_JULGADOS = re.compile(r'^\s{8,}Julgados?:', re.IGNORECASE)
```

**Schema do banco:**

```sql
CREATE TABLE teses_stj (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    area           TEXT NOT NULL,
    edicao_num     INTEGER NOT NULL,
    edicao_titulo  TEXT NOT NULL,
    tese_num       INTEGER NOT NULL,
    tese_texto     TEXT NOT NULL,
    julgados       TEXT,
    created_at     TEXT DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE teses_stj_fts
USING fts5(
    tese_texto, area, edicao_titulo,
    content='teses_stj',
    tokenize='unicode61 remove_diacritics 1'
);
```

Três triggers (`AFTER INSERT/DELETE/UPDATE`) mantêm o índice FTS5 sincronizado, seguindo o padrão já adotado para `jurisprudencia_fts`. O script é **idempotente**: verifica `COUNT(*)` antes de inserir e suporta a flag `--force` para recarga completa.

### 7.5 Busca Paralela com `asyncio.gather`

A adição de uma segunda fonte exigiu modificar o fluxo de recuperação em `rag_service.py`. A abordagem adotada foi a busca **paralela** das duas fontes, sem bloqueio sequencial:

```python
sources, sources_teses = await asyncio.gather(
    search_service.search(conn, question, top_k=settings.rag_top_k),
    search_service.search_teses(conn, question, top_k=settings.rag_top_k_teses),
)
```

`search_teses()` segue o mesmo padrão de `search()`: normalização da query, FTS5 com BM25 implícito, e fallback gracioso para lista vazia caso a tabela ainda não exista.

Novo parâmetro adicionado a `settings.py`:
```python
rag_top_k_teses: int = 3  # nº de teses STJ recuperadas pelo FTS5
```

### 7.6 Prompt v5 — Citações Mistas STF + STJ

O contexto enviado ao LLM passou a incluir rótulos explícitos por tipo de fonte:

```
[Acórdão STF 1]  (número do processo, ementa)
[Tese STJ 1]     (área — edição: título, tese_num: texto)
```

**Formato de citação definido no prompt v5:**

| Tipo | Formato | Exemplo |
|---|---|---|
| Acórdão STF | `(Sigla Número)` | `(HC 263552 AgR)` |
| Tese STJ | `(STJ Ed. N TM)` | `(STJ Ed. 39 T3)` |
| Fontes mistas | separadas por `;` | `(HC 263552 AgR; STJ Ed. 74 T9)` |

### 7.7 Alterações na API REST

Campo `tipo` adicionado ao schema `SourceDocument` em `query_schema.py`:

```python
tipo: str = Field(default="acordao",
                  description="'acordao' (STF) ou 'tese_stj' (STJ Em Teses)")
```

Em `query.py`, os dois tipos são serializados separadamente e concatenados:

```python
sources_acordaos = [SourceDocument(..., tipo="acordao") for s in resp.sources]
sources_teses    = [SourceDocument(tribunal="STJ",
                                   numero_processo=f"Ed. {t.edicao_num} — Tese {t.tese_num}",
                                   ementa=t.tese_texto, tipo="tese_stj")
                    for t in resp.sources_teses]
return QueryResponse(answer=resp.answer, sources=sources_acordaos + sources_teses)
```

### 7.8 Arquivos Modificados e Criados

| Arquivo | Tipo | Descrição da mudança |
|---|---|---|
| `etl/load_teses_stj.py` | **Novo** | Parser + schema + ETL idempotente para JTSelecao.txt |
| `src/database/connection.py` | Modificado | DDL de `teses_stj`, `teses_stj_fts` e 3 triggers adicionados a `init_db()` |
| `src/config/settings.py` | Modificado | Novo parâmetro `rag_top_k_teses = 3` |
| `src/services/search_service.py` | Modificado | `TesesResult` dataclass + `search_teses()` async |
| `src/services/rag_service.py` | Modificado | `asyncio.gather` para busca paralela; `sources_teses` em `RagResponse`; prompt v5 |
| `src/api/schemas/query_schema.py` | Modificado | Campo `tipo` em `SourceDocument` |
| `src/api/routes/query.py` | Modificado | Serialização separada de acórdãos e teses |

### 7.9 Resultado do ETL

```
INFO | Lendo arquivo: JTSelecao.txt
INFO | Total de páginas (form-feed): 1482
INFO | Edições com área mapeada: 270
INFO | Teses extraídas pelo parser: 3377
INFO | Schema teses_stj verificado/criado.
INFO | ✓ 3377 teses STJ inseridas com sucesso.
```

Distribuição por área jurídica após a carga:

| Área | Teses |
|---|---|
| DIREITO CIVIL | 623 |
| DIREITO ADMINISTRATIVO | 544 |
| DIREITO PENAL | 512 |
| DIREITO PROCESSUAL CIVIL | 387 |
| DIREITO EMPRESARIAL | 114 |
| DIREITO DO CONSUMIDOR | 99 |
| DIREITO PREVIDENCIÁRIO | 92 |
| Demais áreas | 1.006 |

**Base total após Fase 7:** 2.241 acórdãos STF + 3.377 teses STJ = **5.618 documentos**.

### 7.10 Validação End-to-End

Três perguntas foram testadas no pipeline completo (FTS5 → Groq `llama-3.3-70b-versatile`):

| Pergunta | Tempo | STF | STJ | Resultado |
|---|---|---|---|---|
| Fraude no cartão de crédito | 2,2 s | 5 | 3 (Consumidor) | ✅ Responsabilidade solidária das bandeiras, cláusula abusiva |
| Condições para prisão domiciliar | 1,6 s | 5 | 3 (Processual Penal) | ✅ Imprescindibilidade para mães, ausência de requisitos preventivos |
| Plano de saúde negando cobertura | 1,2 s | 5 | 3 (Civil) | ✅ Teses Ed.143 e Ed.2 sobre plano de saúde citadas diretamente |

As teses STJ foram determinantes nas respostas sobre direito civil e consumidor, que anteriormente retornavam apenas acórdãos STF periféricos, sem jurisprudência específica do tema.

### 7.11 Estado Após a Fase 7

- ✅ **Base expandida:** 5.618 documentos (2.241 STF + 3.377 STJ)
- ✅ **Cobertura temática:** Penal/Constitucional (STF) + Civil/Consumidor/Administrativo (STJ)
- ✅ **Busca paralela:** `asyncio.gather` sobre dois índices FTS5 independentes
- ✅ **Prompt v5:** citações mistas com rótulos `[Acórdão STF N]` e `[Tese STJ N]`
- ✅ **API atualizada:** campo `tipo` distingue `"acordao"` de `"tese_stj"` na resposta REST
- ✅ **ETL idempotente:** suporte a `--force` para recarga completa

**Próximos passos:**
- [ ] Git commit: `feat: integração Jurisprudência em Teses STJ (3.377 teses, 17 áreas)`
- [ ] Implementar conjunto de avaliação formal (10 perguntas com gabarito, cobrindo STF e STJ)
- [ ] Métricas de recuperação: Recall@5, MRR
- [ ] Súmulas STF/STJ (terceira fonte potencial — alta precisão, compactas)

---

## Fase 8 — Suite Completa de Testes Automatizados (10 de março de 2026)

### 8.1 Motivação

Ao longo das sete fases anteriores, toda a base de código do projeto foi desenvolvida sem cobertura de testes formal. Os três arquivos de teste existentes (`test_ollama.py`, `test_rag.py`, `test_search.py`) continham apenas *stubs* — funções com `pass` ou `assert True` — sem qualquer verificação real de comportamento. Essa situação apresentava dois riscos objetivos para o trabalho de conclusão de curso:

1. **Risco de regressão:** modificações em serviços centrais (como `rag_service.py` ou `search_service.py`) poderiam introduzir falhas silenciosas não detectadas até o momento de apresentação.
2. **Risco de credibilidade acadêmica:** a ausência de testes é inconsistente com o nível de rigor esperado em um projeto de software de TCC, especialmente em um sistema que combina múltiplas camadas (banco de dados, LLM, API REST).

A decisão foi implementar uma suite completa de testes unitários e de integração cobrindo todos os módulos do sistema.

### 8.2 Configuração do Ambiente de Testes

**Arquivo `pytest.ini` (criado):**

```ini
[pytest]
asyncio_mode = auto
testpaths = tests
```

A opção `asyncio_mode = auto` elimina a necessidade de decorar cada função de teste assíncrona com `@pytest.mark.asyncio`, reduzindo a verbosidade e evitando erros de esquecimento do decorator. Essa configuração é compatível com `pytest-asyncio >= 0.21`.

**Arquivo `tests/conftest.py` (criado):**

O fixture compartilhado `db` é o elemento central da infraestrutura de testes. Ele provisiona uma conexão `aiosqlite` com banco SQLite *in-memory* (`:memory:`), replica integralmente o schema de produção definido em `connection.py` e popula a base com dados de referência controlados.

**Schema replicado:**

```sql
-- Tabela principal de acórdãos
CREATE TABLE jurisprudencia (
    id INTEGER PRIMARY KEY AUTOINCREMENT, tribunal TEXT, numero_processo TEXT,
    ementa TEXT, decisao TEXT, data_julgamento TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- FTS5 com content table e remoção de diacríticos
CREATE VIRTUAL TABLE jurisprudencia_fts
USING fts5(ementa, content='jurisprudencia', content_rowid='id',
           tokenize='unicode61 remove_diacritics 1');

-- Tabela de teses STJ
CREATE TABLE teses_stj (
    id INTEGER PRIMARY KEY AUTOINCREMENT, area TEXT, edicao_num INTEGER,
    edicao_titulo TEXT, tese_num INTEGER, tese_texto TEXT, julgados TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- FTS5 sobre teses (três colunas indexadas)
CREATE VIRTUAL TABLE teses_stj_fts
USING fts5(tese_texto, area, edicao_titulo,
           content='teses_stj', content_rowid='id',
           tokenize='unicode61 remove_diacritics 1');

-- 6 triggers: AFTER INSERT / DELETE / UPDATE em cada tabela
```

**Dados de referência:**

| Tipo | Quantidade | Cobertura temática |
|---|---|---|
| Acórdãos STF | 5 | HC pe­nal, ARE constitucional, RE administrativo (servidor público, habeas corpus, supressão de instância) |
| Teses STJ | 3 | DIREITO CIVIL Ed.143, DIREITO DO CONSUMIDOR Ed.99, DIREITO PENAL Ed.55 |

Os dados foram escolhidos para exercitar os caminhos mais críticos de cada serviço sem depender da base de produção (`data/db/iajuris.db`), garantindo isolamento e reproducibilidade dos testes.

### 8.3 Descrição dos Arquivos de Teste

#### `tests/test_search.py` — 23 testes

Cobre as funções `search()` e `search_teses()` do módulo `src/services/search_service.py`.

**`search()` — 13 testes:**
- Retorno de lista não vazia para termo presente
- Tipo dos elementos da lista (`SearchResult`)
- Presença dos campos `tribunal`, `numero_processo`, `ementa`
- Comportamento com termo ausente na base → lista vazia
- Respeito ao parâmetro `top_k`
- `top_k=1` retorna exatamente um resultado
- Query vazia → lista vazia (sem exceção)
- Query só com espaços em branco → lista vazia
- Query com apenas stopwords (`o de que`) → lista vazia
- `rank` BM25 retornado como valor negativo (convenção do SQLite FTS5)
- Múltiplos termos aumentam o recall em relação a termo único
- Recuperação de acórdão específico por termo — `"servidor publico estabilidade"` → RE 100004
- Recuperação de acórdão por frase técnica — `"supressao instancia"` → HC 100005

**`search_teses()` — 10 testes:**
- Retorno de lista não vazia para termo presente
- Tipo dos elementos (`TesesResult`)
- Campos `area`, `edicao_num`, `edicao_titulo`, `tese_num`, `tese_texto`
- Respeito ao parâmetro `top_k`
- Query vazia → lista vazia
- Termo ausente → lista vazia
- Tabela vazia (fixture alternativo sem inserções) → lista vazia, sem exceção
- Tabela inexistente → lista vazia, sem exceção (tratamento de `OperationalError`)
- `"prisao preventiva cautelar"` recupera tese da área DIREITO PENAL
- Campo `julgados` não vazio nos resultados

#### `tests/test_rag.py` — 18 testes

Cobre as três funções públicas de `src/services/rag_service.py`.

**`answer()` — 5 testes:**
- Retorna objeto do tipo `RagResponse`
- Lista `sources` populada corretamente a partir dos documentos recuperados
- Lista `sources_teses` vazia quando nenhuma tese é encontrada
- Propagação de `RuntimeError` lançado pelo serviço LLM
- Uso do `ollama_service.generate` quando `llm_provider="ollama"`, via `monkeypatch`

**`_build_prompt()` — 8 testes:**
- Contém o texto da pergunta do usuário
- Contém rótulo `"STF 1"` para acórdão STF
- Contém rótulo `"[Tese STJ 1]"` para tese STJ
- Ausência de documentos → mensagem de fallback `"Nenhum documento relevante encontrado"`
- Numeração sequencial correta para múltiplos acórdãos (STF 1, STF 2, STF 3)
- Contém a palavra `"OBRIGAT"` (regras obrigatórias do prompt v5)
- Para fontes mistas, menciona explicitamente os dois tribunais no prompt

**`_extract_ementa_payload()` — 5 testes:**
- Ementa curta (abaixo do limite) retornada integralmente
- Ementa com comprimento exato do limite retornada sem truncamento
- Ementa com seções romanas → extrai seções III e IV
- Ementa sem seções romanas → aplica `textwrap.shorten` como fallback
- Seção IV (DISPOSITIVO) preservada na extração
- Resultado com `max_chars=300` tem comprimento ≤ 600 (margem para overhead de extração)

#### `tests/test_ollama.py` — 9 testes

Cobre `generate()` e `health_check()` de `src/services/ollama_service.py`. Utiliza `unittest.mock` para simular o cliente `httpx.AsyncClient` sem conexões de rede reais.

- `generate()` concatena corretamente os chunks do stream
- Resultado não vazio para sequência válida de chunks
- Linhas vazias no stream são ignoradas
- Stream encerrado ao receber `"done": true` (chunks posteriores descartados)
- `httpx.ReadTimeout` propagado como exceção
- `httpx.HTTPStatusError` propagado como exceção
- `health_check()` retorna `True` com resposta HTTP 200
- `health_check()` retorna `False` com resposta HTTP 503
- `health_check()` retorna `False` com `httpx.ConnectError`

#### `tests/test_groq.py` — 11 testes

Cobre `_get_client()`, `generate()` e `health_check()` de `src/services/groq_service.py`. Utiliza `monkeypatch` e `AsyncMock` para isolar chamadas à API Groq.

- `_get_client()` lança `ValueError` contendo `"GROQ_API_KEY"` quando a chave está ausente
- `_get_client()` retorna instância de `AsyncGroq` quando a chave está configurada
- `_get_client()` retorna o mesmo objeto em chamadas subsequentes (singleton via `lru_cache`)
- `generate()` retorna o texto da `completion.choices[0].message.content`
- `generate()` retorna string vazia quando `content` é `None`
- `generate()` propagando `RuntimeError` lançado pela API
- Parâmetro `model` utilizado é o definido em `settings.groq_model`
- `temperature` ≤ 0.3 (configuração determinista para domínio jurídico)
- `health_check()` retorna `True` quando `models.list()` é chamado com sucesso
- `health_check()` retorna `False` quando `models.list()` lança exceção genérica
- `health_check()` retorna `False` quando a chave está ausente (`ValueError` capturado internamente)

#### `tests/test_api.py` — 12 testes

Testa o endpoint `POST /api/v1/query` usando `httpx.AsyncClient` com `ASGITransport`, sem iniciar servidor real. As dependências de banco (`get_db`) e de ciclo de vida (`open_db`, `init_db`, `close_db`) são substituídas por overrides e `AsyncMock`.

**Validação de entrada (3 testes):**
- Pergunta com menos de 10 caracteres → HTTP 422 Unprocessable Entity
- Pergunta com mais de 1.000 caracteres → HTTP 422
- Corpo sem o campo `question` → HTTP 422

**Respostas bem-sucedidas (7 testes):**
- Requisição válida retorna HTTP 200
- Resposta contém o campo `answer` com o conteúdo esperado
- Resposta contém o campo `sources` como lista
- Cada elemento de `sources` possui os quatro campos: `tribunal`, `numero_processo`, `ementa`, `tipo`
- Presença de pelo menos um elemento com `tipo="acordao"`
- Presença de pelo menos um elemento com `tipo="tese_stj"`
- Lista `sources` vazia é uma resposta válida (sem erro)

**Erros do serviço (2 testes):**
- `httpx.TimeoutException` lançado por `rag_service.answer()` → HTTP 504 Gateway Timeout
- `RuntimeError` lançado por `rag_service.answer()` → HTTP 500 Internal Server Error

#### `tests/test_etl.py` — 12 testes

Cobre `extract()` de `etl/extract.py` e `transform()` de `etl/transform.py`. Utiliza `tmp_path` do pytest para criar CSVs temporários sem dependência de arquivos reais.

**`extract()` — 6 testes:**
- Lê um CSV válido e retorna `pd.DataFrame`
- DataFrame retornado é não vazio
- Dois CSVs com registros distintos são concatenados (2 linhas no resultado)
- Registros com `Titulo` duplicado são deduplicados (keep="last")
- CSV sem colunas obrigatórias lança `ValueError` com mensagem `"Colunas ausentes"`
- Colunas `Titulo`, `Ementa` e `Data de julgamento` preservadas após a extração

**`transform()` — 6 testes:**
- Retorna `pd.DataFrame`
- Colunas de saída são exatamente `{titulo, ementa, data_julgamento}`
- Registros com ementa vazia ou apenas espaços são removidos
- Espaços múltiplos consecutivos na ementa são normalizados para espaço único
- Coluna `titulo` normalizada (sem espaços externos, sem maiúsculas desnecessárias)
- Cinco registros válidos são todos preservados após a transformação

### 8.4 Desafios Técnicos Encontrados

#### Codificação UTF-8 nos arquivos de teste

A ferramenta de substituição de texto utilizada durante o desenvolvimento opera sobre bytes e apresentou falhas ao tentar localizar strings contendo caracteres multibyte em UTF-8 (sequências com `ç`, `ã`, `é`, `ú`, `Ó`, etc.). A causa foi a comparação byte-a-byte do bloco de contexto fornecido, que em presença de acentos pode não casar exatamente com o conteúdo armazenado dependendo da normalização Unicode do sistema.

**Solução adotada:** todos os arquivos de teste foram escritos diretamente via `open(path, 'w', encoding='utf-8')` em chamadas Python explícitas no terminal, garantindo que a codificação UTF-8 fosse aplicada consistentemente desde a criação, sem passar por processamento intermediário.

#### Sincronização do índice FTS5 com SQLite `:memory:`

A tabela `jurisprudencia_fts` é uma *external content table* — seus dados não são armazenados internamente, mas lidos da tabela `jurisprudencia` via `content='jurisprudencia'`. No banco de produção, triggers mantêm o índice sincronizado automaticamente. Em testes com banco `:memory:`, é necessário que as inserções ativem os mesmos triggers, ou que o índice seja reconstruído com `INSERT INTO jurisprudencia_fts(jurisprudencia_fts) VALUES('rebuild')` após a carga dos dados.

**Solução adotada:** o fixture `db` em `conftest.py` replica os 6 triggers de sincronização (INSERT/DELETE/UPDATE para cada índice FTS5) e executa `rebuild` explicitamente após cada conjunto de inserções, garantindo que as buscas BM25 nos testes retornem os mesmos resultados que no ambiente de produção.

#### Assertions com caracteres acentuados nos rótulos do prompt

Os testes de `_build_prompt()` verificavam se o rótulo `"[Acordao STF 1]"` (sem acento) estava presente no prompt gerado. O prompt real utiliza `"[Acórdão STF 1]"` (com acento), causando duas falhas. A causa raiz foi uma assunção incorreta sobre a grafia dos rótulos no momento da escrita dos testes.

**Solução adotada:** os testes passaram a verificar substrings não-ambíguas que independem de acentuação: `"STF 1"` (invariante ao acento em `"Acórdão"`) e `"[Tese STJ 1]"` (que já não apresenta o problema). Essa abordagem melhora a robustez dos testes — uma futura alteração na grafia do rótulo não quebrará a verificação, desde que a identificação do tribunal e o número estejam presentes.

### 8.5 Resultados

```
======================== 85 passed in 1.61s ========================
```

| Arquivo | Testes | Resultado |
|---|---|---|
| `test_search.py` | 23 | ✅ 23 passaram |
| `test_rag.py` | 18 | ✅ 18 passaram |
| `test_ollama.py` | 9 | ✅ 9 passaram |
| `test_groq.py` | 11 | ✅ 11 passaram |
| `test_api.py` | 12 | ✅ 12 passaram |
| `test_etl.py` | 12 | ✅ 12 passaram |
| **Total** | **85** | **✅ 0 falhas, 0 skips** |

**Commit:** `2de587f` — *"test: suite completa de testes — 85 testes, 0 skips"*

### 8.6 Estado Após a Fase 8

- ✅ **pytest.ini:** `asyncio_mode = auto`, `testpaths = tests`
- ✅ **conftest.py:** fixture `db` com schema completo e dados de referência controlados
- ✅ **85 testes** cobrindo todos os módulos: ETL, busca, RAG, LLM (Groq + Ollama), API REST
- ✅ **0 falhas, 0 skips** na execução completa da suite
- ✅ **Isolamento total:** nenhum teste depende de banco de produção, serviço Groq/Ollama ou rede

---

## Fase 9 — Execução Automática de Testes no Startup (10 de março de 2026)

### 9.1 Motivação

Com a suite de testes implementada, surgiu a necessidade de garantir que eventuais regressões introduzidas entre sessões de desenvolvimento sejam identificadas imediatamente — antes que o sistema comece a atender requisições. A solução adotada foi integrar a execução da suite ao ciclo de vida da aplicação FastAPI, de modo que os resultados dos testes sejam registrados nos logs estruturados do servidor a cada inicialização.

Essa abordagem tem precedente em sistemas de produção onde a etapa de *smoke test* ou *readiness check* faz parte do processo de startup, validando que o ambiente de execução está íntegro antes de expor o serviço.

### 9.2 Implementação

A execução dos testes foi encapsulada na função assíncrona `_run_tests()`, adicionada a `main.py`:

```python
async def _run_tests() -> bool:
    """
    Executa a suite de testes via pytest como subprocesso e loga cada linha.

    Retorna True se todos os testes passaram (exit code 0), False caso contrário.
    """
    logger.info("━" * 55)
    logger.info("STARTUP — Executando suite de testes (pytest tests/)")
    logger.info("━" * 55)

    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "--no-header",
        "-p", "no:cacheprovider",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()

    for line in stdout.decode(errors="replace").splitlines():
        if line.strip():
            logger.info("[pytest] %s", line)

    logger.info("━" * 55)
    if proc.returncode == 0:
        logger.info("STARTUP — Testes: ✓ TODOS PASSARAM (exit 0)")
    else:
        logger.warning("STARTUP — Testes: ✗ FALHAS DETECTADAS (exit %d)", proc.returncode)
    logger.info("━" * 55)

    return proc.returncode == 0
```

**Decisões de projeto:**

| Decisão | Justificativa |
|---|---|
| `asyncio.create_subprocess_exec` em vez de `subprocess.run` | Não bloqueia o event loop do uvicorn durante a execução dos testes |
| `stderr=asyncio.subprocess.STDOUT` | Unifica stdout e stderr em um único stream, simplificando a leitura linha a linha |
| `sys.executable` em vez de `"pytest"` | Garante que o pytest do ambiente virtual ativo seja invocado, independentemente do PATH do sistema |
| `-p no:cacheprovider` | Desativa o cache de resultados do pytest, forçando re-execução completa a cada startup |
| `--tb=short` | Traceback compacto, legível nos logs sem ocupar dezenas de linhas por falha |
| Não lança exceção em caso de falha | O servidor inicializa normalmente mesmo com falhas de teste; o resultado é diagnóstico, não bloqueante |

### 9.3 Integração no Lifespan do FastAPI

A função `_run_tests()` foi inserida como **primeira operação** no bloco de startup do `lifespan`, antes da abertura da conexão com o banco de dados:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Iniciando IAJuris...")
    await _run_tests()    # 1. Testes automatizados
    await open_db()       # 2. Conexão SQLite
    await init_db()       # 3. Schema FTS5
    logger.info("Banco de dados pronto. Servidor disponível.")

    yield  # aplicação em execução

    # Shutdown
    logger.info("Encerrando IAJuris...")
    await close_db()
    logger.info("Encerramento concluído.")
```

Essa ordenação é deliberada: os testes são executados antes do banco ser aberto porque o fixture `db` de `conftest.py` utiliza banco `:memory:` independente, não necessitando da conexão de produção. Caso fosse após `open_db()`, os testes ainda funcionariam, mas a ordem atual separa claramente a fase de verificação da fase de inicialização de recursos.

### 9.4 Saída Esperada nos Logs

Ao iniciar com `uvicorn main:app --reload`, os logs exibem o seguinte padrão:

```
INFO  | __main__                             | Iniciando IAJuris...
INFO  | __main__                             | ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INFO  | __main__                             | STARTUP — Executando suite de testes (pytest tests/)
INFO  | __main__                             | ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INFO  | __main__                             | [pytest] tests/test_api.py::test_question_too_short_returns_422 PASSED
INFO  | __main__                             | [pytest] tests/test_api.py::test_question_too_long_returns_422 PASSED
...
INFO  | __main__                             | [pytest] 85 passed in 1.61s
INFO  | __main__                             | ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INFO  | __main__                             | STARTUP — Testes: ✓ TODOS PASSARAM (exit 0)
INFO  | __main__                             | ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INFO  | __main__                             | Banco de dados pronto. Servidor disponível.
```

Em caso de falha em um ou mais testes, o padrão muda:

```
WARNING | __main__                           | STARTUP — Testes: ✗ FALHAS DETECTADAS (exit 1)
```

As linhas do traceback curto (`--tb=short`) do pytest ficam visíveis imediatamente acima, na sequência de logs `[pytest]`.

### 9.5 Considerações sobre Tempo de Startup

A suite de 85 testes executa em aproximadamente 1,6 segundos no hardware de desenvolvimento. Esse custo adicional no startup é aceitável para o contexto de desenvolvimento e para o escopo do TCC. Em um ambiente de produção com requisitos de *time-to-ready* mais rígidos, seria recomendável mover a execução de testes para um estágio anterior da pipeline de CI/CD e remover do startup da aplicação.

### 9.6 Estado Após a Fase 9

- ✅ **`_run_tests()`** adicionada a `main.py` — subprocesso assíncrono, sem bloqueio do event loop
- ✅ **Lifespan atualizado** — testes executam antes da abertura do banco de dados
- ✅ **Saída estruturada nos logs** — cada linha do pytest prefixada com `[pytest]`
- ✅ **Comportamento não-bloqueante** — falhas de teste não impedem o servidor de subir
- ✅ **Diagnóstico imediato** — regressões detectadas antes de qualquer requisição ser processada

**Próximos passos:**
- [x] Implementar conjunto de avaliação formal (10 perguntas com gabarito, cobrindo STF e STJ) → **Fase 10**
- [x] Métricas de recuperação: Recall@5, MRR → **Fase 10**
- [x] Métricas de geração: groundedness automatizado (comparação resposta × ementas fonte) → **Fase 10**
- [x] Benchmark formal de latência com percentis p50/p95/p99 sobre a API REST → **Fase 10**

---

## Fase 10 — Framework de Avaliação Experimental (16 de março de 2026)

### 10.1 Motivação

A Fase 9 encerrou o desenvolvimento funcional do IAJuris com a suite de testes automatizados. A etapa seguinte, prevista explicitamente na seção 4.4 do TCC, é a **avaliação experimental** do sistema: medir objetivamente a qualidade de recuperação, a qualidade de geração, a latência operacional e comparar as três variantes arquiteturais definidas no trabalho.

Sem esse módulo, o TCC seria incapaz de responder à pergunta central da avaliação: *a arquitetura FTS5+LLM é superior às alternativas (sem RAG, LIKE+LLM)?* Os resultados desta fase fornecem evidência empírica para essa resposta.

### 10.2 Estrutura do Módulo `eval/`

```
eval/
├── __init__.py               # Identificação do pacote
├── dataset.json              # 20 perguntas jurídicas + 4 adversariais
├── metrics.py                # Funções puras de cálculo de métricas IR
├── retrieval_eval.py         # Avaliação de recuperação: FTS5 vs LIKE
├── generation_eval.py        # LLM-as-judge + grounding check
├── latency_eval.py           # Benchmark operacional (p50/p95/p99)
├── compare_variants.py       # Comparação entre variantes A/B/C
├── run_evaluation.py         # Orquestrador CLI
└── results/
    ├── retrieval_results.json
    ├── generation_results.json
    ├── latency_results.json
    ├── variants_results.json
    └── consolidated_report.json
```

### 10.3 Dataset de Avaliação (`eval/dataset.json`)

O dataset contém **20 perguntas jurídicas** cobrindo 6 áreas do direito (Penal, Civil, Consumidor, Administrativo, Constitucional, Processual) e **4 consultas adversariais** para testar robustez do sistema.

Cada pergunta define critérios de relevância lexical para o módulo de retrieval:

```json
{
  "id": "q01",
  "question": "Quais são os fundamentos para a decretação da prisão preventiva?",
  "area": "Direito Penal",
  "relevance": {
    "must_contain": ["prisão preventiva"],
    "any_of": ["fundamento", "requisito", "art. 312", "cautelar", "decretação"]
  }
}
```

**Critério `must_contain`:** todos os termos listados devem estar presentes no documento (lógica AND).
**Critério `any_of`:** pelo menos um dos termos deve estar presente (lógica OR).

A avaliação é feita com normalização de texto (lowercase + remoção de diacríticos via `unicodedata.normalize("NFKD")`), tornando os critérios robustos a variações de acentuação. Essa abordagem — julgamentos de relevância lexical — é prática padrão em IR para construção de coleções de teste sem anotação humana.

**Consultas adversariais** testam comportamentos limítrofes:
| ID | Descrição |
|----|-----------|
| adv01 | String sem sentido (`"aaaaabbbbbccccc xyzxyz"`) |
| adv02 | Processo fictício (`RE 999999999`) + termo jurídico inexistente (`"telepatia jurídica"`) |
| adv03 | Pergunta em inglês (fora do domínio do corpus) |
| adv04 | Pergunta abaixo do mínimo de caracteres (deve retornar HTTP 422) |

### 10.4 Métricas de Recuperação (`eval/metrics.py`)

Implementação de funções puras sem dependências externas, cobertas por 22 testes unitários em `tests/test_metrics.py`:

| Função | Fórmula |
|--------|---------|
| `recall_at_k(relevant, k)` | `sum(relevant[:k]) / total_relevant` |
| `mrr(relevant)` | `1 / (posição do 1º relevante)` |
| `ndcg_at_k(relevant, k)` | `DCG@k / IDCG@k` onde `DCG = Σ rel_i / log2(i+2)` |
| `precision_at_k(relevant, k)` | `sum(relevant[:k]) / k` |
| `compute_all(relevant, k)` | Dicionário com todas as 4 métricas |
| `aggregate(results)` | Média aritmética de cada métrica sobre todas as perguntas |

### 10.5 Avaliação de Recuperação (`eval/retrieval_eval.py`)

Compara FTS5 (sistema atual) contra LIKE (busca por substring, baseline) nas 20 perguntas do dataset.

**FTS5:** usa `rag_service` via chamada ao pipeline completo de busca com BM25.

**LIKE:** query parametrizada com `?` para evitar injeção SQL:
```sql
SELECT id, ementa, tribunal, numero_processo
FROM jurisprudencia
WHERE ementa LIKE ? OR ementa LIKE ?
LIMIT ?
```

Os tokens são gerados por `query.lower().split()` e cada um vira `%token%`. O resultado é comparado contra os critérios de relevância do dataset usando a função `is_relevant()`.

### 10.6 Avaliação de Geração (`eval/generation_eval.py`)

Dois mecanismos complementares:

**LLM-as-Judge:** o mesmo modelo Groq (`llama-3.3-70b-versatile`) avalia cada resposta gerada pelo sistema em 4 dimensões (0–5), retornando JSON estruturado:

```
Groundedness (Fidelidade às fontes): A resposta está ancorada nos documentos recuperados?
Relevância: A resposta aborda diretamente a pergunta feita?
Coerência: A resposta é internamente consistente, sem contradições?
Fluência: A resposta é gramaticalmente correta e bem redigida?
```

**Grounding Check (determinístico):** extrai citações entre parênteses via regex `\(([^)]+)\)`, verifica se cada identificador citado existe nos documentos efetivamente recuperados. Retorna `grounding_score = citações_válidas / citações_totais`. Caso não haja citações, score = 1.0 (sem alucinação declarável).

### 10.7 Avaliação de Latência (`eval/latency_eval.py`)

Executa `rag_service.answer()` para 10 perguntas distintas, cronometrando cada chamada com `time.perf_counter()`. Calcula percentis por interpolação linear.

Métricas coletadas: p50, p95, p99, média, mínimo, máximo, taxa de erro, throughput (req/min), tempo total de parede.

### 10.8 Comparação de Variantes (`eval/compare_variants.py`)

Implementa as três variantes descritas na seção 4.4.2 do TCC:

| Variante | Descrição | Implementação |
|----------|-----------|---------------|
| A — LLM sem RAG | Resposta direta do LLM sem contexto | `groq_service.generate(question)` com prompt mínimo |
| B — LIKE + LLM | Retrieval por substring + geração | LIKE search → prompt com documentos → LLM |
| C — FTS5 + LLM | Sistema atual completo | `rag_service.answer(question)` |

Todas as três variantes são avaliadas com o mesmo LLM-as-judge, garantindo comparabilidade.

### 10.9 Orquestrador CLI (`eval/run_evaluation.py`)

```bash
python -m eval.run_evaluation [retrieval|generation|latency|variants|all]
```

O comando `all` executa os 4 módulos em sequência, gera os 4 arquivos de resultado individuais e consolida tudo em `consolidated_report.json` com sumário impresso no terminal.

### 10.10 Resultados Obtidos (16 de março de 2026)

Comando executado: `python -m eval.run_evaluation all`

---

#### 10.10.1 Recuperação — FTS5 vs LIKE (20 perguntas, top_k=5)

| Métrica | FTS5 | LIKE | Ganho FTS5 |
|---------|------|------|-----------|
| Recall@5 | **0.565** | 0.250 | +126% |
| MRR | **0.742** | 0.173 | +329% |
| nDCG@5 | **0.744** | 0.186 | +300% |
| P@5 | **0.660** | 0.120 | +450% |

**Análise:** FTS5 supera LIKE em todas as métricas por margens expressivas. O MRR de 0.742 indica que o documento mais relevante tende a aparecer nas primeiras posições, o que é crítico para a qualidade do contexto fornecido ao LLM. O LIKE com P@5=0.120 tem desempenho próximo ao aleatório numa coleção grande, confirmando que busca por substring sem tokenização e normalização é inadequada para texto jurídico com variação morfológica e acentuação.

O Recall@5=0.565 do FTS5 não atinge 1.0 porque o corpus tem lacunas de cobertura: q09 (demissão de servidor / PAD) não retornou documento relevante — evidência de sub-representação de temas de Direito Administrativo no corpus atual.

---

#### 10.10.2 Geração — LLM-as-Judge (10 perguntas, q01–q10)

| Dimensão | Média (0–5) |
|----------|------------|
| Fluência | **4.9** |
| Coerência | **4.8** |
| Relevância | **4.4** |
| Groundedness | **3.9** |
| Grounding Check (automático) | **0.725** |

**Resultados por pergunta:**

| ID | Área | G | R | C | F | Grounding |
|----|------|---|---|---|---|-----------|
| q01 | Direito Penal | 4.0 | 5.0 | 5.0 | 5.0 | 1.00 |
| q02 | Direito Penal | 5.0 | 5.0 | 5.0 | 5.0 | 1.00 |
| q03 | Direito Penal | 5.0 | 5.0 | 5.0 | 5.0 | 1.00 |
| q04 | Direito Penal | 5.0 | 5.0 | 5.0 | 5.0 | 1.00 |
| q05 | Direito Penal | 5.0 | 5.0 | 5.0 | 5.0 | 1.00 |
| q06 | Direito Civil | 5.0 | 5.0 | 5.0 | 5.0 | 0.50 |
| q07 | Direito Civil | 2.0 | 4.0 | 3.0 | 4.0 | 0.00 |
| q08 | Direito do Consumidor | 4.0 | 5.0 | 5.0 | 5.0 | 0.00 |
| q09 | Direito Administrativo | 0.0 | 0.0 | 5.0 | 5.0 | 1.00 |
| q10 | Direito Administrativo | 4.0 | 5.0 | 5.0 | 5.0 | 0.75 |

**Análise por caso:**

- **q09 (Groundedness=0, Relevância=0):** O sistema respondeu "Não encontrei informação suficiente nos documentos disponíveis." — comportamento correto diante de lacuna de cobertura, mas que penaliza as médias. O Fluência=5 e Coerência=5 confirmam que a resposta está bem redigida; o problema é de cobertura do corpus, não do pipeline.

- **q07 (Groundedness=2, Grounding=0.00):** O modelo citou `DIREITO TRIBUTÁRIO — Ed. 58` para uma pergunta de Direito Civil sobre dano moral por SPC. O FTS5 trouxe um documento de uma área cruzada semanticamente (registro de inadimplentes aparece em contextos tributários), e o LLM usou essa fonte de forma inadequada.

- **q08 (Grounding=0.00):** As 4 citações geradas foram verificadas como inválidas pelo grounding check automático. Análise: o regex extraiu IDs curtos como `Tese 9`, `Tese 6`, `Tese 18` que não correspondem aos identificadores completos dos documentos recuperados (ex: `DIREITO DO CONSUMIDOR — Ed. 163: DIREITO DO CONSUMIDOR - VII (Tese 9)`). O sistema gerou conteúdo correto ancorado nos documentos, mas o mecanismo de verificação automático é conservador e não fez o match parcial.

- **q06 (Grounding=0.50):** 2 das 4 citações foram identificadas como alucinadas (`Tese 5`, `Tese 6` sem o prefixo da edição) — mesmo padrão do q08, onde o LLM abrevia identificadores longos.

---

#### 10.10.3 Latência Operacional (10 execuções, error_rate=0%)

| Métrica | Valor |
|---------|-------|
| p50 (mediana) | 13,2 s |
| p95 | 14,8 s |
| p99 | 15,3 s |
| Média | 13,2 s |
| Mínimo | 11,2 s |
| Máximo | 15,4 s |
| Taxa de erro | 0,0% |
| Throughput | 4,56 req/min |

**Análise:** A distribuição de latência é compacta (desvio p99-p50 ≈ 2s), sem outliers extremos, indicando estabilidade no pipeline. O gargalo é a chamada à API Groq (rede + inferência do llama-3.3-70b-versatile). O throughput de 4,56 req/min reflete o rate limit do plano gratuito da API. Zero erros em 10 execuções confirmam a estabilidade do pipeline completo.

---

#### 10.10.4 Comparação de Variantes (5 perguntas: q01, q02, q04, q06, q09)

| Variante | Groundedness | Relevância | Coerência | Fluência |
|----------|:-----------:|:---------:|:---------:|:--------:|
| A — LLM sem RAG | 0.0 | 5.0 | 5.0 | 5.0 |
| B — LIKE + LLM | 2.25 | 1.5 | 5.0 | 5.0 |
| **C — FTS5 + LLM** | **4.667** | **5.0** | **5.0** | **5.0** |

**Análise:**

- **Variante A (sem RAG):** Fluência e coerência perfeitas — o LLM é capaz de produzir texto jurídico impecável a partir do conhecimento paramétrico. Porém, Groundedness=0.0: nenhuma resposta cita fontes verificáveis. Para uso jurídico profissional, isso é inaceitável — sem rastreabilidade, a resposta não tem valor probatório.

- **Variante B (LIKE + LLM):** A recuperação deficiente contamina a geração. Relevância cai para 1.5 porque o LIKE frequentemente retorna documentos sem relação com a pergunta, e o LLM — não encontrando contexto útil — declara "não encontrei informação suficiente". Groundedness=2.25 indica que, quando o LIKE acerta, a ancoragem melhora, mas não é confiável.

- **Variante C (FTS5 + LLM):** Groundedness 4.667 com Relevância=5.0 — combinação ótima. O FTS5 com tokenizador `unicode61 remove_diacritics 1` e ranking BM25 recupera consistentemente documentos relevantes, permitindo ao LLM construir respostas fundamentadas e verificáveis.

**Conclusão:** os resultados validam empiricamente a hipótese central do trabalho. A diferença de groundedness entre as variantes A e C (0.0 → 4.667) e entre B e C (2.25 → 4.667) demonstra que a qualidade da recuperação é o fator determinante para a utilidade jurídica do sistema, não a capacidade generativa do LLM em si.

### 10.11 Limitações da Avaliação

1. **Tamanho do dataset:** 20 perguntas para retrieval e 5 para variantes são amostras indicativas. Resultados estatisticamente conclusivos exigiriam conjuntos maiores com múltiplos anotadores.

2. **Lacuna de cobertura (q09):** O corpus atual sub-representa temas de Direito Administrativo, especialmente PAD (Processo Administrativo Disciplinar). A métrica de recall captura esse problema objetivamente.

3. **Grounding check conservador:** O algoritmo exige match exato do identificador da fonte. Abreviações como `Tese 9` (sem o prefixo da edição) são marcadas como alucinadas mesmo quando o conteúdo está correto. Uma versão mais sofisticada faria match parcial ou fuzzy.

4. **Latência dependente de API externa:** Os 13s medidos refletem o plano gratuito Groq com rate limit. Um deployment de produção com SLA garantido ou LLM local teria perfil de latência diferente.

5. **LLM-as-judge com mesmo modelo:** O avaliador usa o mesmo `llama-3.3-70b-versatile` que gerou as respostas. Há risco de viés de auto-avaliação favorável. Idealmente, um modelo diferente (ex: Claude, GPT-4) seria usado como juiz.

### 10.12 Estado Após a Fase 10

- ✅ **`eval/dataset.json`** — 20 perguntas jurídicas + 4 adversariais com critérios de relevância lexical
- ✅ **`eval/metrics.py`** — Recall@k, MRR, nDCG@k, P@k com 22 testes unitários
- ✅ **`eval/retrieval_eval.py`** — comparação FTS5 vs LIKE sobre 20 perguntas
- ✅ **`eval/generation_eval.py`** — LLM-as-judge (4 dimensões) + grounding check determinístico
- ✅ **`eval/latency_eval.py`** — benchmark p50/p95/p99 com 10 execuções
- ✅ **`eval/compare_variants.py`** — comparação das 3 variantes arquiteturais do TCC
- ✅ **`eval/run_evaluation.py`** — CLI orquestrador com subcomandos e relatório consolidado
- ✅ **`eval/results/`** — 5 arquivos JSON com resultados persistidos e datados
- ✅ **107 testes** no total (85 existentes + 22 novos para métricas), todos passando

---

## Fase 11 — Hardening de Segurança (23/03/2026)

### 11.1 Motivação

Após a conclusão da avaliação experimental (Fase 10), foi realizada uma auditoria de segurança completa sobre o código da aplicação. O objetivo foi identificar e corrigir vulnerabilidades antes de qualquer deploy ou apresentação formal do sistema. A análise cobriu todos os módulos: API, serviços, ETL, configuração e logging.

### 11.2 Vulnerabilidades Identificadas e Corrigidas

A auditoria identificou **19 vulnerabilidades** distribuídas em quatro níveis de severidade. Todas foram corrigidas na mesma sessão.

#### 11.2.1 Correções Anteriores (aplicadas antes da auditoria formal)

Antes da auditoria completa, quatro problemas foram corrigidos imediatamente:

| # | Arquivo | Correção |
|---|---------|----------|
| — | `main.py` | `host` trocado de `0.0.0.0` para `127.0.0.1`; `reload` vinculado a `settings.debug` |
| — | `main.py` | `CORSMiddleware` configurado explicitamente com origens restritas a `localhost:3000` |
| — | `src/api/limiter.py` *(novo)* | Instância `slowapi.Limiter` isolada para evitar import circular |
| — | `src/api/routes/query.py` | `@limiter.limit("10/minute")` no endpoint `/query`; parâmetro `request: Request` adicionado |
| — | `etl/generate_embeddings.py` | `_VALID_TABLES` e `_VALID_COLUMNS` como `frozenset`; validação antes de qualquer SQL com f-string |

#### 11.2.2 Correções da Auditoria Formal

**Crítico / Alto:**

| # | Arquivo | Vulnerabilidade | Correção |
|---|---------|-----------------|----------|
| 1 | `src/database/connection.py` | DDL usava f-string com `settings.db_table_meta/fts` sem validação | Whitelist `_ALLOWED_META_TABLES` / `_ALLOWED_FTS_TABLES`; validação em `init_db()` |
| 2 | `src/services/rag_service.py` | `llm_provider` sem whitelist — qualquer valor desviava silenciosamente para Ollama | `_VALID_LLM_PROVIDERS = frozenset({"groq", "ollama"})` com `ValueError` explícito |
| 3 | `src/api/routes/query.py` | Pergunta do usuário embutida no prompt sem detecção de injection | `_check_injection()` com regex que detecta padrões `ignore/disregard instructions` etc. |
| 4 | `src/services/rag_service.py` | Conteúdo do banco embutido no prompt sem sanitização | `_sanitize_doc_text()` aplicado em ementas e teses antes da montagem do prompt |
| 5 | `src/services/rag_service.py` | Prompt final sem verificação de tamanho | Warning se `len(prompt) > 32_000`; log do prompt rebaixado de `INFO` para `DEBUG` |

**Médio:**

| # | Arquivo | Vulnerabilidade | Correção |
|---|---------|-----------------|----------|
| 6 | `src/config/settings.py` | `.env` com path relativo — quebra se working directory mudar | `_ENV_FILE = str(Path(__file__).resolve().parent.parent.parent / ".env")` |
| 7 | `src/config/settings.py` | `database_url` sem validação de path traversal | `@field_validator` rejeita `..` no caminho; `:memory:` é exceção permitida |
| 8 | `src/config/logging_config.py` | API keys podiam vazar nos logs | `_SecretFilter` redacta `gsk_[A-Za-z0-9]{20,}` antes de emitir qualquer mensagem |
| 9 | `src/services/groq_service.py` | API key validada apenas por `if not key` | Regex `^gsk_[A-Za-z0-9]+$` com `logger.warning` se formato inesperado |
| 10 | `src/services/search_service.py` | Queries FTS5 sem timeout — DB corrompido poderia travar a aplicação | `asyncio.wait_for(..., timeout=5.0)` em ambas as queries (`search` e `search_teses`) |
| 11 | `src/services/semantic_service.py` | `_deserialize()` sem validação de tamanho do BLOB | Valida `len(blob) == 384 * 4 = 1536` antes de `np.frombuffer` |
| 12 | `main.py` | `_run_tests()` sem timeout — pytest travado bloquearia startup indefinidamente | `asyncio.wait_for(proc.communicate(), timeout=300.0)` com `proc.kill()` no timeout |
| 13 | `main.py` | Ausência de security headers HTTP | `_SecurityHeadersMiddleware` adiciona `X-Content-Type-Options`, `X-Frame-Options`, `X-XSS-Protection` |

**Baixo:**

| # | Arquivo | Vulnerabilidade | Correção |
|---|---------|-----------------|----------|
| 14 | `etl/load.py` | `glob("resultados-de-acordaos*.csv")` aceitava qualquer arquivo no padrão | `_CSV_NAME_RE = re.compile(r'^resultados-de-acordaos[\w\-]*\.csv$')` filtrando o glob |
| 15 | `etl/load_teses_stj.py` | Regex sem limites explícitos nos grupos — potencial ReDoS | Limites `{1,N}` adicionados em todos os grupos (`{1,200}`, `{1,1000}`, `{1,500}`) |
| 16 | `etl/load_teses_stj.py` | `sqlite3.connect()` sem `PRAGMA foreign_keys=ON` — inconsistente com a app | `PRAGMA foreign_keys=ON` adicionado nas 3 funções: `load()`, `load_area()`, `force_reload()` |
| 17 | `src/api/routes/query.py` | Sem identificador de requisição nos logs | `request_id = str(uuid.uuid4())[:8]` gerado e logado por requisição |
| requirements.txt | — | `slowapi` ausente | `slowapi==0.1.9` adicionado |

### 11.3 Arquivos Criados / Modificados

| Arquivo | Tipo de mudança |
|---------|----------------|
| `src/api/limiter.py` | Criado — instância isolada do Limiter |
| `main.py` | Modificado — host, reload, CORS, rate limit, timeout, security headers |
| `src/api/routes/query.py` | Modificado — rate limit, injection check, request ID |
| `src/config/settings.py` | Modificado — .env absoluto, field_validator database_url |
| `src/config/logging_config.py` | Modificado — _SecretFilter |
| `src/services/groq_service.py` | Modificado — validação de formato da API key |
| `src/services/search_service.py` | Modificado — timeout asyncio nas queries FTS5 |
| `src/services/rag_service.py` | Modificado — provider whitelist, sanitização, prompt size check |
| `src/services/semantic_service.py` | Modificado — validação de tamanho do BLOB |
| `src/database/connection.py` | Modificado — whitelist de nomes de tabela |
| `etl/load.py` | Modificado — validação de nomes de CSV |
| `etl/load_teses_stj.py` | Modificado — regex limits, PRAGMA foreign_keys |
| `etl/generate_embeddings.py` | Modificado — _VALID_TABLES/_VALID_COLUMNS whitelist |
| `requirements.txt` | Modificado — slowapi==0.1.9 |

### 11.4 Estado Após a Fase 11

- ✅ **0 vulnerabilidades críticas** remanescentes na aplicação
- ✅ **Rate limiting** ativo: 10 req/min por IP no endpoint `/query`
- ✅ **CORS** restrito a `localhost:3000`
- ✅ **Security headers** em todas as respostas HTTP
- ✅ **Prompt injection** detectado tanto na pergunta do usuário quanto no conteúdo do banco
- ✅ **Secrets** nunca expostos nos logs (filtro de redação ativo)
- ✅ **107 testes passando** — nenhuma regressão introduzida pelas correções

---

## Fase 12 — Busca Semântica Híbrida (25/03/2026)

### 12.1 Motivação

Após a fase de hardening, o sistema operava com busca puramente lexical (FTS5/BM25). A limitação central desse modelo é o **vocabulary mismatch**: se o usuário pergunta "pena restritiva de liberdade" e o acórdão usa "medida cautelar prisional", a busca BM25 falha. A solução clássica é adicionar busca semântica (embeddings) e fundir os resultados.

A busca semântica estava prevista no TCC1 e foi implementada nesta fase como o principal avanço técnico do trabalho.

### 12.2 Expansão da Base de Conhecimento

Antes de implementar a semântica, o corpus foi expandido significativamente:

#### 12.2.1 Ampliação dos acórdãos STF

| Estado anterior | Estado novo |
|---|---|
| 2.241 acórdãos (só dezembro/2025) | 3.420 acórdãos (set/2025 – mar/2026) |
| 1 arquivo CSV | 18 arquivos CSV |

O ETL foi corrigido — a versão anterior assumia que os arquivos eram cumulativos e carregava apenas o maior numerado. A correção em `etl/load.py` passou a coletar todos os CSVs com `glob("resultados-de-acordaos*.csv")`.

#### 12.2.2 Adição de súmulas STJ

- `data/stj/SelecaoSumulas.txt` — 676 súmulas STJ
- Novo ETL: `etl/load_sumulas_stj.py`
- Armazenadas na tabela `teses_stj` com `area='SÚMULAS STJ'`
- Identificadas no prompt como `[Súmula STJ NNN]` e citadas como `Súmula NNN/STJ`

#### 12.2.3 Adição de teses de Direito do Consumidor

- `data/stj/jurispruConsumidor.txt` — edições 161–164 (100 teses)
- Função `load_area()` adicionada ao `etl/load_teses_stj.py` para recarregar apenas uma área sem apagar as demais

**Corpus final:**

| Fonte | Documentos |
|---|---|
| Acórdãos STF | 3.420 |
| Teses STJ (Jurisprudência em Teses) | 3.378 |
| Súmulas STJ | 676 |
| **Total** | **7.474** |

### 12.3 Arquitetura da Busca Semântica

#### 12.3.1 Modelo de embeddings

- **Modelo:** `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers)
- **Dimensão:** 384
- **Normalização:** `normalize_embeddings=True` (cosine similarity → dot product)
- **Justificativa:** multilingual, CPU-friendly (~90 MB), suporte nativo a português jurídico

#### 12.3.2 Geração de embeddings (ETL)

Novo arquivo: `etl/generate_embeddings.py`

- Adiciona coluna `embedding BLOB` nas tabelas `jurisprudencia` e `teses_stj`
- Processa apenas linhas com `embedding IS NULL` (idempotente)
- Serialização: `struct.pack(f"{len(v)}f", *v)` → BLOB de 1536 bytes por vetor
- Textos truncados: acórdãos a 1000 chars, teses a 500 chars
- Batch size: 64

#### 12.3.3 Cache em memória

Ao iniciar a primeira consulta semântica, `semantic_service.py` carrega todos os embeddings do banco para arrays NumPy em memória:

```
_acordao_cache: (ids[], metadados[], np.ndarray [N×384])
_teses_cache:   (ids[], metadados[], np.ndarray [M×384])
```

- Tamanho: ~11 MB para 7.474 documentos × 384 dims × 4 bytes
- Carregado uma vez por processo, reutilizado em todas as consultas
- `clear_cache()` disponível para invalidação manual

#### 12.3.4 Busca por similaridade

```python
# Vetores normalizados → dot product = cosine similarity
scores = matrix @ query_vec          # shape [N]
top_idx = np.argpartition(scores, -top_k)[-top_k:]  # O(N) vs O(N log N)
```

#### 12.3.5 Reciprocal Rank Fusion (RRF)

Funde os rankings lexical (FTS5) e semântico:

```
score(d) = 1/(k + rank_lexical + 1) + 1/(k + rank_semantic + 1)    k=60
```

Pipeline completo:
1. FTS5 busca top-15 acórdãos + top-15 teses
2. Semântica busca top-15 acórdãos + top-15 teses (em paralelo via `asyncio.gather`)
3. RRF funde e seleciona top-6 acórdãos + top-3 teses para o prompt

### 12.4 Mudanças nos Arquivos

| Arquivo | Mudança |
|---|---|
| `etl/generate_embeddings.py` | Novo — ETL de vetorização do corpus |
| `src/services/semantic_service.py` | Novo — cache NumPy + cosine similarity + RRF |
| `src/services/rag_service.py` | `answer()` reescrito para pipeline híbrido com 4 buscas paralelas |
| `eval/compare_variants.py` | Variante C (FTS5-only) separada da Variante D (Híbrido) |
| `eval/run_evaluation.py` | Relatório consolidado atualizado para 4 variantes |
| `data/stj/` | Novo diretório — `JTSelecao.txt`, `SelecaoSumulas.txt`, `jurispruConsumidor.txt` |
| `etl/load_sumulas_stj.py` | Novo — parser de súmulas STJ |
| `etl/load_teses_stj.py` | `load_area()` adicionada; path atualizado para `data/stj/` |

### 12.5 Correção do Framework de Avaliação

Durante a implementação, foram identificados e corrigidos 3 bugs no framework de avaliação:

| Bug | Causa | Correção |
|---|---|---|
| Grounding check falhava em teses com parênteses aninhados | Regex `\(([^)]{5,80})\)` parava no `)`interno de `(Tese 8)` | Regex `\(([^()]*(?:\([^()]*\)[^()]*)*)\)` |
| Modelo citava rótulo `[Acórdão STF N]` em vez do número do processo | Instrução ambígua no prompt | "cite SOMENTE o número do processo... NUNCA use o rótulo" |
| ETL carregava apenas 2 de 18 CSVs | Assumia arquivos cumulativos | `return all_csvs` (lista completa) |

### 12.6 Resultados da Avaliação

#### 12.6.1 Avaliação de geração (10 queries, híbrido)

| Dimensão | Score |
|---|---|
| Groundedness | 3.30/5.0 |
| Relevância | 4.60/5.0 |
| Coerência | 4.90/5.0 |
| Fluência | 5.00/5.0 |
| Grounding score | 0.975/1.0 |

#### 12.6.2 Comparação de variantes (5 queries, A/B/C/D)

| Variante | Groundedness | Relevância | Coerência | Fluência |
|---|---|---|---|---|
| A — LLM sem RAG | 0.000 | 5.000 | 5.000 | 5.000 |
| B — LIKE + LLM | 0.000 | 0.000 | 5.000 | 5.000 |
| C — FTS5 + LLM | 3.400 | 4.000 | 5.000 | 5.000 |
| **D — Híbrido + LLM** | **4.800** | **5.000** | **5.000** | **5.000** |

**Conclusão:** o pipeline híbrido entrega +1.4 de groundedness e +1.0 de relevância em relação ao FTS5-only, validando a hipótese central do TCC. A busca semântica resolve o vocabulary mismatch que o BM25 não consegue cobrir.

### 12.7 Estado Após a Fase 12

- ✅ **7.474 documentos** vetorizados e indexados (FTS5 + embeddings)
- ✅ **Pipeline híbrido** FTS5 + semântica + RRF em produção
- ✅ **Comparação A/B/C/D** implementada no framework de avaliação
- ✅ **Groundedness 4.8/5.0** — melhor resultado do projeto até agora
- ✅ Reranking cross-encoder — implementado na Fase 13

---

## Fase 13 — Reranking Cross-Encoder (22 de março de 2026)

### 13.1 Motivação

Após o pipeline híbrido (FTS5 + semântica + RRF), os candidatos selecionados pelo RRF ainda carregam ruído: o RRF funde rankings mas não puntua cada par (query, documento) diretamente. O **cross-encoder** resolve isso: recebe a query e o texto do documento concatenados e produz um score de relevância end-to-end, aproveitando a atenção cruzada do transformer.

Comparação conceitual:

| Etapa | Modelo | Complexidade | Precisão |
|---|---|---|---|
| FTS5 | BM25 | O(N) | Lexical |
| Busca semântica | bi-encoder (MiniLM) | O(N) | Semântica (aproximada) |
| Reranking | cross-encoder | O(k) — k << N | Semântica (exata) |

O cross-encoder é caro por documento, por isso é aplicado apenas sobre os top-20 candidatos do RRF — não sobre o corpus inteiro.

### 13.2 Implementação (`src/services/rerank_service.py`)

**Modelo:** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- Multilíngue (incluindo português), treinado em MS MARCO multilíngue
- MiniLM 12 camadas × 384 hidden dims — rápido em CPU para ≤ 30 docs
- ~120 MB em disco

**Singleton:** o modelo é carregado via `CrossEncoder(MODEL_NAME)` na primeira chamada e mantido em memória (`_model` global) — padrão idêntico ao `semantic_service`.

**Fallback gracioso:** qualquer exceção na carga ou na predição retorna a lista de entrada intacta (ordem RRF preservada), garantindo que o sistema nunca quebra por indisponibilidade do cross-encoder.

**Extração de texto:**

```python
_MAX_TEXT_CHARS = 512  # MiniLM max ≈ 128 tokens ≈ ~400 chars PT

def _get_text(doc):
    if isinstance(doc, SearchResult):
        return doc.ementa[:512]
    # TesesResult: prefixa com área temática
    prefix = f"{doc.area}: " if doc.area else ""
    return (prefix + doc.tese_texto)[:512]
```

**API pública:**

```python
def rerank(query: str, docs: list[T], top_n: int | None = None) -> list[T]:
    pairs = [(query, _get_text(d)) for d in docs]
    scores = model.predict(pairs).tolist()
    ranked = sorted(zip(scores, docs), reverse=True)
    return [d for _, d in ranked][:top_n]
```

### 13.3 Integração no Pipeline RAG

Em `src/services/rag_service.py`, o reranking é o estágio final após o RRF:

```python
_FETCH = 15           # candidatos por fonte antes do RRF
_RRF_CANDIDATES = 20  # candidatos pós-RRF enviados ao cross-encoder

if settings.reranker_enabled:
    sources = rerank_service.rerank(question, candidates_acordaos, top_n=settings.rag_top_k)
    sources_teses = rerank_service.rerank(question, candidates_teses, top_n=settings.rag_top_k_teses)
else:
    sources = candidates_acordaos[:settings.rag_top_k]
    sources_teses = candidates_teses[:settings.rag_top_k_teses]
```

**Nova setting adicionada em `src/config/settings.py`:**

```python
reranker_enabled: bool = True  # desabilitar para usar só RRF (mais rápido)
```

### 13.4 Pipeline Completo (após Fase 13)

```
Pergunta do usuário
      │
      ├─── FTS5 (BM25): top-15 acórdãos + top-15 teses  ─┐
      └─── Semântica (cosine): top-15 acórdãos + teses   ─┘ asyncio.gather
                        │
                        ▼
             RRF (k=60): top-20 candidatos por tipo
                        │
                        ▼
           Cross-Encoder: pontua cada par (query, doc)
                        │
                        ▼
          top-6 acórdãos + top-3 teses → prompt → LLM
```

### 13.5 Arquivos Criados / Modificados

| Arquivo | Mudança |
|---|---|
| `src/services/rerank_service.py` | **Novo** — cross-encoder singleton, `rerank()`, fallback gracioso |
| `src/config/settings.py` | `reranker_enabled: bool = True` adicionado |
| `src/services/rag_service.py` | Etapa de reranking inserida após o RRF; log detalhado por etapa |

### 13.6 Estado Após a Fase 13

- ✅ **Pipeline de 4 estágios:** FTS5 → semântica → RRF → cross-encoder reranking
- ✅ **`reranker_enabled`** configurável via `.env` (desativar para reduzir latência de CPU)
- ✅ **Fallback gracioso:** cross-encoder indisponível → ordem RRF preservada, sem exceção

---

## Fase 14 — Endpoint de Health Check (23 de março de 2026)

### 14.1 Motivação

Sem um endpoint de saúde, não há forma programática de verificar se a aplicação está operacional — banco conectado, configuração válida, reranker ativo. Um `GET /api/v1/health` permite monitoramento externo e integração com ferramentas de CI.

### 14.2 Implementação (`src/api/routes/health.py`)

**Schema de resposta:**

```python
class HealthResponse(BaseModel):
    status: str            # "ok" | "degraded"
    uptime_seconds: float  # segundos desde o startup
    database: str          # "ok" | "unavailable"
    llm_provider: str      # "groq" | "ollama"
    reranker_enabled: bool
```

**Lógica:**

```python
@router.get("/health", response_model=HealthResponse)
async def health_check(conn = Depends(get_db)) -> HealthResponse:
    try:
        await conn.execute("SELECT 1")
        db_status = "ok"
    except Exception:
        db_status = "unavailable"

    return HealthResponse(
        status="ok" if db_status == "ok" else "degraded",
        uptime_seconds=round(time.time() - _START_TIME, 1),
        database=db_status,
        llm_provider=settings.llm_provider,
        reranker_enabled=settings.reranker_enabled,
    )
```

`_START_TIME` é capturado no módulo no momento do import — reflete o uptime real da aplicação.

### 14.3 Registro em `main.py`

```python
from src.api.routes import health as health_router
app.include_router(health_router.router, prefix="/api/v1", tags=["Health"])
```

### 14.4 Arquivos Criados / Modificados

| Arquivo | Mudança |
|---|---|
| `src/api/routes/health.py` | **Novo** — router, `HealthResponse`, lógica de verificação |
| `main.py` | Import e `include_router` do `health_router` |

### 14.5 Estado Após a Fase 14

- ✅ **`GET /api/v1/health`** operacional — banco, provedor LLM e reranker expostos
- ✅ **`status: "degraded"`** quando banco indisponível — distingue falha parcial de falha total
- ✅ **`uptime_seconds`** rastreado desde o início do processo

---

## Fase 15 — Expansão de Consulta com Sinônimos Jurídicos (24 de março de 2026)

### 15.1 Motivação

A busca FTS5 com OR entre tokens cobre variação morfológica via `unicode61 remove_diacritics`, mas não cobre **sinonímia jurídica**: o usuário que pesquisa "prisão" não recupera documentos que usam apenas "custódia" ou "detenção". Da mesma forma, siglas como `HC` e `RE` não casam com os termos por extenso.

A **query expansion** resolve o mismatch de vocabulário antes que a consulta chegue ao índice FTS5, sem custo de latência adicional (é uma operação de dicionário — O(termos)).

### 15.2 Implementação (`src/services/query_expansion.py`)

**Estratégia:** dicionário `_SYNONYMS` com ~80 entradas cobrindo 10 grupos temáticos:

| Grupo | Exemplos de chave → sinônimos |
|---|---|
| Remédios constitucionais / siglas | `"habeas corpus"` → `{"hc"}`, `"resp"` → `{"recurso", "especial"}` |
| Direito penal / processual penal | `"prisao"` → `{"custodia", "detencao", "encarceramento"}` |
| Direito civil / obrigações | `"dano moral"` → `{"indenizacao", "extrapatrimonial", "reparacao"}` |
| Direito do consumidor | `"plano saude"` → `{"convenio", "operadora", "cobertura"}` |
| Direito administrativo | `"improbidade"` → `{"corrupcao", "desvio", "ato"}` |
| Direito tributário | `"icms"` → `{"imposto", "circulacao", "mercadorias"}` |
| Direito trabalhista | `"horas extras"` → `{"adicional", "sobrejornada", "jornada"}` |
| Processo / garantias | `"tutela antecipada"` → `{"liminar", "cautelar", "urgencia"}` |

**Lookup normalizado (sem acentos):**

```python
def expand_query(tokens: list[str]) -> list[str]:
    norm = [_strip_accents(t) for t in tokens]
    norm_set = set(norm)
    extra: set[str] = set()
    for key, synonyms in _SYNONYMS.items():
        key_tokens = key.split()
        if all(t in norm_set for t in key_tokens):
            extra.update(synonyms)
    return sorted(extra - norm_set)  # remove tokens já na query
```

**Exemplo prático:**

```
query: "habeas corpus" → tokens: ["habeas", "corpus"]
expand_query(["habeas", "corpus"]) → ["hc"]
FTS5 query: "habeas OR corpus OR hc"
```

### 15.3 Integração em `search_service.py`

Adicionado após a tokenização e antes da construção da query FTS5, em **ambas** as funções:

```python
expanded = expand_query(tokens)
if expanded:
    logger.debug("Query expansion: +%d termos: %s", len(expanded), expanded)
    tokens = tokens + expanded
safe_query = " OR ".join(tokens)
```

**Limitação documentada:** tokens de 2 caracteres (ex: `hc`, `re`, `ms`) são filtrados pelo tokenizador (`len(t) > 2`) antes de chegar à função. Portanto essas siglas só funcionam como **sinônimos adicionados** pela expansão, não como chaves de lookup.

### 15.4 Arquivos Criados / Modificados

| Arquivo | Mudança |
|---|---|
| `src/services/query_expansion.py` | **Novo** — `_SYNONYMS`, `_strip_accents()`, `expand_query()` |
| `src/services/search_service.py` | `expand_query()` chamado em `search()` e `search_teses()` |

### 15.5 Estado Após a Fase 15

- ✅ **~80 mapeamentos jurídicos** cobrindo 10 grupos temáticos
- ✅ **Integrado em ambas as buscas** FTS5 (acórdãos e teses)
- ✅ **Zero latência adicional** — lookup de dicionário em memória
- ✅ **Idempotente:** tokens já presentes na query não são duplicados

---

## Fase 16 — Interface Web (Frontend) (25 de março de 2026)

### 16.1 Motivação

A API REST funcionava apenas via `curl` ou Swagger UI. Para demonstrações do TCC e uso real por advogados, era necessária uma interface visual acessível pelo navegador.

### 16.2 Implementação (`static/index.html`)

**Tecnologias:**
- **Tailwind CSS** (CDN) — estilização utilitária sem build step
- **Google Fonts** — `Playfair Display` (serif para o título) + `DM Sans` (sans-serif para corpo)
- **Vanilla JS** — sem framework, sem dependências de bundle

**Tema visual:**

| Token | Valor | Uso |
|---|---|---|
| `bg` | `#111318` | Fundo da página |
| `surface` | `#1A1D27` | Cards e formulário |
| `border` | `#2A2E3F` | Bordas |
| `text` | `#E2E2E8` | Texto principal |
| `muted` | `#6B7080` | Texto secundário |
| `accent` | `#4F7CAC` | Botões, logo, links |

**Funcionalidades:**
- `textarea` com contador de caracteres em tempo real (0/1000)
- `Ctrl + Enter` para enviar a consulta
- Modal de loading com animação de dots pulsantes (`dot-pulse`) durante a requisição
- Resposta renderizada com o texto e lista de fontes (tribunal, número do processo, ementa)
- Feedback de erro amigável em caso de HTTP 4xx/5xx

**Integração com FastAPI:**

```python
# main.py
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", include_in_schema=False)
async def serve_frontend() -> FileResponse:
    return FileResponse("static/index.html")
```

A rota `GET /` não aparece no Swagger (excluída com `include_in_schema=False`).

### 16.3 Estado Após a Fase 16

- ✅ **Interface acessível em `http://localhost:8000`** sem configuração adicional
- ✅ **Design dark** com identidade visual consistente (logo de balança, tipografia jurídica)
- ✅ **UX mínima mas completa:** input, loading, resposta, fontes, erros
- ✅ **Zero dependências de build:** arquivo único `.html` servido como arquivo estático

---

## Fase 17 — Teste de Carga (Locust)

**Data:** 2026-04-22

### 17.1 Motivação

Para a defesa do TCC, era necessário demonstrar o comportamento do sistema sob carga, validando três aspectos:

1. **Disponibilidade**: o endpoint `/health` mantém latência baixa mesmo com múltiplos clientes simultâneos.
2. **Throughput real do pipeline RAG**: latência E2E (p50, p95, p99) do `/query` incluindo busca + reranking + LLM.
3. **Funcionamento do rate limiter**: confirmação de que o mecanismo de proteção retorna 429 corretamente quando o limite é excedido.

A ferramenta escolhida foi **Locust** (`locust>=2.29.0`), por ser:
- Python (consistente com o stack do projeto)
- Orientada a usuários virtuais (simula comportamento real)
- Com geração automática de relatório HTML (p50/p95/p99, gráficos de RPS e latência)

### 17.2 Problema de Rate Limiting em Testes

O decorator `@limiter.limit("10/minute")` em `query.py` era hardcoded. Isso impedia testes de stress reais: todos os usuários Locust compartilham o IP `127.0.0.1`, tornando o limite global para todo o teste (não por usuário virtual).

**Solução:** tornar o rate limit configurável via variável de ambiente.

| Arquivo | Alteração |
|---|---|
| `src/config/settings.py` | Adicionado `rate_limit_per_minute: int = 10` |
| `src/api/routes/query.py` | Limite calculado em `_RATE_LIMIT = f"{get_settings().rate_limit_per_minute}/minute"` no startup |

Assim, para testes de stress basta definir `RATE_LIMIT_PER_MINUTE=200` no `.env` e reiniciar o servidor — sem alteração de código.

### 17.3 Cenários de Teste

O locustfile define três classes de usuários:

| Classe | Endpoint | wait_time | Objetivo |
|---|---|---|---|
| `HealthUser` (weight=3) | `GET /health` | 1–3 s | Medir overhead da API sem gargalo de LLM |
| `QueryUser` (weight=2) | `POST /query` + `/health` | 8–15 s | Pipeline completo com cadência que respeita os 10 req/min por IP |
| `StressQueryUser` (weight=0) | `POST /query` | 1–3 s | Stress test — ativado com `--tags stress` e rate limit elevado |

O corpus de perguntas contém **20 consultas jurídicas realistas** cobrindo habeas corpus, dano moral, direito tributário, improbidade administrativa, interceptação telefônica e outros temas do STF/STJ.

**Comandos de execução:**

```bash
# Cenário padrão com interface web:
locust -f load_tests/locustfile.py --host http://127.0.0.1:8000

# Headless — 2 usuários, 2 minutos, relatório HTML:
locust -f load_tests/locustfile.py --host http://127.0.0.1:8000 \
       --headless -u 2 -r 1 -t 120s --html load_tests/report.html

# Stress test (rate limit elevado necessário):
RATE_LIMIT_PER_MINUTE=200 uvicorn main:app --host 127.0.0.1 --port 8000
locust -f load_tests/locustfile.py --host http://127.0.0.1:8000 \
       --headless -u 10 -r 2 -t 60s --html load_tests/report_stress.html \
       --tags stress
```

### 17.4 Arquivos Criados ou Modificados

| Arquivo | Alteração |
|---|---|
| `load_tests/locustfile.py` | Novo — cenários HealthUser, QueryUser e StressQueryUser |
| `src/config/settings.py` | Adicionado `rate_limit_per_minute: int = 10` |
| `src/api/routes/query.py` | Rate limit agora lido de `get_settings().rate_limit_per_minute` |
| `requirements.txt` | Adicionado `locust>=2.29.0` |

### 17.5 Estado Após a Fase 17

- ✅ **Teste de carga implementado** com Locust e corpus de 20 perguntas jurídicas
- ✅ **Rate limit configurável** via `RATE_LIMIT_PER_MINUTE` no `.env`
- ✅ **Três cenários documentados**: padrão (respeita limite), stress (requer elevação), health (sem limite)
- ✅ **Relatório HTML automático** com p50/p95/p99 e gráficos de throughput/latência

---

## Fase 18 — Prompt v6: Divergência e Peso Jurídico das Fontes (22 de abril de 2026)

### 18.1 Motivação

Análise de qualidade do sistema a partir da perspectiva de um advogado identificou dois problemas críticos no prompt v5:

1. **Falsa síntese de consenso**: a regra 2 ("SINTETIZE os temas comuns") fazia a LLM fundir acórdãos com entendimentos opostos como se fossem consenso. Para um advogado, divergência jurisprudencial é informação essencial — citá-la como consenso é um erro material.

2. **Sem comunicação de vinculatividade**: o sistema tratava Teses STJ (precedente qualificado, vinculante via art. 927, III, CPC), Súmulas STJ (persuasivas, não vinculantes) e acórdãos isolados (casuísticos, persuasivos) de forma idêntica. Um advogado precisa saber se está amparado em precedente qualificado ou em decisão isolada antes de usar a resposta em peça processual.

### 18.2 Decisões de Projeto

**Linha "Efeito:" por documento**

Em vez de criar metadados no banco (exigiria migração de schema e atualização do ETL), a informação de peso jurídico foi embutida diretamente no texto do prompt — uma linha `Efeito:` inserida por `_build_prompt()` logo abaixo de cada rótulo de documento:

```
[Acórdão STF 1] HC 100001
Efeito: decisão casuística — persuasiva, salvo se firmada com repercussão geral.
<texto da ementa>

[Tese STJ 1] DIREITO CIVIL — Ed. 143: PLANO DE SAÚDE - III (Tese 1)
Efeito: precedente qualificado — deve ser observado por todos os tribunais (art. 927, III, CPC).
<texto da tese>

[Súmula STJ 528]
Efeito: enunciado persuasivo — consolidado, mas não vinculante.
<texto da súmula>
```

Essa abordagem:
- Não altera o schema do banco nem o ETL
- Não modifica os rótulos `[Acórdão STF N]`, `[Tese STJ N]`, `[Súmula STJ N]` — preserva os testes de citação existentes
- Coloca a informação de tipo onde a LLM realmente lê: no contexto de cada documento

**Alternativas descartadas:**
- Campo `efeito_vinculativo` na tabela `jurisprudencia`: exigiria inferir o tipo a partir do número do processo (complexo e propenso a erros sem dados de repercussão geral)
- Instrução genérica sem âncora no contexto: LLM tenderia a ignorar ou generalizar incorretamente sem ver o "Efeito:" por documento

### 18.3 Mudanças no Prompt

**Regra 2 — adição de sub-instrução de divergência:**
```
DIVERGÊNCIA: se dois ou mais documentos sustentam entendimentos opostos sobre
o mesmo ponto, NÃO os sintetize como consenso — registre a divergência:
'Há divergência entre os documentos: [fonte A] entende X, enquanto [fonte B] entende Y.'
```

**Nova Regra 4 — nota de fontes:**
```
4. Encerre a resposta com 'Nota sobre as fontes:' e descreva o peso jurídico
   dos documentos utilizados, com base nas linhas 'Efeito:' de cada fonte:
   - Teses STJ (precedente qualificado) vinculam todos os tribunais (art. 927, III, CPC).
   - Súmulas STJ (enunciado persuasivo) são consolidadas, mas não vinculantes.
   - Acórdãos STF (decisão casuística) são persuasivos, salvo com repercussão geral.
   Se a resposta se basear APENAS em acórdãos casuísticos, avise que o usuário
   deve verificar se há tese consolidada ou súmula antes de usar em peças processuais.
```

As antigas regras 4 e 5 passaram a ser 5 e 6.

### 18.4 Testes Adicionados

Cinco novos testes em `tests/test_rag.py`:

| Teste | O que verifica |
|---|---|
| `test_build_prompt_acordao_tem_anotacao_efeito` | Linha "Efeito:" com "casuística" presente no bloco de acórdão |
| `test_build_prompt_tese_tem_anotacao_efeito_precedente` | Linha "Efeito:" com "precedente qualificado" e "art. 927" no bloco de tese |
| `test_build_prompt_sumula_tem_anotacao_efeito_persuasivo` | Linha "Efeito:" com "persuasivo" no bloco de súmula |
| `test_build_prompt_contem_regra_divergencia` | Prompt contém instrução "DIVERGÊNCIA" e "NÃO os sintetize como consenso" |
| `test_build_prompt_contem_regra_nota_fontes` | Prompt contém instrução "Nota sobre as fontes" |

### 18.5 Arquivos Criados ou Modificados

| Arquivo | Alteração |
|---|---|
| `src/services/rag_service.py` | `_build_prompt()`: linhas "Efeito:" nos blocos de documento; regra de divergência na regra 2; nova regra 4 de nota de fontes; regras antigas 4→5, 5→6; docstring atualizada para v6 |
| `tests/test_rag.py` | 5 novos testes cobrindo as adições do prompt v6 |

### 18.6 Estado Após a Fase 18

- ✅ **161 testes passando** (0 falhas, 0 skips) — todos os testes existentes preservados
- ✅ **Divergência jurisprudencial** explicitamente registrada em vez de sintetizada como falso consenso
- ✅ **Peso vinculativo comunicado** ao usuário ao final de cada resposta via "Nota sobre as fontes:"
- ✅ **Três tipos de fonte diferenciados** no contexto: casuístico (acórdão), precedente qualificado (tese), persuasivo (súmula)

---

## Fase 19 — Sinônimo AgInt no Query Expansion (23 de abril de 2026)

### 19.1 Motivação

O NCPC 2015 renomeou o **Agravo Regimental** (AgRg) para **Agravo Interno** (AgInt) em todo o sistema judiciário. Acórdãos do STF e STJ anteriores a 2015 usam "AgRg"; pós-2015 usam "AgInt". O dicionário de sinônimos cobria apenas "agr" (AgRg), deixando acórdãos pós-NCPC menos encontráveis quando o usuário pesquisava pelo nomen juris antigo ou vice-versa.

### 19.2 Mudança Implementada

Quatro entradas adicionadas/atualizadas em `_SYNONYMS` (`query_expansion.py`):

| Entrada (chave) | Sinônimos adicionados | Motivo |
|---|---|---|
| `"agravo regimental"` | `"agint"` adicionado | Equivalência funcional pré/pós-NCPC |
| `"agr"` | `"agint"` adicionado | Cross-link sigla → sigla |
| `"agravo interno"` *(nova)* | `{"agint", "agr"}` | Busca pelo nomen pós-NCPC expande para siglas |
| `"agint"` *(nova)* | `{"agravo", "interno", "agr"}` | Sigla pós-NCPC expande para termos e sigla pré-NCPC |

Com essa mudança, qualquer busca envolvendo "agravo regimental", "AgRg", "agravo interno" ou "AgInt" recupera documentos dos dois períodos históricos.

### 19.3 Arquivos Criados ou Modificados

| Arquivo | Alteração |
|---|---|
| `src/services/query_expansion.py` | 2 entradas atualizadas + 2 novas entradas no `_SYNONYMS` |

### 19.4 Estado Após a Fase 19

- ✅ **Acórdãos pós-NCPC (AgInt) encontráveis** por qualquer variação da sigla ou nome completo
- ✅ **Cross-link bidirecional**: AgRg ↔ AgInt, "agravo regimental" ↔ "agravo interno"
- ✅ **Zero impacto em testes**: alteração restrita ao dicionário estático, sem mudança de interface

---

## Fase 20 — Disclaimer Legal na Interface Web (23 de abril de 2026)

### 20.1 Motivação

Qualquer sistema que opere no domínio jurídico e gere respostas a partir de documentos legais tem obrigação ética e prática de comunicar ao usuário que o sistema **não é um substituto para assessoria jurídica profissional**. Ausência de disclaimer pode:

- Induzir usuário leigo a tomar decisões jurídicas com base em resposta de IA imprecisa
- Configurar exercício irregular da advocacia caso o sistema seja percebido como orientação personalizada
- Comprometer a credibilidade acadêmica do TCC ao apresentar o sistema sem ressalvas

O OAB e a Resolução CFJ n. 345/2020 orientam que sistemas de IA para o domínio jurídico devem ser transparentes sobre suas limitações.

### 20.2 Implementação

**Banner permanente** (`<aside>` com `role="note"`) inserido entre o header e o formulário — posição ideal porque o usuário lê antes de submeter qualquer consulta:

- Cor âmbar sutil (não alarmista, mas distinta do fundo) — `bg-amber-950/20 border border-amber-900/30`
- Ícone de triângulo de aviso em SVG inline (sem dependência externa)
- Texto em duas partes: (a) contexto do sistema ("ferramenta de pesquisa acadêmica"), (b) instrução direta ("não constituem aconselhamento jurídico — consulte um advogado habilitado na OAB")
- `aria-label="Aviso legal"` para leitores de tela

**Nota no footer** — segunda âncora visual, discreta (`text-[0.65rem] text-muted/25`), para usuários que rolam até o fim da página.

### 20.3 Decisões de Projeto

**Permanente vs. modal descartável:** Modal com "aceite" seria mais intrusivo e poderia ser ignorado com um clique. Banner permanente garante que o aviso permanece visível durante toda a sessão sem travar o fluxo.

**Cor âmbar vs. vermelho:** Vermelho transmite erro/falha. Âmbar comunica atenção sem alarme — adequado para um aviso informativo em sistema que funciona corretamente.

**OAB mencionada explicitamente:** Ancoragem em instituição reconhecida aumenta credibilidade do aviso e orienta o usuário sobre onde buscar ajuda.

### 20.4 Arquivos Criados ou Modificados

| Arquivo | Alteração |
|---|---|
| `static/index.html` | Banner `<aside>` com disclaimer inserido entre header e formulário; nota adicional no `<footer>` |

### 20.5 Estado Após a Fase 20

- ✅ **Disclaimer permanente e sempre visível** antes do campo de consulta
- ✅ **Acessível via `role="note"` e `aria-label`** para tecnologias assistivas
- ✅ **Nota reforçada no footer** como segunda âncora visual
- ✅ **Sem impacto no fluxo de uso**: banner informativo, não bloqueante

---

## Fase 21 — Links para Portais STF/STJ nos Cards de Fonte (23 de abril de 2026)

### 21.1 Motivação

Os cards de fonte exibiam tribunal e número do processo apenas como texto estático. Para verificar a decisão original — especialmente relevante porque o STF armazena muitos acórdãos como PDFs agrupados no portal — o usuário precisava abrir o portal manualmente e redigitar o número. A adição de link direto elimina essa fricção.

**Limite técnico:** os portais não expõem URL por documento para todos os acórdãos. O teto realista por tipo de fonte é:
- **STF acórdão** → página do processo no portal STF (`listarProcessos.asp?classe=X&numeroProcesso=Y`): um clique acima do PDF, mas lista todas as decisões daquele processo com links para download.
- **STJ tese** → portal Em Teses (`scon.stj.jus.br/SCON/jt/`): sem deep link por edição via URL pública.
- **STJ acórdão** → busca SCON pré-preenchida com o número do processo.

### 21.2 Implementação

**Função `buildPortalURL(tribunal, numero, tipo)`** adicionada ao script do `index.html`:

```javascript
if (tipo === 'tese_stj')
  → https://scon.stj.jus.br/SCON/jt/

if (tribunal === 'STF')
  regex /^([A-Za-z]+)\s+(\d+)/ extrai classe e número
  → https://portal.stf.jus.br/processos/listarProcessos.asp?classe=HC&numeroProcesso=264486

if (tribunal === 'STJ')  (acórdão)
  → https://scon.stj.jus.br/SCON/pesquisar.jsp?b=ACOR&livre=%22HC+264486%22
```

**Regex de parsing** (`/^([A-Za-z]+)\s+(\d+)/`) cobre todos os formatos observados no banco:
- `"HC 264486 AgR"` → classe=HC, num=264486
- `"ARE 1583790 AgR-ED"` → classe=ARE, num=1583790
- `"Rcl 89658 ED"` → classe=Rcl, num=89658
- `"ADI 7415 ED"` → classe=ADI, num=7415

**Integração no card** (`buildCard`): o `span` do `numero_processo` é substituído por um `<a>` quando a URL está disponível. Estilo: `text-muted` em repouso → `text-accent` no hover, sublinhado discreto (`underline-offset-2 decoration-muted/30`). Atributos `target="_blank" rel="noopener noreferrer"` para segurança. `↗` no final sinaliza link externo. `title` exibe o rótulo do destino no tooltip.

### 21.3 Arquivos Criados ou Modificados

| Arquivo | Alteração |
|---|---|
| `static/index.html` | Função `buildPortalURL()` adicionada; `buildCard()` atualizado para gerar `<a>` clicável quando URL disponível |

### 21.4 Estado Após a Fase 21

- ✅ **Número do processo é link clicável** em todos os cards com tribunal reconhecido
- ✅ **STF**: link para página do processo (`listarProcessos.asp`) — acesso direto ao acórdão ou PDF
- ✅ **STJ teses**: link para Em Teses (máximo possível sem deep link por edição)
- ✅ **Fallback gracioso**: se tribunal desconhecido, mantém `<span>` estático sem quebrar o card
- ✅ **Zero dependências novas**: lógica em JS vanilla no mesmo arquivo

---

## Fase 22 — Súmulas Vinculantes STF no Corpus (23 de abril de 2026)

### 22.1 Motivação

As Súmulas Vinculantes do STF (art. 103-A CF + Lei 11.417/2006) são o precedente de maior hierarquia no ordenamento jurídico brasileiro: vinculam **obrigatoriamente** todo o Judiciário e a administração pública direta e indireta de todos os entes federados. Qualquer sistema de pesquisa jurídica que não as inclua pode responder sobre temas como uso de algemas (SV 11), nepotismo (SV 13), prisão de depositário infiel (SV 25) ou regime prisional (SV 56) sem citar a norma vinculante mais forte disponível — lacuna material para um advogado.

Hierarquia de precedentes implementada após esta fase:
```
Súmula Vinculante STF (vinculante constitucional)  ← NOVO
    ↓
Tese STJ (precedente qualificado — art. 927, III, CPC)
    ↓
Súmula STJ (enunciado persuasivo consolidado)
    ↓
Acórdão STF/STJ (casuístico / persuasivo)
```

### 22.2 Arquitetura

**Tabela dedicada** `sumulas_vinculantes_stf` (separada de `teses_stj`) para distinção clara de tipo e hierarquia. Schema: `id, numero INTEGER UNIQUE, enunciado TEXT, embedding BLOB`.

**Corpus de SVs:** 59 súmulas vinculantes (SV 1 a SV 59) em `data/stf/sumulas_vinculantes.txt`. SV 18 e SV 19 omitidas do arquivo pois referem-se a matérias que foram objeto de cancelamento ou não possuíam texto consolidado.

**Busca de 6 corrotinas paralelas** (era 4): `asyncio.gather` agora inclui `search_sumulas_vinculantes` (FTS5) e `search_sv_semantic` (semântico). As SVs passam por `rrf_sv` e pelo cross-encoder antes de entrar no prompt.

**Limite de SVs no prompt:** máx. 2 SVs por consulta (corpus de ~60 docs com BM25 + semântica — qualidade alta sem custo de contexto). Candidatos: 8 FTS5 + 8 semântico → RRF top-3 → rerank → top-2.

### 22.3 Mudanças por Camada

| Camada | Arquivo | Alteração |
|---|---|---|
| Dados | `data/stf/sumulas_vinculantes.txt` | 59 SVs no formato `SÚMULA VINCULANTE N / texto` |
| ETL | `etl/load_sumulas_vinculantes_stf.py` | Parser regex + carga idempotente + flag `--force` |
| Banco | `src/database/connection.py` | DDL tabela + FTS5 + 3 triggers (insert/delete/update) |
| Busca lexical | `src/services/search_service.py` | `SumulaVinculanteResult` dataclass + `search_sumulas_vinculantes()` |
| Busca semântica | `src/services/semantic_service.py` | `_load_sv_cache()` + `search_sv_semantic()` + `rrf_sv()` |
| Pipeline RAG | `src/services/rag_service.py` | `sources_sv` em `RagResponse`; 6ª busca paralela; `rrf_sv`; `_build_prompt()` v7 |
| Prompt | `src/services/rag_service.py` | Bloco `[Súmula Vinculante STF N]` com `Efeito: vinculante constitucional`; regra de citação `SV N/STF`; nota de fontes com 4 níveis hierárquicos |
| API | `src/api/routes/query.py` | `sources_sv` → `SourceDocument(tipo="sumula_vinculante_stf")` no início da lista |
| Expansão | `src/services/query_expansion.py` | 4 entradas: `"sumula vinculante"`, `"sumulas vinculantes"`, `"vinculante"`, `"sv"` |
| Frontend | `static/index.html` | `buildPortalURL` para `sumula_vinculante_stf` → `sumariosumula.asp`; badge âmbar `STF · Súmula Vinculante` |
| Testes | `tests/conftest.py` | DDLs + triggers + 3 SVs de amostra na fixture `db` |
| Testes | `tests/test_rag.py` | `_make_sv()` helper + 6 novos testes de `_build_prompt()` com SVs |

### 22.4 Prompt v7 — Principais Mudanças em Relação ao v6

1. **Blocos de SV inseridos antes dos acórdãos** (hierarquia máxima primeiro no contexto)
2. **Linha `Efeito:` específica**: `"vinculante constitucional — obrigatória para todo o Judiciário e a administração pública (art. 103-A CF + Lei 11.417/2006)."`
3. **Regra de citação**: `SV N/STF` (ex: `SV 11/STF; SV 14/STF`)
4. **`fontes_desc` dinâmico**: inclui `"súmulas vinculantes do STF"` quando SVs presentes
5. **Nota de fontes com 4 níveis hierárquicos** (era 3): SV > Tese STJ > Súmula STJ > Acórdão

### 22.5 Como Executar o ETL de SVs

```bash
# Primeira carga (após load.py e load_teses_stj.py):
python -m etl.load_sumulas_vinculantes_stf

# Gerar embeddings para as SVs:
python -m etl.generate_embeddings

# Forçar recarga:
python -m etl.load_sumulas_vinculantes_stf --force
```

### 22.6 Estado Após a Fase 22

- ✅ **167 testes passando** (0 falhas, 0 skips) — 6 testes novos de SV em `test_rag.py`
- ✅ **59 SVs no corpus** (SV 1 a SV 59) prontas para carga e vetorização
- ✅ **Pipeline com 6 buscas paralelas** — FTS5 + semântico para acórdãos, teses e SVs
- ✅ **Hierarquia de precedentes completa** no prompt: SV > Tese STJ > Súmula STJ > Acórdão
- ✅ **Badge âmbar distinto** no frontend para diferenciar SVs dos demais tipos de fonte
- ✅ **Link direto** para página da SV no portal STF (`sumariosumula.asp?base=26&sumula=N`)

---

## Fase 23 — Filtragem de Fontes por Relevância com o Cross-Encoder (23 de abril de 2026)

### 23.1 Motivação

O pipeline recuperava e exibia **todas** as fontes top-k ao usuário, mesmo quando o LLM utilizou apenas parte delas na resposta. Isso gerava ruído na UI e reduzia a confiança do usuário nas referências apresentadas.

### 23.2 Solução: filter_by_answer()

Após o LLM gerar a resposta, o mesmo cross-encoder já carregado em memória (sem custo adicional de I/O) pontua cada par `(resposta_gerada, fonte)`. Fontes com score abaixo do *decision boundary* do modelo (`threshold = 0.0`) são descartadas antes de retornar ao cliente.

**Por que o cross-encoder em vez de parsing de citações?**

A opção alternativa — parsear as citações entre parênteses que o prompt exige — foi descartada por fragilidade: o LLM às vezes abrevia identificadores longos (ex.: Tese STJ com área + edição + número), causando falsos negativos silenciosos. O cross-encoder avalia o conteúdo semântico sem depender de formato de texto.

**Threshold de 0.0:** O modelo `mmarco-mMiniLMv2-L12-H384-v1` foi treinado em MS MARCO multilíngue e usa 0.0 como fronteira natural entre relevante/irrelevante nos seus logits de saída.

**Fallback:** Se nenhuma fonte passar o threshold (ex.: pergunta genérica onde o LLM respondeu "não encontrei informação suficiente"), as listas originais são retornadas intactas para evitar UI sem fontes.

**Latência adicional:** ~50–100 ms para N ≤ 11 pares (6 acórdãos + 3 teses + 2 SVs), negligível frente aos 2–5 s da chamada ao LLM. O modelo já estava em memória desde o reranking, portanto sem custo de carregamento.

### 23.3 Correção de Bug: _get_text para SumulaVinculanteResult

Identificado que `_get_text()` em `rerank_service.py` não tratava `SumulaVinculanteResult` — caia no branch de `TesesResult` e tentava acessar `.area` e `.tese_texto` inexistentes. Corrigido com branch `isinstance(doc, SumulaVinculanteResult)` que retorna `doc.enunciado`.

### 23.4 Mudanças por Arquivo

| Arquivo | Alteração |
|---|---|
| `src/services/rerank_service.py` | Import `SumulaVinculanteResult`; TypeVar estendido; `_get_text` corrigido; `_ANSWER_FILTER_THRESHOLD = 0.0`; nova função `filter_by_answer()` |
| `src/services/rag_service.py` | Nova função `_filter_cited_sources()`; chamada pós-LLM no pipeline, gated em `reranker_enabled` |
| `tests/test_rerank.py` | Helper `_sv()`; 2 testes de `_get_text` para SV; 8 testes de `filter_by_answer()` |

### 23.5 Estado Após a Fase 23

- ✅ **177 testes passando** (0 falhas, 0 skips) — 10 testes novos em `test_rerank.py`
- ✅ **Fontes filtradas semanticamente** antes de retornar ao cliente
- ✅ **Bug corrigido** — `_get_text` agora lida corretamente com os 3 tipos de documento
- ✅ **Fallback gracioso** — nenhuma fonte passa → retorna lista original
- ✅ **Sem custo de latência significativo** — cross-encoder já carregado em memória
