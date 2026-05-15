# IAJuris — Instruções para o Claude Code

## Regra obrigatória: Diário Técnico e README

**Toda vez que uma mudança relevante for feita no projeto — novo arquivo, novo módulo, nova funcionalidade, nova configuração, novo teste, correção de bug significativa ou alteração de arquitetura — as seguintes ações são obrigatórias após concluir a implementação:**

1. **Atualizar o `DIARIO_TECNICO.md`** com uma entrada cronológica descrevendo:
   - O que foi feito e por quê (motivação técnica)
   - Decisões de projeto e alternativas descartadas
   - Problemas encontrados e soluções adotadas
   - Arquivos criados ou modificados (tabela)
   - Estado final da fase (checkboxes ✅)

2. **Atualizar o `README.md`** se alguma das seções abaixo ficou desatualizada:
   - Base de conhecimento (número de documentos)
   - Como funciona (diagrama do pipeline)
   - Stack tecnológica
   - Carga dos dados (ETL — comandos)
   - Estrutura do projeto (árvore de diretórios)
   - Variáveis de configuração disponíveis

Não peça confirmação — atualize os dois documentos imediatamente ao terminar cada tarefa.

## Contexto do projeto

Sistema RAG para consulta de jurisprudência brasileira (STF/STJ) — TCC de Engenharia de Software, UnB FGA. Autores: Matheus Ferreira Diogo e Wesley Lira. Orientador: Prof. Dr. Henrique Gomes de Moura.

## Pipeline atual (resumo)

```
Pergunta → FTS5/BM25 + query expansion
         → Semântica (MiniLM, cosine)
         → RRF (Reciprocal Rank Fusion)
         → Cross-encoder reranking
         → Prompt v5
         → LLM (Groq llama-3.3-70b-versatile ou Ollama llama3.2:3b)
         → Resposta com fontes citadas
```

## Stack

Python 3.12 · FastAPI · SQLite FTS5 (aiosqlite) · sentence-transformers · NumPy · slowapi · pytest

## Fases já documentadas no DIARIO_TECNICO.md

| Fase | Descrição |
|------|-----------|
| 1 | Configuração e infraestrutura |
| 2 | ETL STF |
| 3 | Serviços (search, Ollama, RAG) |
| 4 | API REST |
| 5 | Integração Groq + prompt v4 |
| 6 | Extrator de seções + análise de grounding |
| 7 | ETL STJ teses + busca paralela + prompt v5 |
| 8 | Suite completa de testes (85 testes) |
| 9 | Execução automática de testes no startup |
| 10 | Framework de avaliação experimental |
| 11 | Hardening de segurança (19 vulnerabilidades) |
| 12 | Busca semântica híbrida + RRF + corpus expandido (7.474 docs) |
| 13 | Cross-encoder reranking |
| 14 | Endpoint /health |
| 15 | Query expansion com sinônimos jurídicos |
| 16 | Interface web (frontend) |
