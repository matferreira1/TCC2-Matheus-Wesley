# Guia de Deploy — IAJuris

Deploy em PaaS (Railway / Render / Fly.io) com banco SQLite versionado via Git LFS.

---

## Visão geral

```
[Local] ETL já executado → data/db/iajuris.db gerado
           ↓
    Git LFS → push para GitHub (banco versionado como artefato binário)
           ↓
[Docker build na plataforma]
    Stage 1 (builder):
      pip install requirements.txt (sem locust)
      python download_models.py → baixa MiniLM + cross-encoder (~400 MB)
    Stage 2 (runtime):
      copia pacotes + modelos cacheados + código + iajuris.db
      CMD: uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
           ↓
Cold start: ~5–10s (modelos na imagem, banco presente, testes pulados)
```

---

## Pré-requisitos

- Git LFS instalado localmente
- Docker instalado (para teste local antes do deploy)
- ETL já executado e `data/db/iajuris.db` gerado
- `GROQ_API_KEY` válida

---

## Passo 1 — Patch no main.py

O `main.py` atual executa a suite de testes a cada startup. Em produção isso
adiciona 30–60s de latência e pode causar timeout no health check da plataforma.

Aplique o patch descrito em `main_patch.py`: substitua o bloco `lifespan` para
respeitar a variável `RUN_TESTS_ON_STARTUP`.

```python
# Adicione no topo de main.py
import os

# Substitua o lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando IAJuris...")

    if os.getenv("RUN_TESTS_ON_STARTUP", "true").lower() == "true":
        await _run_tests()
    else:
        logger.info("STARTUP — Testes pulados (RUN_TESTS_ON_STARTUP=false).")

    await open_db()
    await init_db()
    logger.info("Banco de dados pronto. Servidor disponível.")

    yield

    logger.info("Encerrando IAJuris...")
    await close_db()
    logger.info("Encerramento concluído.")
```

---

## Passo 2 — Configurar Git LFS

O `iajuris.db` tem centenas de MB de embeddings BLOB. Git rejeita arquivos >100MB.
O Git LFS armazena o binário externamente e mantém um ponteiro no repositório.

```bash
# Instalar Git LFS (uma vez por máquina)
brew install git-lfs          # macOS
sudo apt install git-lfs      # Ubuntu/Debian

# Ativar LFS no repositório
git lfs install

# Rastrear o banco
git lfs track "data/db/iajuris.db"

# Commitar o .gitattributes gerado pelo LFS
git add .gitattributes
git commit -m "chore: configura Git LFS para iajuris.db"

# Adicionar e enviar o banco
git add data/db/iajuris.db
git commit -m "feat: adiciona banco iajuris.db via Git LFS"
git push origin main
```

> **Verificação:** acesse o arquivo no GitHub — deve exibir "Stored with Git LFS".

---

## Passo 3 — Testar o build localmente

```bash
# Build (~10–15 min na 1ª vez — baixa modelos no builder stage)
docker build -t iajuris:local .

# Subir o container
docker run --rm \
  -p 8000:8000 \
  -e GROQ_API_KEY="sua_chave_aqui" \
  -e LLM_PROVIDER=groq \
  -e RUN_TESTS_ON_STARTUP=false \
  -e DEBUG=false \
  iajuris:local

# Health check
curl http://localhost:8000/api/v1/health
# Esperado: {"status":"ok"} em < 5s

# Query de teste
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "O que é habeas corpus?"}'
```

---

## Passo 4 — Deploy por plataforma

### Railway

1. [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**
2. Selecione `TCC2-Matheus-Wesley` — Railway detecta o `Dockerfile` automaticamente
3. **Variables** → adicione:

   | Variável | Valor |
   |---|---|
   | `GROQ_API_KEY` | sua chave (secret) |
   | `LLM_PROVIDER` | `groq` |
   | `RUN_TESTS_ON_STARTUP` | `false` |
   | `DEBUG` | `false` |
   | `PORT` | `8000` |

4. **Settings → Networking** → expose a porta `8000`
5. **Deploy** → acompanhe os logs do build

> **Plano:** Hobby ($5/mês, 512MB RAM) é suficiente.
> Se o build travar por OOM, use o plano Pro com 1GB durante o build.

---

### Render

1. [render.com](https://render.com) → **New** → **Web Service**
2. Conecte o repositório GitHub
3. Configure:
   - **Environment:** Docker
   - **Dockerfile Path:** `./Dockerfile`
   - **Instance Type:** Starter (512MB) — mínimo recomendado
4. **Environment Variables:**

   | Variável | Valor |
   |---|---|
   | `GROQ_API_KEY` | sua chave (secret) |
   | `LLM_PROVIDER` | `groq` |
   | `RUN_TESTS_ON_STARTUP` | `false` |
   | `DEBUG` | `false` |

5. **Create Web Service**

> **Atenção:** o plano Free hiberna após 15 min de inatividade (cold start lento).
> Use Starter ($7/mês) para instância sempre ativa.

---

### Fly.io

```bash
# Instalar flyctl
curl -L https://fly.io/install.sh | sh
fly auth login

# Inicializar (na raiz do projeto — não faz deploy ainda)
fly launch --name iajuris --region gru --no-deploy

# Edite fly.toml para garantir memória suficiente:
# [[vm]]
#   memory = "1gb"
#   cpu_kind = "shared"
#   cpus = 1

# Definir secrets
fly secrets set GROQ_API_KEY="sua_chave_aqui"
fly secrets set LLM_PROVIDER=groq
fly secrets set RUN_TESTS_ON_STARTUP=false
fly secrets set DEBUG=false

# Deploy
fly deploy

# Logs em tempo real
fly logs
```

---

## Passo 5 — Atualizar o banco no futuro

Quando re-executar o ETL e gerar um novo `iajuris.db`:

```bash
# Substituir o banco
cp /caminho/do/novo/iajuris.db data/db/iajuris.db

# Commitar via LFS (detecta automaticamente por .gitattributes)
git add data/db/iajuris.db
git commit -m "feat: atualiza corpus iajuris.db (mês/ano)"
git push origin main
# A plataforma dispara redeploy automático
```

---

## Variáveis de ambiente de produção

| Variável | Valor em produção | Descrição |
|---|---|---|
| `GROQ_API_KEY` | sua chave | **Secret** — nunca commitar |
| `LLM_PROVIDER` | `groq` | Provider LLM |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Modelo padrão |
| `RUN_TESTS_ON_STARTUP` | `false` | Pula pytest no lifespan |
| `DEBUG` | `false` | Desativa logs verbosos |
| `RAG_TOP_K` | `6` | Acórdãos STF após reranking |
| `RAG_TOP_K_TESES` | `3` | Teses/súmulas STJ após reranking |
| `RERANKER_ENABLED` | `true` | Cross-encoder ativo |
| `RATE_LIMIT_PER_MINUTE` | `10` | Req/min por IP em /query |

---

## Troubleshooting

**Build falha com OOM (out of memory):**
O download dos modelos consome ~1GB de RAM no builder stage.
No Railway, configure uma instância de build maior temporariamente.
No Render, use Instance Type Standard durante o primeiro build.

**`iajuris.db` não encontrado no container:**
Execute `git lfs ls-files` — o banco deve aparecer na lista.
Confirme que `.dockerignore` não está excluindo `data/db/iajuris.db`
(ele exclui `iajuris.db` na raiz, que é a cópia duplicada, não o banco correto).

**Modelos baixados a cada restart (cache miss):**
Confirme que `SENTENCE_TRANSFORMERS_HOME` está definida como `/app/models_cache`
tanto no builder quanto no runtime — o Dockerfile já faz isso por padrão.

**Health check falhando no startup:**
O `start-period=60s` no HEALTHCHECK dá tempo para o lifespan completar.
Se ainda falhar, verifique se `RUN_TESTS_ON_STARTUP=false` está definido —
com os testes ativos o startup pode ultrapassar 60s.

**Erro de autenticação Groq:**
Confirme que `GROQ_API_KEY` está nas variáveis de ambiente da plataforma,
não em um `.env` commitado (o `.dockerignore` exclui o `.env` corretamente).

**CORS bloqueando o frontend:**
O `main.py` permite apenas `localhost:3000` e `localhost:8000`.
Em produção, adicione a URL da plataforma na lista `allow_origins` do CORSMiddleware:
```python
allow_origins=[
    "https://seu-app.railway.app",  # ou .onrender.com / .fly.dev
    "http://localhost:8000",
]
```