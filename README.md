- Chack out the different braches for `llamaindex-dag-pipeline` and `langchain-dag-pipeline`

# RAG · Qdrant + Voyage + Claude

End-to-end local RAG: ingest PDFs → Qdrant → query via FastAPI → answer with Claude.

## Stack

- **Vector DB**: Qdrant (Docker)
- **Embeddings**: Voyage AI (`voyage-3`, 1024-dim)
- **Generation**: Anthropic Claude (Sonnet 4.6)
- **API**: FastAPI
- **UI**: Streamlit
- **Orchestration** (optional): Dagster, ingesting PDFs from Azure Blob Storage

## Project layout

```
rag-qdrant/
├── docker-compose.yml
├── Makefile                       # `make help` for shortcuts
├── pyproject.toml
├── .env.example
├── src/
│   ├── config.py                  # pydantic-settings
│   ├── ingest/
│   │   ├── loaders.py             # PDF -> page records
│   │   ├── chunker.py             # recursive char splitter
│   │   └── pipeline.py            # load -> chunk -> embed -> upsert
│   ├── retrieval/
│   │   ├── embeddings.py          # Voyage wrapper
│   │   └── store.py               # Qdrant wrapper
│   ├── generation/
│   │   └── llm.py                 # Claude wrapper
│   └── api/main.py                # FastAPI app
├── rag_pipelines/                 # Dagster orchestration (optional)
│   ├── definitions.py             # Definitions entrypoint
│   ├── assets.py                  # `ingested_pdf` partitioned asset
│   ├── resources.py               # AzureBlobResource
│   └── sensors.py                 # poll Azure container for new PDFs
├── scripts/
│   ├── smoke_test.py              # Phase 1 sanity check
│   ├── ingest_cli.py              # ingest PDFs
│   ├── ui.py                      # Streamlit UI
│   └── learn/                     # standalone learning walkthroughs
│       ├── qdrant_walkthrough.py
│       ├── voyage_walkthrough.py
│       ├── chunking_walkthrough.py
│       └── generation_walkthrough.py
└── data/pdfs/                     # drop your PDFs here
```

## First-time setup

### 1. Start Qdrant

```bash
docker compose up -d
# verify dashboard:  http://localhost:6333/dashboard
```

### 2. Install deps

```bash
uv sync                          # core deps only
uv sync --group dev --group dagster   # add dev + Dagster extras (or: `make install`)
```

### 3. Configure

```bash
cp .env.example .env
# fill in VOYAGE_API_KEY and ANTHROPIC_API_KEY
# (AZURE_BLOB_* vars are only needed if you run the Dagster pipeline)
```

## The learning path (recommended order)

These standalone scripts walk through each layer in isolation. Run them in order:

```bash
# Layer 1: Qdrant mechanics (no embeddings, fake vectors)
uv run python -m scripts.learn.qdrant_walkthrough

# Layer 2: Real Voyage embeddings, semantic search
uv run python -m scripts.learn.voyage_walkthrough

# Layer 3: Chunking strategies compared
uv run python -m scripts.learn.chunking_walkthrough

# Layer 4: Claude generation, closing the RAG loop
uv run python -m scripts.learn.generation_walkthrough
```

## Running the production stack

### 4. Smoke test (no PDFs needed)

```bash
uv run python -m scripts.smoke_test
```

You should see retrieved hits with cosine scores. **Do not skip this step** — it isolates Voyage + Qdrant from any PDF/chunking issues.

### 5. Ingest PDFs

```bash
# drop PDFs into ./data/pdfs/
uv run python -m scripts.ingest_cli ./data/pdfs
```

### 6. Run the API

```bash
uv run uvicorn src.api.main:app --reload
# http://localhost:8000/docs  for Swagger
```

### 7. Run the UI

```bash
uv run streamlit run scripts/ui.py
```

## Quick API check

```bash
curl http://localhost:8000/health
curl http://localhost:8000/stats

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "what is in this document?"}'
```

## Orchestrated ingestion (Dagster + Azure Blob)

The local `scripts.ingest_cli` flow is fine for one-off PDFs. For continuous ingestion
from a shared Azure Blob container, use the Dagster pipeline in `rag_pipelines/`:

- **Asset** `ingested_pdf` is dynamically partitioned — one partition per blob name.
  Already-materialized partitions are skipped, so re-runs are idempotent.
- **Sensor** `new_pdf_blob_sensor` polls the container every 60s, registers new blobs
  as partitions, and kicks off runs for them. It ships **stopped by default** — enable
  it from the Dagster UI when you're ready.
- **Resource** `AzureBlobResource` reads `AZURE_BLOB_CONNECTION_STRING` and
  `AZURE_BLOB_CONTAINER` from env so the same code runs locally and in Dagster+ Cloud.

```bash
# install the dagster extra (one-time)
uv sync --group dagster

# set Azure creds in .env, then:
make dagster                     # http://localhost:3000
# or run everything together:
make dev                         # API :8000 + UI :8501 + Dagster :3000
```

## Make targets

```
make install     Sync deps (dev + dagster groups)
make qdrant      Start Qdrant via docker compose
make api         Run FastAPI on :8000
make ui          Run Streamlit UI on :8501
make dagster     Run Dagster dev server on :3000
make dev         Run API + UI + Dagster together
make ingest      Ingest PDFs from ./data/pdfs into Qdrant
make test        Run pytest
make lint        Lint with ruff
make down        Stop Qdrant
```

## Where to go next

- **Phase 2**: streaming responses (`/query/stream` with SSE), streaming UI
- **Phase 3**: hybrid search (sparse + dense via Qdrant's BM25 support)
- **Phase 4**: reranking with `voyage-rerank-2`
- **Phase 5**: eval harness — measure retrieval & answer quality
- **Phase 6**: containerize the whole thing (API + UI + Qdrant in compose)
