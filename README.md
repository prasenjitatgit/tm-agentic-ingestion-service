# tm-agentic-ingestion-service

Production-grade agentic ingestion service for a multimodal RAG system.

- **Orchestration:** LangGraph (StateGraph, conditional routing, centralized retry)
- **API:** FastAPI (`POST /v1/ingestion/run`, returns `job_id` + `STARTED`, runs in `BackgroundTasks`)
- **Storage:** PostgreSQL + pgvector (HNSW), AWS S3
- **Models (NVIDIA NIM):**
  - VLM `llama-3.1-nemotron-nano-vl-8b-v1` — image summarization
  - Embeddings `llama-nemotron-embed-1b-v2` (1536-dim, cosine)
- **Resilience:** `pybreaker` (`fail_max=3`, `reset_timeout=60`) on every LLM/VLM/embed call
- **Secrets:** AWS Parameter Store (placeholder names in `.env.example`)

## Topology

```
fetch_metadata → select_parser
   → parse_pdf | parse_docx | parse_pptx | parse_excel
      → (has_images?) → summarize_images → assemble
                     ↘ (no images) ──────────────╮
   → generate_chunks → embed_chunks → update_status

(any node) ──error──► retry_handler ──► failed_node | failure_handler
```

## Layout

```
app/
  main.py                       # FastAPI app entry (middleware + router wiring)
  core/
    config.py                   # Settings + AWS Parameter Store loader
    security.py                 # Auth / correlation-id middleware
    circuit_breaker.py          # pybreaker singletons (fail_max=3, reset_timeout=60)
  api/
    v1/ingestion.py             # POST /v1/ingestion/run + job registry + health
  graph/
    state.py                    # AgentState (TypedDict) + Pydantic value objects
    workflow.py                 # build_graph() + edge routing + run_ingestion_job
    nodes/
      fetcher.py                # fetch_metadata
      parsers.py                # select_parser + parse_pdf/docx/pptx/excel
      processor.py              # summarize_images + assemble
      chunker.py                # generate_chunks + tag-aware recursive splitter
      embedder.py               # embed_chunks
      status.py                 # @node, classify(), retry/failure/update_status
  services/
    s3_service.py               # KB-partitioned S3 client
    llm_service.py              # NVIDIA NIM VLM + embedding clients (breaker-wrapped)
    db_service.py               # SQLAlchemy engine, SessionLocal, session_scope, ping
  models/
    db_models.py                # Document, DocumentChunk, Embedding (+ status consts)
  utils/
    logger.py                   # Structured (structlog) logging setup
docker/init-db.sql              # pgvector + tables + HNSW index (mounted on first boot)
tests/unit/                     # chunking + retry classifier + S3 helpers
tests/integration/              # E2E LangGraph against real Postgres+pgvector
```

## Quick start (local Python)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env       # set NVIDIA_API_KEY, PG_DSN, etc.

psql "$PG_DSN_PSQL" -f docker/init-db.sql
uvicorn app.main:app --reload --port 8080

# Trigger ingestion (non-blocking)
curl -X POST http://localhost:8080/v1/ingestion/run \
  -H "Content-Type: application/json" \
  -d '{"knowledge_base_type": "Maintenance", "max_documents": 5}'
# → {"job_id": "...", "status": "STARTED"}

# Poll job
curl http://localhost:8080/v1/ingestion/jobs/<job_id>
```

## Quick start (Docker)

```bash
# Provide the NVIDIA NIM key (and optional AWS creds) in your shell env.
export NVIDIA_API_KEY=...
export AWS_ACCESS_KEY_ID=...   # optional, for real S3 uploads
export AWS_SECRET_ACCESS_KEY=...

# Bring up Postgres+pgvector and the app. Schema bootstraps on first run.
docker compose up --build

# Trigger
curl -X POST http://localhost:8080/v1/ingestion/run \
  -H "Content-Type: application/json" \
  -d '{"knowledge_base_type": "Maintenance", "max_documents": 1}'
```

`docker-compose.yml` mounts `docker/init-db.sql` into the Postgres container's
`/docker-entrypoint-initdb.d/`, so the schema (including the `vector(1536)`
column and HNSW index) is provisioned automatically on first launch.

## Tests

```bash
# Unit tests (no external dependencies)
pytest tests/unit -q

# Integration tests (require Postgres+pgvector running)
docker compose up -d postgres
INTEGRATION_PG_DSN=postgresql+psycopg://rag:rag@localhost:5432/rag \
  pytest tests/integration -q
```

The integration suite stubs S3, the VLM, and the embedding model in-process,
exercises the compiled LangGraph end-to-end against the real database, and
asserts that:

- a document with images is processed through every node and ends `EMBEDDED`
- a document without images skips `summarize_images` / `assemble`
- a non-retryable embedding failure marks the document `FAILED` with
  `last_error.failed_node = "embed_chunks"`
- an empty `documents` table results in a graceful no-op

## Constraints honoured

| # | Constraint | Where |
|---|---|---|
| 1 | Don't split `IMAGE_REFERENCE` / `TABLE_REFERENCE` / `SPEAKER_NOTES` | `graph/nodes/chunker.py` (`PROTECTED_TAGS`) |
| 2 | Skip `summarize_images` & `assemble` when no images | `graph/workflow.py::route_after_parser` |
| 3 | Centralized retry via `retry_handler` → `failed_node` or `failure_handler` | `graph/nodes/status.py::retry_handler_node` |
| 4 | All LLM/VLM calls wrapped in `pybreaker` | `core/circuit_breaker.py` + `services/llm_service.py` |
| 5 | Sequential per-document processing | `graph/workflow.py::run_ingestion_job` |
| 6 | Chunk size 800 / overlap 100 | `graph/nodes/chunker.py` defaults from `Settings` |
| 7 | FastAPI `POST /v1/ingestion/run` (background task) | `api/v1/ingestion.py` |

## Endpoints

| Method | Path                                | Purpose                                   |
|--------|-------------------------------------|-------------------------------------------|
| POST   | `/v1/ingestion/run`                 | Start ingestion job (returns `job_id`)    |
| GET    | `/v1/ingestion/jobs/{job_id}`       | Job status + per-doc results              |
| GET    | `/v1/documents/{doc_id}`            | Document status snapshot                  |
| GET    | `/healthz`                          | Liveness                                  |
| GET    | `/readyz`                           | Readiness (DB + S3 + NIM key)             |
```
