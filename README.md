# tm-agentic-ingestion-service

Production-grade agentic ingestion service for a multimodal RAG system.

- **Orchestration:** LangGraph (StateGraph, conditional routing, centralized retry)
- **Execution:** Standalone Python script (`run_ingestion.py`) — no HTTP server required
- **Storage:** PostgreSQL + pgvector (HNSW), AWS S3
- **Models (NVIDIA NIM):**
  - VLM `llama-3.1-nemotron-nano-vl-8b-v1` — image summarization
  - Embeddings `llama-nemotron-embed-1b-v2` (1024-dim, cosine)
- **Resilience:** `pybreaker` (`fail_max=3`, `reset_timeout=60`) on every LLM/VLM/embed call
- **Resumability:** Priority-based fetching (EMBEDDED > CHUNKED > TO BE INGESTED), atomic status advancement, idempotent nodes
- **Secrets:** AWS Parameter Store (placeholder names in `.env.example`)

## Topology

```
fetch_metadata → route_by_status
   ├─ TO BE INGESTED → select_parser → docling_convert
   │                        ↓
   │          (has_images?) → summarize_images → assemble
   │                    ↘ (no images) ─────────────╮
   │                                               ▼
   │                                     generate_chunks
   │                                               ↓
   ├─ CHUNKED ──────────────────────────► embed_chunks
   │                                               ↓
   ├─ EMBEDDED ─────────────────────► update_status_ingested → END
   │
   └─ no doc → END

(any node) ──error──► retry_handler ──► failed_node | failure_handler
```

## Layout

```
run_ingestion.py                # Standalone entry point (processes all pending docs)
app/
  main.py                       # FastAPI app entry (optional HTTP interface)
  core/
    config.py                   # Settings + AWS Parameter Store loader
    security.py                 # Auth / correlation-id middleware
    circuit_breaker.py          # pybreaker singletons (fail_max=3, reset_timeout=60)
  api/
    v1/ingestion.py             # POST /v1/ingestion/documents + POST /v1/ingestion/run
  graph/
    state.py                    # AgentState (TypedDict) + Pydantic value objects
    workflow.py                 # build_graph() + edge routing + run_ingestion_job
    nodes/
      fetcher.py                # fetch_metadata (priority-based, dead letter guard)
      docling_converter.py      # docling_convert (IBM Docling unified converter)
      parsers.py                # select_parser (S3 download + hash verification)
      processor.py              # summarize_images + assemble
      chunker.py                # generate_chunks (idempotent, atomic status)
      embedder.py               # embed_chunks (idempotent, resumable from DB)
      status.py                 # @node, classify(), retry/failure/update_status
  services/
    s3_service.py               # KB-partitioned S3 client
    llm_service.py              # NVIDIA NIM VLM + embedding clients (breaker-wrapped)
    db_service.py               # SQLAlchemy engine, SessionLocal, session_scope, ping
    batch_ops.py                # Bulk insert/delete for chunks and embeddings
    docling_service.py          # IBM Docling converter wrapper (lazy-loaded)
  models/
    db_models.py                # Document, DocumentChunk, Embedding (+ status consts)
  utils/
    logger.py                   # Structured (structlog) logging setup
docker/init-db.sql              # pgvector + tables + HNSW index (mounted on first boot)
tests/unit/                     # chunking + retry + fetcher + embedder + batch ops
tests/integration/              # E2E LangGraph graph topology + routing
```

## Quick start (local Python)

```bash
python -m venv venv && venv\Scripts\activate   # Windows
# python -m venv .venv && source .venv/bin/activate  # Linux/Mac

pip install -e ".[dev]"
cp .env.example .env       # set NVIDIA_API_KEY, PG_DSN, etc.

# Bootstrap the database schema
psql "$PG_DSN" -f docker/init-db.sql

# Insert documents into the documents table (status defaults to 'TO BE INGESTED')
# Then run the ingestion pipeline:
python run_ingestion.py
```

### CLI options

```bash
# Process all pending documents (unlimited)
python run_ingestion.py

# Limit to 5 documents
python run_ingestion.py --max-documents 5

# Filter by knowledge base type
python run_ingestion.py --knowledge-base-type Maintenance
```

### Optional: FastAPI server (for programmatic document submission)

```bash
uvicorn app.main:app --reload --port 8080

# Submit a document via API
curl -X POST http://localhost:8080/v1/ingestion/documents \
  -H "Content-Type: application/json" \
  -d '{"doc_type": "PDF", "doc_hash": "abc123", "doc_name": "manual.pdf", "source_url": "https://example.com", "source_updated_at": "2025-01-15T10:00:00Z", "s3_url": "s3://bucket/manual.pdf", "knowledge_base_type": "Maintenance", "created_by": "user"}'
```

## Quick start (Docker)

```bash
export NVIDIA_API_KEY=...
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...

# Bring up Postgres+pgvector. Schema bootstraps on first run.
docker compose up --build
```

`docker-compose.yml` mounts `docker/init-db.sql` into the Postgres container's
`/docker-entrypoint-initdb.d/`, so the schema (including the `vector(1024)`
column and HNSW index) is provisioned automatically on first launch.

## Pipeline Resumability

The pipeline is stateful and resumable. If a crash occurs mid-pipeline:

1. **Status preserved** — Each processing node atomically writes its output data AND advances the document status in a single DB transaction. A crash leaves the document at its last completed checkpoint.
2. **Priority fetching** — On the next run, the fetcher picks up incomplete documents closest to completion first (EMBEDDED > CHUNKED > TO BE INGESTED).
3. **Conditional routing** — The router inspects `DocMetadata.status` and skips already-completed stages.
4. **Idempotent nodes** — Chunker and embedder clean up partial data before re-processing, so re-runs are safe.
5. **Dead letter guard** — Documents that fail too many times (default: 5) are skipped to prevent blocking.

## Tests

```bash
# Unit tests (no external dependencies)
pytest tests/unit -q

# Integration tests (require Postgres+pgvector running)
docker compose up -d postgres
INTEGRATION_PG_DSN=postgresql+psycopg://rag:rag@localhost:5432/rag \
  pytest tests/integration -q
```

## Constraints honoured

| # | Constraint | Where |
|---|---|---|
| 1 | Don't split `IMAGE_REFERENCE` / `TABLE_REFERENCE` / `SPEAKER_NOTES` | `graph/nodes/chunker.py` (`PROTECTED_TAGS`) |
| 2 | Skip `summarize_images` & `assemble` when no images | `graph/workflow.py::route_after_parser` |
| 3 | Centralized retry via `retry_handler` → `failed_node` or `failure_handler` | `graph/nodes/status.py::retry_handler_node` |
| 4 | All LLM/VLM calls wrapped in `pybreaker` | `core/circuit_breaker.py` + `services/llm_service.py` |
| 5 | Sequential per-document processing | `graph/workflow.py::run_ingestion_job` |
| 6 | Chunk size 800 / overlap 100 | `graph/nodes/chunker.py` defaults from `Settings` |
| 7 | Atomic status advancement in data-writing nodes | `graph/nodes/chunker.py`, `graph/nodes/embedder.py` |
| 8 | Priority-based resumable fetching with dead letter guard | `graph/nodes/fetcher.py` |
