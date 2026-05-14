-- =============================================================================
-- tm-agentic-ingestion-service — PostgreSQL bootstrap schema
-- Mounted into the postgres container's /docker-entrypoint-initdb.d/ so it
-- runs once on first boot. Requires PostgreSQL 15+ and the pgvector image.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ── documents ───────────────────────────────────────────────────────────────
-- Status lifecycle: NEW → PROCESSING → EMBEDDED | FAILED
CREATE TABLE IF NOT EXISTS documents (
    doc_id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_type            TEXT NOT NULL CHECK (doc_type IN ('PDF', 'EXCEL', 'PPT', 'DOCX')),
    doc_hash            TEXT NOT NULL UNIQUE,
    doc_name            TEXT NOT NULL,
    author              TEXT,
    source              TEXT,
    version             TEXT,
    s3_url              TEXT NOT NULL,
    knowledge_base_type TEXT NOT NULL
        CHECK (knowledge_base_type IN ('Maintenance', 'Construction', 'BusinessIntelligence')),
    status              TEXT NOT NULL DEFAULT 'NEW'
        CHECK (status IN ('NEW', 'PROCESSING', 'EMBEDDED', 'FAILED')),
    last_error          JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by          TEXT NOT NULL DEFAULT 'ingestion-service',
    updated_by          TEXT NOT NULL DEFAULT 'ingestion-service'
);

CREATE INDEX IF NOT EXISTS idx_documents_status   ON documents (status, created_at);
CREATE INDEX IF NOT EXISTS idx_documents_kb_type  ON documents (knowledge_base_type);

-- ── document_chunks ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id             UUID NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    content            TEXT NOT NULL,
    normalized_content TEXT NOT NULL,
    chunk_hash         TEXT NOT NULL,
    chunk_simhash      BIGINT,
    section            TEXT,
    page               INTEGER,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by         TEXT NOT NULL DEFAULT 'ingestion-service',
    updated_by         TEXT NOT NULL DEFAULT 'ingestion-service',
    UNIQUE (doc_id, chunk_hash)
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_doc_id   ON document_chunks (doc_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_simhash  ON document_chunks (chunk_simhash);

-- ── embeddings ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS embeddings (
    embedding_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id     UUID NOT NULL UNIQUE REFERENCES document_chunks (chunk_id) ON DELETE CASCADE,
    vector       vector(1536) NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by   TEXT NOT NULL DEFAULT 'ingestion-service',
    updated_by   TEXT NOT NULL DEFAULT 'ingestion-service'
);

-- HNSW index for production-grade O(log n) cosine search
CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw_cosine
    ON embeddings USING hnsw (vector vector_cosine_ops);
