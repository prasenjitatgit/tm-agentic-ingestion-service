-- =============================================================================
-- tm-agentic-ingestion-service — PostgreSQL bootstrap schema
-- Mounted into the postgres container's /docker-entrypoint-initdb.d/ so it
-- runs once on first boot. Requires PostgreSQL 15+ and the pgvector image.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ── documents ───────────────────────────────────────────────────────────────
-- Status lifecycle:
--   Ingestion path: TO BE INGESTED → CHUNKED → EMBEDDED → INGESTED
--   Deletion path:  INGESTED → TO BE DELETED → DELETED
CREATE TABLE IF NOT EXISTS documents (
    doc_id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_type            VARCHAR(30) NOT NULL CHECK (doc_type IN ('PDF', 'EXCEL', 'PPT', 'DOCX')),
    doc_hash            VARCHAR(100) NOT NULL UNIQUE,
    doc_name            VARCHAR(200) NOT NULL,
    author              VARCHAR(100),
    source_url          VARCHAR(500) NOT NULL,
    source_updated_at   TIMESTAMPTZ NOT NULL,
    version             VARCHAR(20),
    s3_url              VARCHAR(500) NOT NULL,
    knowledge_base_type VARCHAR(50) NOT NULL
        CHECK (knowledge_base_type IN ('Maintenance', 'Construction', 'BusinessIntelligence')),
    status              VARCHAR(20) NOT NULL DEFAULT 'TO BE INGESTED'
        CHECK (status IN ('TO BE INGESTED', 'CHUNKED', 'EMBEDDED', 'INGESTED', 'TO BE DELETED', 'DELETED')),
    error               JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by          VARCHAR(100) NOT NULL,
    updated_by          VARCHAR(100) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_status_created ON documents (status, created_at);
CREATE INDEX IF NOT EXISTS idx_documents_kb_type        ON documents (knowledge_base_type);

-- ── document_chunks ─────────────────────────────────────────────────────────
-- Simplified: chunks are immutable once created (no update columns)
CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id     UUID NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    content    TEXT NOT NULL,
    page_no    INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON document_chunks (doc_id);

-- ── embeddings ──────────────────────────────────────────────────────────────
-- Simplified: embeddings are immutable once created (no update columns)
-- Vector dimension: 1024 (NVIDIA llama-nemotron-embed-1b-v2)
CREATE TABLE IF NOT EXISTS embeddings (
    embedding_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id     UUID NOT NULL UNIQUE REFERENCES document_chunks (chunk_id) ON DELETE CASCADE,
    vector       vector(1024) NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by   TEXT NOT NULL
);

-- HNSW index for production-grade O(log n) cosine search
CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw_cosine
    ON embeddings USING hnsw (vector vector_cosine_ops);
