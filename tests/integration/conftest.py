"""Integration test fixtures.

These tests require a running PostgreSQL **with pgvector** and the schema
from `docker/init-db.sql` already applied. Bring it up with:

    docker compose up -d postgres

The DSN is read from `INTEGRATION_PG_DSN` (defaults to the docker-compose
mapping). If Postgres is unreachable, all integration tests are skipped.
"""

from __future__ import annotations

import os

import pytest

# ── 1. Set env BEFORE app modules import ─────────────────────────────────────
INTEGRATION_PG_DSN = os.environ.get(
    "INTEGRATION_PG_DSN",
    "postgresql+psycopg://rag:rag@localhost:5432/rag",
)
os.environ["PG_DSN"] = INTEGRATION_PG_DSN
os.environ.setdefault("USE_PARAMETER_STORE", "false")
os.environ.setdefault("S3_BUCKET", "charter-rag-test")
os.environ.setdefault("NVIDIA_API_KEY", "test-key")
os.environ.setdefault("LOG_FORMAT", "console")
os.environ.setdefault("EMBED_DIM", "1536")
os.environ.setdefault("MAX_RETRIES", "1")  # fail fast in tests

# ── 2. Rebuild engine / SessionLocal so they pick up INTEGRATION_PG_DSN ──────
from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.services import db_service as _db  # noqa: E402

get_settings.cache_clear()
_settings = get_settings()

_db.engine = create_engine(
    _settings.pg_dsn,
    pool_size=_settings.pg_pool_size,
    max_overflow=_settings.pg_pool_max_overflow,
    pool_pre_ping=True,
    future=True,
)
_db.SessionLocal = sessionmaker(
    bind=_db.engine, expire_on_commit=False, autoflush=False, future=True
)


# ── 3. Per-test DB cleanup ───────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _truncate_tables():
    """Wipe pipeline tables before each integration test for isolation."""
    with _db.engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE embeddings, document_chunks, documents CASCADE;"))
    yield


@pytest.fixture
def engine():
    return _db.engine


@pytest.fixture
def SessionLocal():
    return _db.SessionLocal
