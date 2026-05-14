"""SQLAlchemy ORM models.

Tables:
    documents        — file metadata + lifecycle (NEW → PROCESSING → EMBEDDED | FAILED)
    document_chunks  — split text segments
    embeddings       — vector per chunk, HNSW-indexed for cosine search

NOTE: This module matches the ACTUAL production database schema where:
    - documents.doc_id is VARCHAR(100), not UUID
    - documents.error (JSONB) stores failure details
    - documents has extra columns: operation, failed_step
    - document_chunks uses doc_id VARCHAR(100) FK
    - embeddings.vector dimension is controlled by config (EMBED_DIM)
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.core.config import get_settings

# ── Status constants ─────────────────────────────────────────────────────────
STATUS_NEW = "NEW"
STATUS_PROCESSING = "PROCESSING"
STATUS_EMBEDDED = "EMBEDDED"
STATUS_FAILED = "FAILED"

ALLOWED_STATUSES = (STATUS_NEW, STATUS_PROCESSING, STATUS_EMBEDDED, STATUS_FAILED)

_settings = get_settings()


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""


class Document(Base):
    """File metadata + ingestion lifecycle.

    Matches the production schema where doc_id is VARCHAR(100).
    """

    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    doc_type: Mapped[str] = mapped_column(String(30), nullable=False)
    doc_hash: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    doc_name: Mapped[str] = mapped_column(String(200), nullable=False)
    author: Mapped[str | None] = mapped_column(String(100))
    source: Mapped[str | None] = mapped_column(String(500))
    version: Mapped[str | None] = mapped_column(String(20))
    s3_url: Mapped[str] = mapped_column(String(500), nullable=False)
    knowledge_base_type: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default=STATUS_NEW)
    error: Mapped[dict | None] = mapped_column(JSONB)

    # Extra lifecycle columns present in production schema.
    operation: Mapped[str] = mapped_column(String(10), nullable=False, default=STATUS_NEW)
    failed_step: Mapped[str | None] = mapped_column(String(30))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    created_by: Mapped[str] = mapped_column(String(100), default="ingestion-service")
    updated_by: Mapped[str] = mapped_column(String(100), default="ingestion-service")


class DocumentChunk(Base):
    """Single text chunk linked to a Document."""

    __tablename__ = "document_chunks"

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    doc_id: Mapped[str] = mapped_column(
        String(100), ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_content: Mapped[str | None] = mapped_column(Text)
    chunk_hash: Mapped[str | None] = mapped_column(String(128))
    chunk_simhash: Mapped[int | None] = mapped_column(BigInteger)
    section: Mapped[str | None] = mapped_column(Text)
    page: Mapped[int | None] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    created_by: Mapped[str] = mapped_column(String(100), default="ingestion-service")
    updated_by: Mapped[str] = mapped_column(String(100), default="ingestion-service")


class Embedding(Base):
    """One vector per chunk, HNSW-indexed for cosine search."""

    __tablename__ = "embeddings"

    embedding_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_chunks.chunk_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    vector: Mapped[list[float]] = mapped_column(Vector(_settings.embed_dim), nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    created_by: Mapped[str] = mapped_column(String(100), default="ingestion-service")
    updated_by: Mapped[str] = mapped_column(String(100), default="ingestion-service")
