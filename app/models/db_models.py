"""SQLAlchemy ORM models.

Tables:
    documents        — file metadata + lifecycle (TO BE INGESTED → CHUNKED → EMBEDDED → INGESTED)
    document_chunks  — split text segments (immutable once created)
    embeddings       — vector per chunk, HNSW-indexed for cosine search (immutable once created)

Status lifecycle:
    Ingestion path:  TO BE INGESTED → CHUNKED → EMBEDDED → INGESTED
    Deletion path:   INGESTED → TO BE DELETED → DELETED
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.core.config import get_settings

# ── Status constants ─────────────────────────────────────────────────────────
STATUS_TO_BE_INGESTED = "TO BE INGESTED"
STATUS_CHUNKED = "CHUNKED"
STATUS_EMBEDDED = "EMBEDDED"
STATUS_INGESTED = "INGESTED"
STATUS_TO_BE_DELETED = "TO BE DELETED"
STATUS_DELETED = "DELETED"

ALLOWED_STATUSES = (
    STATUS_TO_BE_INGESTED,
    STATUS_CHUNKED,
    STATUS_EMBEDDED,
    STATUS_INGESTED,
    STATUS_TO_BE_DELETED,
    STATUS_DELETED,
)

_settings = get_settings()


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""


class Document(Base):
    """File metadata + ingestion lifecycle.

    Primary key is UUID with server-generated default via gen_random_uuid().
    """

    __tablename__ = "documents"

    doc_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    doc_type: Mapped[str] = mapped_column(String(30), nullable=False)
    doc_hash: Mapped[str] = mapped_column(String(100), nullable=False)
    doc_name: Mapped[str] = mapped_column(String(200), nullable=False)
    author: Mapped[str | None] = mapped_column(String(100))
    source_url: Mapped[str] = mapped_column(String(500), nullable=False)
    source_updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    version: Mapped[str | None] = mapped_column(String(20))
    s3_url: Mapped[str] = mapped_column(String(500), nullable=False)
    knowledge_base_type: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=STATUS_TO_BE_INGESTED
    )
    error: Mapped[dict | None] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    created_by: Mapped[str] = mapped_column(String(100), nullable=False)
    updated_by: Mapped[str] = mapped_column(String(100), nullable=False)


class DocumentChunk(Base):
    """Single text chunk linked to a Document.

    Chunks are immutable once created — no updated_at/updated_by columns.
    """

    __tablename__ = "document_chunks"

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    doc_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.doc_id"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    page_no: Mapped[int | None] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    created_by: Mapped[str] = mapped_column(String(100), nullable=False)


class Embedding(Base):
    """One vector per chunk, HNSW-indexed for cosine search.

    Embeddings are immutable once created — no updated_at/updated_by columns.
    """

    __tablename__ = "embeddings"

    embedding_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_chunks.chunk_id"),
        nullable=False,
        unique=True,
    )
    vector: Mapped[list[float]] = mapped_column(
        Vector(_settings.embed_dim), nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    created_by: Mapped[str] = mapped_column(String(100), nullable=False)
