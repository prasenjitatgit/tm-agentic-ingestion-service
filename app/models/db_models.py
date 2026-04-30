"""SQLAlchemy ORM models.

Tables:
    documents        — file metadata + lifecycle (NEW → PROCESSING → EMBEDDED | FAILED)
    document_chunks  — split text segments (UNIQUE on (doc_id, chunk_hash))
    embeddings       — vector(1536) per chunk, HNSW-indexed for cosine search
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
    """File metadata + ingestion lifecycle."""

    __tablename__ = "documents"

    doc_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    doc_type: Mapped[str] = mapped_column(String(16), nullable=False)
    doc_hash: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    doc_name: Mapped[str] = mapped_column(Text, nullable=False)
    author: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(Text)
    version: Mapped[str | None] = mapped_column(Text)
    s3_url: Mapped[str] = mapped_column(Text, nullable=False)
    knowledge_base_type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default=STATUS_NEW)
    last_error: Mapped[dict | None] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    created_by: Mapped[str] = mapped_column(String(64), default="ingestion-service")
    updated_by: Mapped[str] = mapped_column(String(64), default="ingestion-service")


class DocumentChunk(Base):
    """Single text chunk linked to a Document."""

    __tablename__ = "document_chunks"
    __table_args__ = (UniqueConstraint("doc_id", "chunk_hash", name="uq_chunk_doc_hash"),)

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    doc_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    chunk_simhash: Mapped[int | None] = mapped_column(BigInteger)
    section: Mapped[str | None] = mapped_column(Text)
    page: Mapped[int | None] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    created_by: Mapped[str] = mapped_column(String(64), default="ingestion-service")
    updated_by: Mapped[str] = mapped_column(String(64), default="ingestion-service")


class Embedding(Base):
    """One vector per chunk (1536-dim cosine, HNSW-indexed)."""

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
    created_by: Mapped[str] = mapped_column(String(64), default="ingestion-service")
    updated_by: Mapped[str] = mapped_column(String(64), default="ingestion-service")
