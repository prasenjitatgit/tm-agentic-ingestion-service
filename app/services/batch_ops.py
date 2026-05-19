"""Batch insert operations for chunks and embeddings.

Provides efficient bulk INSERT statements to minimize database round-trips
during the ingestion pipeline.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.models.db_models import DocumentChunk, Embedding


def batch_insert_chunks(
    session: Session, doc_id: uuid.UUID, chunks: list[dict[str, Any]]
) -> list[uuid.UUID]:
    """Batch insert chunks for a document using a single INSERT statement.

    Args:
        session: Active SQLAlchemy database session.
        doc_id: UUID of the parent document.
        chunks: List of dicts with keys: content, page_no (optional), created_by (optional).

    Returns:
        List of generated chunk_id UUIDs in the same order as input chunks.
    """
    rows = [
        {
            "chunk_id": uuid.uuid4(),
            "doc_id": doc_id,
            "content": chunk["content"],
            "page_no": chunk.get("page_no"),
            "created_at": func.now(),
            "created_by": chunk.get("created_by", "ingestion-service"),
        }
        for chunk in chunks
    ]

    stmt = pg_insert(DocumentChunk).values(rows)
    session.execute(stmt)

    return [row["chunk_id"] for row in rows]


def batch_insert_embeddings(
    session: Session, embeddings: list[dict[str, Any]]
) -> None:
    """Batch insert embeddings using a single INSERT with ON CONFLICT DO NOTHING.

    Duplicate chunk_ids are silently skipped for idempotency.

    Args:
        session: Active SQLAlchemy database session.
        embeddings: List of dicts with keys: chunk_id, vector, created_by (optional).
    """
    rows = [
        {
            "embedding_id": uuid.uuid4(),
            "chunk_id": emb["chunk_id"],
            "vector": emb["vector"],
            "created_at": func.now(),
            "created_by": emb.get("created_by", "ingestion-service"),
        }
        for emb in embeddings
    ]

    stmt = pg_insert(Embedding).values(rows)
    stmt = stmt.on_conflict_do_nothing(index_elements=["chunk_id"])
    session.execute(stmt)
