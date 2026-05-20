"""`embed_chunks` node — batch-embed and persist vectors.

The node is idempotent with resumability support: if in-memory chunks are
empty, it loads them from the database. It deletes any existing embeddings
for the document before inserting new ones, and advances the document status
to EMBEDDED within the same atomic transaction.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select

from app.graph.nodes.status import node
from app.graph.state import AgentState, Chunk
from app.models.db_models import Document, DocumentChunk, STATUS_EMBEDDED
from app.services.batch_ops import batch_insert_embeddings, delete_embeddings_for_document
from app.services.db_service import session_scope
from app.services.llm_service import embed_texts


@node("embed_chunks")
def embed_chunks_node(state: AgentState) -> dict[str, Any]:
    """Idempotent embedder with resumability: load chunks if needed, cleanup → insert → status.

    1. If in-memory chunks list is empty/None, load chunks from document_chunks
       table by doc_id (resumability for CHUNKED documents).
    2. If still no chunks after DB query, return empty result without modifying status.
    3. Call embedding service to generate vectors.
    4. Within a single transaction: delete old embeddings → insert new embeddings
       → update status to EMBEDDED, set error to NULL.

    On embedding service failure (before DB write), the error propagates and
    status remains at CHUNKED. On transaction failure, all changes roll back.
    """
    meta = state["doc_metadata"]
    doc_id = uuid.UUID(meta.doc_id)

    # 1. Check in-memory chunks; if empty, load from DB.
    chunks: list[Chunk] = list(state.get("chunks") or [])

    if not chunks:
        with session_scope() as session:
            rows = session.execute(
                select(DocumentChunk).where(DocumentChunk.doc_id == doc_id)
            ).scalars().all()

        chunks = [
            Chunk(
                chunk_id=str(row.chunk_id),
                page_no=row.page_no,
                content=row.content,
            )
            for row in rows
        ]

    # 2. If still no chunks, return empty result without modifying status.
    if not chunks:
        return {"chunks": []}

    # 3. Call embedding service to generate vectors.
    # If this fails, the exception propagates — status stays at CHUNKED.
    vectors = embed_texts([c.content for c in chunks])

    if len(vectors) != len(chunks):
        raise RuntimeError(
            f"Embedding count mismatch: got {len(vectors)} for {len(chunks)} chunks"
        )

    # Build embedding rows for batch insert.
    embedding_rows = [
        {
            "chunk_id": uuid.UUID(c.chunk_id) if isinstance(c.chunk_id, str) else c.chunk_id,
            "vector": vec,
            "created_by": "ingestion-service",
        }
        for c, vec in zip(chunks, vectors, strict=True)
    ]

    # 4. Atomic transaction: delete old embeddings → insert new → update status.
    with session_scope() as session:
        # Delete existing embeddings for this document's chunks.
        delete_embeddings_for_document(session, doc_id)

        # Insert new embeddings.
        batch_insert_embeddings(session, embedding_rows)

        # Update document status to EMBEDDED and clear error.
        session.query(Document).filter(Document.doc_id == doc_id).update(
            {"status": STATUS_EMBEDDED, "error": None}
        )

    # Return updated chunks with vectors attached.
    updated = [
        c.model_copy(update={"vector": vec})
        for c, vec in zip(chunks, vectors, strict=True)
    ]
    return {"chunks": updated}
