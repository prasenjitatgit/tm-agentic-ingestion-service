"""`embed_chunks` node — batch-embed and persist vectors."""

from __future__ import annotations

import uuid
from typing import Any

from app.graph.nodes.status import node
from app.graph.state import AgentState
from app.services.batch_ops import batch_insert_embeddings
from app.services.db_service import session_scope
from app.services.llm_service import embed_texts


@node("embed_chunks")
def embed_chunks_node(state: AgentState) -> dict[str, Any]:
    """Embed each chunk in batches and persist to the embeddings table."""
    chunks = state.get("chunks") or []
    if not chunks:
        return {"chunks": []}

    vectors = embed_texts([c.content for c in chunks])
    if len(vectors) != len(chunks):
        raise RuntimeError(
            f"Embedding count mismatch: got {len(vectors)} for {len(chunks)} chunks"
        )

    embedding_rows = [
        {
            "chunk_id": uuid.UUID(c.chunk_id) if isinstance(c.chunk_id, str) else c.chunk_id,
            "vector": vec,
            "created_by": "ingestion-service",
        }
        for c, vec in zip(chunks, vectors, strict=True)
    ]

    with session_scope() as session:
        batch_insert_embeddings(session, embedding_rows)

    updated = [
        c.model_copy(update={"vector": vec})
        for c, vec in zip(chunks, vectors, strict=True)
    ]
    return {"chunks": updated}
