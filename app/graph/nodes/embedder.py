"""`embed_chunks` node — batch-embed and persist vectors."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.graph.nodes.status import node
from app.graph.state import AgentState
from app.models.db_models import Embedding
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

    rows = [
        {
            "embedding_id": uuid.uuid4(),
            "chunk_id": uuid.UUID(c.chunk_id) if isinstance(c.chunk_id, str) else c.chunk_id,
            "vector": vec,
        }
        for c, vec in zip(chunks, vectors, strict=True)
    ]
    with session_scope() as session:
        stmt = pg_insert(Embedding).values(rows)
        stmt = stmt.on_conflict_do_nothing(index_elements=["chunk_id"])
        session.execute(stmt)

    updated = [
        c.model_copy(update={"embedding_id": str(r["embedding_id"]), "vector": r["vector"]})
        for c, r in zip(chunks, rows, strict=True)
    ]
    return {"chunks": updated}
