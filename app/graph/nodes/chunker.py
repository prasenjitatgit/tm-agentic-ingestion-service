"""`generate_chunks` node — Markdown-aware chunking via MarkdownChunker.

Splits assembled (or raw) text blocks into chunks using the MarkdownChunker,
which respects heading boundaries, protected blocks (tables, images, speaker
notes), and applies overlap between adjacent chunks.

The node is idempotent: it deletes any existing chunks and embeddings for the
document before inserting new chunks, and advances the document status to
CHUNKED within the same atomic transaction.
"""

from __future__ import annotations

import uuid
from typing import Any

from app.core.config import get_settings
from app.graph.nodes.status import node
from app.graph.state import AgentState, AssembledText, Chunk
from app.models.db_models import Document, STATUS_CHUNKED
from app.services.batch_ops import batch_insert_chunks, delete_chunks_and_embeddings_for_document
from app.services.db_service import session_scope


# ─────────────────────────────────────────────────────────────────────────────
#  NODE
# ─────────────────────────────────────────────────────────────────────────────


@node("generate_chunks")
def generate_chunks_node(state: AgentState) -> dict[str, Any]:
    """Idempotent chunker: cleanup → insert → status update in one transaction.

    Within a single session_scope() transaction:
    1. Delete all embeddings for the document's chunks (cascading cleanup).
    2. Delete all existing chunks for the document.
    3. Insert new chunks.
    4. Update document status to CHUNKED and set error=NULL.

    If zero chunks are produced, returns {"chunks": []} without modifying the
    database. On transaction failure, all changes are rolled back automatically.
    """
    meta = state["doc_metadata"]

    blocks = list(state.get("assembled_texts") or [])
    if not blocks:
        # No-image branch: derive AssembledText from raw text blocks.
        for tb in state.get("extracted_text_blocks") or []:
            blocks.append(AssembledText(section=tb.section, page=tb.page, text=tb.text))

    settings = get_settings()

    from app.graph.nodes.markdown_chunker import MarkdownChunker
    from app.services.llm_service import embed_texts

    chunker = MarkdownChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_table_size=settings.max_table_size,
        similarity_threshold=settings.semantic_similarity_threshold,
        embed_fn=embed_texts,
    )

    chunk_dicts: list[dict[str, Any]] = []

    for block in blocks:
        pieces = chunker.chunk(block.text)
        for piece in pieces:
            chunk_dicts.append(
                {
                    "content": piece,
                    "page_no": block.page,
                    "created_by": "ingestion-service",
                }
            )

    # If zero chunks are produced, skip insertion and leave status unchanged.
    if not chunk_dicts:
        return {"chunks": []}

    doc_id = uuid.UUID(meta.doc_id)

    # Atomic transaction: cleanup → insert → status update.
    with session_scope() as session:
        # 1 & 2. Delete existing embeddings and chunks for this document.
        delete_chunks_and_embeddings_for_document(session, doc_id)

        # 3. Insert new chunks.
        chunk_ids = batch_insert_chunks(session, doc_id=doc_id, chunks=chunk_dicts)

        # 4. Update document status to CHUNKED and clear error.
        session.query(Document).filter(Document.doc_id == doc_id).update(
            {"status": STATUS_CHUNKED, "error": None}
        )

    # Build simplified Chunk state objects.
    state_chunks: list[Chunk] = [
        Chunk(
            chunk_id=str(chunk_id),
            page_no=chunk_dict["page_no"],
            content=chunk_dict["content"],
        )
        for chunk_id, chunk_dict in zip(chunk_ids, chunk_dicts)
    ]

    return {"chunks": state_chunks}
