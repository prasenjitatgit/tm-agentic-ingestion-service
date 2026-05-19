"""`generate_chunks` node — Markdown-aware chunking via MarkdownChunker.

Splits assembled (or raw) text blocks into chunks using the MarkdownChunker,
which respects heading boundaries, protected blocks (tables, images, speaker
notes), and applies overlap between adjacent chunks.
"""

from __future__ import annotations

import uuid
from typing import Any

from app.core.config import get_settings
from app.graph.nodes.status import node
from app.graph.state import AgentState, AssembledText, Chunk
from app.services.batch_ops import batch_insert_chunks
from app.services.db_service import session_scope


# ─────────────────────────────────────────────────────────────────────────────
#  NODE
# ─────────────────────────────────────────────────────────────────────────────


@node("generate_chunks")
def generate_chunks_node(state: AgentState) -> dict[str, Any]:
    """Split assembled (or raw) text into chunks and persist via batch insert."""
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

    if not chunk_dicts:
        return {"chunks": []}

    # Persist all chunks in a single batch INSERT.
    with session_scope() as session:
        chunk_ids = batch_insert_chunks(
            session, doc_id=uuid.UUID(meta.doc_id), chunks=chunk_dicts
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
