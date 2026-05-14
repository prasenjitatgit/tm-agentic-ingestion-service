"""`generate_chunks` node — Markdown-aware chunking via MarkdownChunker.

Splits assembled (or raw) text blocks into chunks using the MarkdownChunker,
which respects heading boundaries, protected blocks (tables, images, speaker
notes), and applies overlap between adjacent chunks.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid
from typing import Any

from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.core.config import get_settings
from app.graph.nodes.status import node
from app.graph.state import AgentState, AssembledText, Chunk
from app.models.db_models import DocumentChunk
from app.services.db_service import session_scope

_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Cheap normalization for hashing / simhash (NFKC, lower, collapse ws)."""
    text = unicodedata.normalize("NFKC", text).lower()
    return _WHITESPACE.sub(" ", text).strip()


def chunk_hash(content: str) -> str:
    """SHA-256 over normalized content (idempotency key)."""
    return hashlib.sha256(normalize(content).encode("utf-8")).hexdigest()


def chunk_simhash(content: str) -> int:
    """64-bit positive simhash (fits PostgreSQL BIGINT).

    For table chunks, uses data-row content with positional tokens to ensure
    chunks with the same column headers but different data produce distinct
    simhash values.
    """
    from simhash import Simhash

    text = _extract_simhash_tokens(content)
    tokens = normalize(text).split()
    if not tokens:
        tokens = normalize(content).split()
    return int(Simhash(tokens).value & 0x7FFFFFFFFFFFFFFF)


# Patterns for extracting meaningful tokens for simhash
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_TABLE_SEPARATOR_RE = re.compile(r"^\|[\s\-:|]+\|$", re.MULTILINE)


def _extract_simhash_tokens(content: str) -> str:
    """Extract tokens that differentiate this chunk from others.

    For table content:
    - Strips HTML comment markers and separator rows
    - Strips the header row (first pipe-delimited row)
    - Adds row-index prefixes to data cells so identical values in different
      positions produce different tokens (e.g., "r1:germany" vs "r3:germany")

    For non-table content:
    - Returns the content as-is (no transformation needed)
    """
    # Check if this is a table chunk
    if "<!-- TABLE_BLOCK -->" not in content and not content.strip().startswith("|"):
        return content

    # Remove HTML comments
    text = _HTML_COMMENT_RE.sub("", content)

    # Remove separator rows
    text = _TABLE_SEPARATOR_RE.sub("", text)

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # Find pipe-delimited rows
    table_rows = [line for line in lines if line.startswith("|")]
    non_table_text = [line for line in lines if not line.startswith("|")]

    if not table_rows:
        return content

    # Skip the first row (header — repeated in every sub-table)
    data_rows = table_rows[1:] if len(table_rows) > 1 else table_rows

    # Build positional tokens: prefix each cell value with its row index
    positional_tokens: list[str] = []
    for row_idx, row in enumerate(data_rows):
        cells = [c.strip() for c in row.strip("|").split("|")]
        for col_idx, cell in enumerate(cells):
            if cell:
                positional_tokens.append(f"r{row_idx}c{col_idx}:{cell}")

    # Combine non-table text with positional table tokens
    parts = non_table_text + positional_tokens
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
#  NODE
# ─────────────────────────────────────────────────────────────────────────────


@node("generate_chunks")
def generate_chunks_node(state: AgentState) -> dict[str, Any]:
    """Split assembled (or raw) text into chunks and persist with idempotency."""
    meta = state["doc_metadata"]

    blocks = list(state.get("assembled_texts") or [])
    if not blocks:
        # No-image branch: derive AssembledText from raw text blocks.
        for tb in state.get("extracted_text_blocks") or []:
            blocks.append(AssembledText(section=tb.section, page=tb.page, text=tb.text))

    rows: list[dict[str, Any]] = []
    state_chunks: list[Chunk] = []

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

    for block in blocks:
        pieces = chunker.chunk(block.text)

        for piece in pieces:
            normalized = normalize(piece)
            h = chunk_hash(piece)
            sh = chunk_simhash(piece)
            chunk_id = uuid.uuid4()
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": meta.doc_id,
                    "content": piece,
                    "normalized_content": normalized,
                    "chunk_hash": h,
                    "chunk_simhash": sh,
                    "section": block.section,
                    "page": block.page,
                }
            )
            state_chunks.append(
                Chunk(
                    chunk_id=str(chunk_id),
                    chunk_hash=h,
                    chunk_simhash=sh,
                    section=block.section,
                    page_no=block.page,
                    content=piece,
                    normalized_content=normalized,
                )
            )

    if not rows:
        return {"chunks": []}

    with session_scope() as session:
        stmt = pg_insert(DocumentChunk).values(rows)
        stmt = stmt.on_conflict_do_nothing(index_elements=["doc_id", "chunk_hash"])
        session.execute(stmt)

    # Reconcile state.chunks with the actual DB row ids (handles dedupes).
    with session_scope() as session:
        existing = {
            row.chunk_hash: row.chunk_id
            for row in session.query(DocumentChunk)
            .filter(
                DocumentChunk.doc_id == meta.doc_id,
                DocumentChunk.chunk_hash.in_([c.chunk_hash for c in state_chunks]),
            )
            .all()
        }
    reconciled = [
        c.model_copy(update={"chunk_id": str(existing[c.chunk_hash])})
        for c in state_chunks
        if c.chunk_hash in existing
    ]
    return {"chunks": reconciled}
