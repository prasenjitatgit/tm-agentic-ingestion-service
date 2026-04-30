"""`generate_chunks` node + tag-aware recursive splitter.

Hard constraint (per spec):
    The splitter must NEVER break the following structural tags across two
    chunks: IMAGE_REFERENCE, TABLE_REFERENCE, SPEAKER_NOTES.

Strategy: tokenize the input into atomic *spans* — each protected tag is
one indivisible unit, plain text is split into sentence-ish pieces — then
greedily pack spans into chunks of `chunk_size` characters with
`chunk_overlap` overlap. A protected span larger than `chunk_size` is
emitted whole as one oversized chunk.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.core.config import get_settings
from app.graph.nodes.status import node
from app.graph.state import AgentState, AssembledText, Chunk
from app.models.db_models import DocumentChunk
from app.services.db_service import session_scope

_settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
#  TAG-AWARE SPLITTER
# ─────────────────────────────────────────────────────────────────────────────

PROTECTED_TAGS: tuple[str, ...] = ("IMAGE_REFERENCE", "TABLE_REFERENCE", "SPEAKER_NOTES")

# Matches an entire <TAG ...>...</TAG> block (non-greedy, dotall).
_TAG_PATTERN = re.compile(
    r"<(?P<tag>" + "|".join(PROTECTED_TAGS) + r")(?:\s[^>]*)?>.*?</(?P=tag)>",
    flags=re.DOTALL,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class _Span:
    """An atomic chunkable unit. `protected=True` means it must stay whole."""

    text: str
    protected: bool


def _atomize(text: str) -> list[_Span]:
    """Tokenize `text` into protected and unprotected spans, preserving order."""
    spans: list[_Span] = []
    cursor = 0
    for match in _TAG_PATTERN.finditer(text):
        if match.start() > cursor:
            for piece in _split_plain(text[cursor : match.start()]):
                if piece:
                    spans.append(_Span(piece, protected=False))
        spans.append(_Span(match.group(0), protected=True))
        cursor = match.end()
    if cursor < len(text):
        for piece in _split_plain(text[cursor:]):
            if piece:
                spans.append(_Span(piece, protected=False))
    return spans


def _split_plain(text: str) -> list[str]:
    """Split unprotected text into sentence-ish units for finer packing."""
    text = text.strip()
    if not text:
        return []
    return [p.strip() for p in _SENTENCE_SPLIT.split(text) if p.strip()]


def _pack(spans: list[_Span], chunk_size: int, overlap: int) -> list[str]:
    """Greedy pack spans into chunks bounded by `chunk_size` with `overlap`."""
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        chunks.append(" ".join(buf).strip())
        if overlap <= 0:
            buf = []
            buf_len = 0
            return
        tail: list[str] = []
        tail_len = 0
        for piece in reversed(buf):
            if tail_len + len(piece) + 1 > overlap:
                break
            tail.insert(0, piece)
            tail_len += len(piece) + 1
        buf = tail
        buf_len = sum(len(p) + 1 for p in buf)

    for span in spans:
        piece = span.text

        if span.protected and len(piece) > chunk_size:
            # Oversized protected block: flush current buf and emit it whole.
            flush()
            if buf:
                chunks.append(" ".join(buf).strip())
                buf, buf_len = [], 0
            chunks.append(piece)
            continue

        if buf_len + len(piece) + 1 > chunk_size and buf:
            flush()

        buf.append(piece)
        buf_len += len(piece) + 1

    if buf:
        chunks.append(" ".join(buf).strip())

    return [c for c in chunks if c]


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Public entry point: tag-aware recursive chunker."""
    if not text or not text.strip():
        return []
    return _pack(_atomize(text), chunk_size=chunk_size, overlap=chunk_overlap)


def normalize(text: str) -> str:
    """Cheap normalization for hashing / simhash (NFKC, lower, collapse ws)."""
    text = unicodedata.normalize("NFKC", text).lower()
    return _WHITESPACE.sub(" ", text).strip()


def chunk_hash(content: str) -> str:
    """SHA-256 over normalized content (idempotency key)."""
    return hashlib.sha256(normalize(content).encode("utf-8")).hexdigest()


def chunk_simhash(content: str) -> int:
    """64-bit positive simhash (fits PostgreSQL BIGINT)."""
    from simhash import Simhash

    tokens = normalize(content).split()
    return int(Simhash(tokens).value & 0x7FFFFFFFFFFFFFFF)


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

    for block in blocks:
        for piece in split_text(
            block.text, chunk_size=_settings.chunk_size, chunk_overlap=_settings.chunk_overlap
        ):
            normalized = normalize(piece)
            h = chunk_hash(piece)
            sh = chunk_simhash(piece)
            chunk_id = uuid.uuid4()
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": uuid.UUID(meta.doc_id),
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
                DocumentChunk.doc_id == uuid.UUID(meta.doc_id),
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
