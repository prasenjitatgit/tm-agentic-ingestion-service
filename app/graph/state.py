"""LangGraph state schema (TypedDict + Pydantic value objects).

All structures carried through `StateGraph` are defined here so any node
module can import them without creating cycles back to `workflow.py`.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict

from pydantic import BaseModel

from app.core.config import get_settings

DocType = Literal["PDF", "EXCEL", "PPT", "DOCX"]
KnowledgeBaseType = Literal["Maintenance", "Construction", "BusinessIntelligence"]


class DocMetadata(BaseModel):
    """Stable view of a `documents` row carried through the graph."""

    doc_id: str  # UUID serialized as string for state compatibility
    doc_hash: str
    doc_name: str
    s3_url: str
    doc_type: DocType
    knowledge_base_type: KnowledgeBaseType


class ImageItem(BaseModel):
    """Image extracted by a parser (summary populated later by VLM)."""

    page: int
    image_index: int
    s3_url: str
    summary: str = ""


class TextBlock(BaseModel):
    """Raw text block produced by a parser."""

    page: int
    section: Optional[str] = None
    text: str


class AssembledText(BaseModel):
    """Text block after image summaries have been spliced in."""

    section: Optional[str] = None
    page: int
    text: str


class Chunk(BaseModel):
    """A single chunk ready for embedding / persistence."""

    chunk_id: Optional[str] = None
    page_no: Optional[int] = None
    content: str
    embedding_id: Optional[str] = None
    vector: Optional[list[float]] = None


class ErrorInfo(BaseModel):
    """Captured exception details for routing/observability."""

    type: str
    message: str
    traceback: Optional[str] = None


class RetryContext(BaseModel):
    """Centralized retry state."""

    retry_count: int = 0
    max_retries: int = 3
    failed_node: Optional[str] = None
    is_retryable: bool = False


class AgentState(TypedDict, total=False):
    """LangGraph state schema (TypedDict per LangGraph convention)."""

    # Inputs
    request_filter: dict[str, Any]
    correlation_id: str

    # Document context
    doc_metadata: DocMetadata
    local_path: str

    # Buffers
    extracted_image_items: list[ImageItem]
    extracted_text_blocks: list[TextBlock]
    assembled_texts: list[AssembledText]
    chunks: list[Chunk]

    # Resilience
    error: Optional[ErrorInfo]
    retry_context: RetryContext

    # Output summary
    result: dict[str, Any]


def initial_state(request_filter: dict[str, Any], correlation_id: str) -> AgentState:
    """Build a fresh state object for one document run."""
    settings = get_settings()
    return AgentState(
        request_filter=request_filter,
        correlation_id=correlation_id,
        extracted_image_items=[],
        extracted_text_blocks=[],
        assembled_texts=[],
        chunks=[],
        error=None,
        retry_context=RetryContext(retry_count=0, max_retries=settings.max_retries),
        result={},
    )
