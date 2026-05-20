"""`fetch_metadata` node — atomically claim the next eligible document by priority."""

from __future__ import annotations

from typing import Any

from sqlalchemy import case, cast, or_, select
from sqlalchemy.orm import Session
from sqlalchemy.types import Integer as SAInteger

from app.graph.nodes.status import node
from app.graph.state import ALLOWED_STATUSES, AgentState, DocMetadata
from app.models.db_models import (
    STATUS_CHUNKED,
    STATUS_EMBEDDED,
    STATUS_TO_BE_DELETED,
    STATUS_TO_BE_INGESTED,
    Document,
)
from app.services.db_service import session_scope


def _claim_next_document(
    session: Session, kb_filter: str | None, max_retries: int
) -> Document | None:
    """Claim the next eligible document by priority: EMBEDDED > CHUNKED > TO BE INGESTED.

    Excludes documents whose error->'retry_count' >= max_retries.
    Uses FOR UPDATE SKIP LOCKED for concurrent safety.
    Treats NULL error or missing retry_count key as retry_count of 0.
    Claims exactly one document per invocation.
    """
    # Priority ordering: EMBEDDED (1) > CHUNKED (2) > TO BE INGESTED (3)
    status_priority = case(
        (Document.status == STATUS_EMBEDDED, 1),
        (Document.status == STATUS_CHUNKED, 2),
        (Document.status == STATUS_TO_BE_INGESTED, 3),
    )

    # Dead letter guard: exclude documents with retry_count >= max_retries
    # Treat NULL error or missing retry_count key as retry_count of 0 (eligible)
    retry_count_expr = cast(
        Document.error["retry_count"].as_string(), SAInteger
    )
    dead_letter_filter = or_(
        Document.error.is_(None),
        Document.error["retry_count"].is_(None),
        retry_count_expr < max_retries,
    )

    stmt = (
        select(Document)
        .where(
            Document.status.in_([STATUS_EMBEDDED, STATUS_CHUNKED, STATUS_TO_BE_INGESTED]),
            dead_letter_filter,
        )
    )

    if kb_filter:
        stmt = stmt.where(Document.knowledge_base_type == kb_filter)

    stmt = (
        stmt.order_by(status_priority, Document.created_at.asc())
        .limit(1)
        .with_for_update(skip_locked=True)
    )

    row = session.execute(stmt).scalar_one_or_none()
    return row


def _claim_deletion(session: Session) -> Document | None:
    """Atomically claim the next TO BE DELETED document using FOR UPDATE SKIP LOCKED."""
    stmt = (
        select(Document)
        .where(Document.status == STATUS_TO_BE_DELETED)
        .order_by(Document.created_at.asc())
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    row = session.execute(stmt).scalar_one_or_none()
    return row


@node("fetch_metadata")
def fetch_metadata_node(state: AgentState) -> dict[str, Any]:
    """Claim the next eligible document by priority and hydrate state."""
    from app.core.config import get_settings

    settings = get_settings()
    request_filter = state.get("request_filter") or {}
    kb_filter = request_filter.get("knowledge_base_type")

    with session_scope() as session:
        row = _claim_next_document(session, kb_filter, settings.max_document_retries)
        if row is None:
            return {"doc_metadata": None}

        # If the status value is not in ALLOWED_STATUSES, treat as unclaimed
        if row.status not in ALLOWED_STATUSES:
            return {"doc_metadata": None}

        meta = DocMetadata(
            doc_id=str(row.doc_id),
            doc_hash=row.doc_hash,
            doc_name=row.doc_name,
            s3_url=row.s3_url,
            doc_type=row.doc_type,
            knowledge_base_type=row.knowledge_base_type,
            status=row.status,
        )

    return {"doc_metadata": meta}


@node("fetch_deletion")
def fetch_deletion_node(state: AgentState) -> dict[str, Any]:
    """Claim the next TO BE DELETED document atomically and hydrate state."""
    with session_scope() as session:
        row = _claim_deletion(session)
        if row is None:
            return {"doc_metadata": None}
        meta = DocMetadata(
            doc_id=str(row.doc_id),
            doc_hash=row.doc_hash,
            doc_name=row.doc_name,
            s3_url=row.s3_url,
            doc_type=row.doc_type,
            knowledge_base_type=row.knowledge_base_type,
        )

    return {"doc_metadata": meta}
