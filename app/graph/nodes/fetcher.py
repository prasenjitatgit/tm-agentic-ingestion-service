"""`fetch_metadata` node — atomically claim the next NEW document."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.graph.nodes.status import node
from app.graph.state import AgentState, DocMetadata
from app.models.db_models import STATUS_NEW, STATUS_PROCESSING, Document
from app.services.db_service import session_scope


def _claim_pending(session: Session, kb_filter: str | None) -> Document | None:
    """Atomically claim the next NEW document and mark it PROCESSING."""
    stmt = select(Document).where(Document.status == STATUS_NEW)
    if kb_filter:
        stmt = stmt.where(Document.knowledge_base_type == kb_filter)
    stmt = stmt.order_by(Document.created_at.asc()).limit(1).with_for_update(skip_locked=True)

    row = session.execute(stmt).scalar_one_or_none()
    if row is None:
        return None
    row.status = STATUS_PROCESSING
    row.updated_by = "ingestion-service"
    session.add(row)
    return row


@node("fetch_metadata")
def fetch_metadata_node(state: AgentState) -> dict[str, Any]:
    """Claim the next NEW document atomically and hydrate state."""
    request_filter = state.get("request_filter") or {}
    kb_filter = request_filter.get("knowledge_base_type")

    with session_scope() as session:
        row = _claim_pending(session, kb_filter)
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
