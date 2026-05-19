"""Delete-document graph node.

Handles cleanup of documents marked for deletion by removing associated
embeddings and chunks, then updating the document status to DELETED.

All operations execute within a single transaction for atomicity — if any
step fails, the transaction rolls back and the document remains in
TO BE DELETED status for retry.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func

from app.graph.nodes.status import node
from app.graph.state import AgentState
from app.models.db_models import (
    STATUS_DELETED,
    Document,
    DocumentChunk,
    Embedding,
)
from app.services.db_service import session_scope
from app.utils.logger import get_logger

log = get_logger(__name__)


@node("delete_document")
def delete_document_node(state: AgentState) -> dict[str, Any]:
    """Delete embeddings and chunks for a document marked TO BE DELETED.

    Steps (executed in a single transaction):
        1. Query all chunk_ids belonging to the document
        2. Delete all embeddings referencing those chunk_ids
        3. Delete all chunks for the document
        4. Update document status to DELETED

    Returns:
        State delta with result containing doc_id and status DELETED.
    """
    meta = state["doc_metadata"]
    doc_id = meta.doc_id

    with session_scope() as session:
        # Step 1: Get all chunk_ids for this document
        chunk_ids = (
            session.query(DocumentChunk.chunk_id)
            .filter(DocumentChunk.doc_id == doc_id)
            .all()
        )
        chunk_id_list = [cid for (cid,) in chunk_ids]

        # Step 2: Delete embeddings (must happen before chunks due to FK)
        if chunk_id_list:
            session.query(Embedding).filter(
                Embedding.chunk_id.in_(chunk_id_list)
            ).delete(synchronize_session=False)

        # Step 3: Delete chunks
        session.query(DocumentChunk).filter(
            DocumentChunk.doc_id == doc_id
        ).delete(synchronize_session=False)

        # Step 4: Update document status to DELETED
        session.query(Document).filter(
            Document.doc_id == doc_id
        ).update(
            {"status": STATUS_DELETED, "updated_at": func.now()},
            synchronize_session=False,
        )

    log.info(
        "document_deleted",
        doc_id=doc_id,
        chunks_removed=len(chunk_id_list),
    )

    return {"result": {"doc_id": doc_id, "status": STATUS_DELETED}}
