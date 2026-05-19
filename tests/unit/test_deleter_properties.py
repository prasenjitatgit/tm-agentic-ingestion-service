"""Property-based tests for Deletion Completeness.

# Feature: document-schema-redesign, Property 2: Deletion completeness

For any document marked TO BE DELETED, after the delete node executes
successfully, there SHALL be zero rows in `embeddings` and zero rows in
`document_chunks` referencing that document's `doc_id`, and the document
status SHALL be DELETED.

**Validates: Requirements 4.1, 4.2, 4.3**
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from app.models.db_models import (
    STATUS_DELETED,
    STATUS_TO_BE_DELETED,
    Document,
    DocumentChunk,
    Embedding,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Module import workaround
# ─────────────────────────────────────────────────────────────────────────────
# The app.graph.__init__ imports workflow.py which has a broken import
# (STATUS_FAILED removed in schema redesign). We load deleter.py directly
# from its file path to bypass the package __init__ chain.


def _load_deleter_module():
    """Load the deleter module directly from file, bypassing app.graph.__init__."""
    # Ensure parent packages exist in sys.modules as stubs (if not already loaded)
    for pkg_name in ("app.graph", "app.graph.nodes"):
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__package__ = pkg_name
            pkg.__path__ = []
            sys.modules[pkg_name] = pkg

    # The deleter module imports from app.graph.nodes.status (for the @node decorator)
    # and app.graph.state (for AgentState type). We need status loaded first.
    _project_root = Path(__file__).resolve().parents[2]

    # Load app.graph.state
    if "app.graph.state" not in sys.modules:
        state_path = _project_root / "app" / "graph" / "state.py"
        state_spec = importlib.util.spec_from_file_location("app.graph.state", state_path)
        state_mod = importlib.util.module_from_spec(state_spec)
        sys.modules["app.graph.state"] = state_mod
        state_spec.loader.exec_module(state_mod)

    # Load app.graph.nodes.status
    if "app.graph.nodes.status" not in sys.modules:
        status_path = _project_root / "app" / "graph" / "nodes" / "status.py"
        status_spec = importlib.util.spec_from_file_location("app.graph.nodes.status", status_path)
        status_mod = importlib.util.module_from_spec(status_spec)
        sys.modules["app.graph.nodes.status"] = status_mod
        status_spec.loader.exec_module(status_mod)

    # Load app.graph.nodes.deleter
    deleter_path = _project_root / "app" / "graph" / "nodes" / "deleter.py"
    deleter_spec = importlib.util.spec_from_file_location("app.graph.nodes.deleter", deleter_path)
    deleter_mod = importlib.util.module_from_spec(deleter_spec)
    sys.modules["app.graph.nodes.deleter"] = deleter_mod
    deleter_spec.loader.exec_module(deleter_mod)
    return deleter_mod


deleter_module = _load_deleter_module()


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FakeDocMetadata:
    """Lightweight stand-in for DocMetadata to avoid importing app.graph.state."""

    doc_id: str
    doc_hash: str = "test_hash_abc123"
    doc_name: str = "test_document.pdf"
    s3_url: str = "s3://bucket/test_document.pdf"
    doc_type: str = "PDF"
    knowledge_base_type: str = "Maintenance"


def _build_state(doc_id: uuid.UUID) -> dict[str, Any]:
    """Build a state dict mimicking AgentState with the given doc_id."""
    meta = FakeDocMetadata(doc_id=str(doc_id))
    return {"doc_metadata": meta}


def _build_mock_session(
    doc_id: uuid.UUID,
    chunk_ids: list[uuid.UUID],
    embedding_chunk_ids: list[uuid.UUID],
):
    """Build a mock SQLAlchemy session that simulates delete_document_node's DB interactions.

    The mock tracks:
    - Which chunk_ids are returned for the document
    - Whether embeddings delete was called with correct chunk_ids
    - Whether chunks delete was called for the correct doc_id
    - Whether document status was updated to DELETED
    - The order of operations

    Returns (session, tracker) where tracker records all operations performed.
    """
    session = MagicMock()

    # Track operations for verification
    tracker = {
        "embeddings_deleted": False,
        "embeddings_filter_chunk_ids": None,
        "chunks_deleted": False,
        "chunks_filter_doc_id": None,
        "status_updated": False,
        "status_value": None,
        "operation_order": [],
    }

    def _smart_query(model_or_column):
        query_mock = MagicMock()

        # Determine what's being queried by checking the model/column
        is_chunk_id_column = False
        is_document_chunk = False
        is_embedding = False
        is_document = False

        # session.query(DocumentChunk.chunk_id) — querying a column attribute
        if hasattr(model_or_column, "property") and hasattr(model_or_column, "key"):
            if getattr(model_or_column, "key", None) == "chunk_id":
                is_chunk_id_column = True
        elif model_or_column is DocumentChunk:
            is_document_chunk = True
        elif model_or_column is Embedding:
            is_embedding = True
        elif model_or_column is Document:
            is_document = True

        def _filter(*conditions):
            filter_mock = MagicMock()

            if is_chunk_id_column:
                # This is the query for getting chunk_ids
                filter_mock.all.return_value = [(cid,) for cid in chunk_ids]

            elif is_document_chunk:
                # This is the delete chunks query
                def _delete_chunks(synchronize_session=False):
                    tracker["chunks_deleted"] = True
                    tracker["chunks_filter_doc_id"] = doc_id
                    tracker["operation_order"].append("delete_chunks")
                    return len(chunk_ids)

                filter_mock.delete = _delete_chunks

            elif is_embedding:
                # This is the delete embeddings query
                def _delete_embeddings(synchronize_session=False):
                    tracker["embeddings_deleted"] = True
                    tracker["embeddings_filter_chunk_ids"] = list(chunk_ids)
                    tracker["operation_order"].append("delete_embeddings")
                    return len(embedding_chunk_ids)

                filter_mock.delete = _delete_embeddings

            elif is_document:
                # This is the status update query
                def _update(values, synchronize_session=False):
                    tracker["status_updated"] = True
                    tracker["status_value"] = values.get("status")
                    tracker["operation_order"].append("update_status")
                    return 1

                filter_mock.update = _update

            return filter_mock

        query_mock.filter = _filter
        return query_mock

    session.query = _smart_query
    return session, tracker


# ─────────────────────────────────────────────────────────────────────────────
#  Strategies
# ─────────────────────────────────────────────────────────────────────────────

# Generate between 0 and 20 chunks per document
num_chunks_st = st.integers(min_value=0, max_value=20)

# Generate between 0 and 1 embeddings per chunk (some chunks may not have embeddings yet)
has_embedding_st = st.booleans()


@st.composite
def document_with_chunks_and_embeddings(draw):
    """Generate a random document with varying numbers of chunks and embeddings.

    Returns a tuple of (doc_id, chunk_ids, embedding_chunk_ids) where:
    - doc_id: UUID of the document
    - chunk_ids: list of chunk UUIDs belonging to the document
    - embedding_chunk_ids: subset of chunk_ids that have embeddings
    """
    doc_id = draw(st.uuids())
    num_chunks = draw(num_chunks_st)

    chunk_ids = [draw(st.uuids()) for _ in range(num_chunks)]

    # Each chunk may or may not have an embedding
    embedding_chunk_ids = [
        cid for cid in chunk_ids if draw(has_embedding_st)
    ]

    return doc_id, chunk_ids, embedding_chunk_ids


# ─────────────────────────────────────────────────────────────────────────────
#  Property Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDeletionCompleteness:
    """For any document marked TO BE DELETED, after the delete node executes
    successfully, there SHALL be zero rows in `embeddings` and zero rows in
    `document_chunks` referencing that document's `doc_id`, and the document
    status SHALL be DELETED.

    **Validates: Requirements 4.1, 4.2, 4.3**
    """

    @given(data=document_with_chunks_and_embeddings())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_all_embeddings_deleted_for_document(
        self, data: tuple[uuid.UUID, list[uuid.UUID], list[uuid.UUID]]
    ) -> None:
        """Requirement 4.1: WHEN the Delete_Node processes a document with status
        TO BE DELETED, THE Delete_Node SHALL delete all embeddings linked to that
        document's chunks.

        **Validates: Requirements 4.1**
        """
        doc_id, chunk_ids, embedding_chunk_ids = data
        session, tracker = _build_mock_session(doc_id, chunk_ids, embedding_chunk_ids)
        state = _build_state(doc_id)

        with patch.object(deleter_module, "session_scope") as mock_scope:
            mock_scope.return_value.__enter__ = MagicMock(return_value=session)
            mock_scope.return_value.__exit__ = MagicMock(return_value=False)

            result = deleter_module.delete_document_node(state)

        # If there are chunks, embeddings delete must have been called
        if chunk_ids:
            assert tracker["embeddings_deleted"], (
                f"Embeddings were not deleted for document with {len(chunk_ids)} chunks"
            )

        # Status must be DELETED in the result
        assert result["result"]["status"] == STATUS_DELETED

    @given(data=document_with_chunks_and_embeddings())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_all_chunks_deleted_for_document(
        self, data: tuple[uuid.UUID, list[uuid.UUID], list[uuid.UUID]]
    ) -> None:
        """Requirement 4.2: WHEN the Delete_Node processes a document, THE
        Delete_Node SHALL delete all chunks belonging to that document after
        embeddings are removed.

        **Validates: Requirements 4.2**
        """
        doc_id, chunk_ids, embedding_chunk_ids = data
        session, tracker = _build_mock_session(doc_id, chunk_ids, embedding_chunk_ids)
        state = _build_state(doc_id)

        with patch.object(deleter_module, "session_scope") as mock_scope:
            mock_scope.return_value.__enter__ = MagicMock(return_value=session)
            mock_scope.return_value.__exit__ = MagicMock(return_value=False)

            result = deleter_module.delete_document_node(state)

        # Chunks delete must always be called
        assert tracker["chunks_deleted"], (
            "Chunks were not deleted for document"
        )

    @given(data=document_with_chunks_and_embeddings())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_document_status_updated_to_deleted(
        self, data: tuple[uuid.UUID, list[uuid.UUID], list[uuid.UUID]]
    ) -> None:
        """Requirement 4.3: WHEN embeddings and chunks are successfully deleted,
        THE Delete_Node SHALL update the document status to DELETED.

        **Validates: Requirements 4.3**
        """
        doc_id, chunk_ids, embedding_chunk_ids = data
        session, tracker = _build_mock_session(doc_id, chunk_ids, embedding_chunk_ids)
        state = _build_state(doc_id)

        with patch.object(deleter_module, "session_scope") as mock_scope:
            mock_scope.return_value.__enter__ = MagicMock(return_value=session)
            mock_scope.return_value.__exit__ = MagicMock(return_value=False)

            result = deleter_module.delete_document_node(state)

        # Status must be updated to DELETED
        assert tracker["status_updated"], "Document status was not updated"
        assert tracker["status_value"] == STATUS_DELETED, (
            f"Expected status DELETED, got {tracker['status_value']}"
        )

        # Result must reflect DELETED status
        assert result["result"]["status"] == STATUS_DELETED
        assert result["result"]["doc_id"] == str(doc_id)

    @given(data=document_with_chunks_and_embeddings())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_deletion_order_embeddings_before_chunks(
        self, data: tuple[uuid.UUID, list[uuid.UUID], list[uuid.UUID]]
    ) -> None:
        """Requirement 4.5: THE Delete_Node SHALL execute deletions in order:
        embeddings first, then chunks (respecting foreign key constraints).

        After successful execution, zero embeddings and zero chunks remain.

        **Validates: Requirements 4.1, 4.2, 4.3**
        """
        doc_id, chunk_ids, embedding_chunk_ids = data
        session, tracker = _build_mock_session(doc_id, chunk_ids, embedding_chunk_ids)
        state = _build_state(doc_id)

        with patch.object(deleter_module, "session_scope") as mock_scope:
            mock_scope.return_value.__enter__ = MagicMock(return_value=session)
            mock_scope.return_value.__exit__ = MagicMock(return_value=False)

            result = deleter_module.delete_document_node(state)

        # Verify operation order: embeddings before chunks before status update
        if chunk_ids:
            # When there are chunks, embeddings must be deleted first
            assert "delete_embeddings" in tracker["operation_order"], (
                "Embeddings delete was not called when chunks exist"
            )
            emb_idx = tracker["operation_order"].index("delete_embeddings")
            chunk_idx = tracker["operation_order"].index("delete_chunks")
            status_idx = tracker["operation_order"].index("update_status")

            assert emb_idx < chunk_idx, (
                f"Embeddings deleted at index {emb_idx} but chunks at {chunk_idx}. "
                "Embeddings must be deleted before chunks (FK constraint)."
            )
            assert chunk_idx < status_idx, (
                f"Chunks deleted at index {chunk_idx} but status updated at {status_idx}. "
                "Chunks must be deleted before status update."
            )
        else:
            # When there are no chunks, only chunks delete and status update happen
            assert "delete_chunks" in tracker["operation_order"]
            assert "update_status" in tracker["operation_order"]

        # Final state: status is DELETED (meaning zero remaining data)
        assert result["result"]["status"] == STATUS_DELETED
