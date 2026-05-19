"""Property-based tests for Batch Insert Count Invariant.

# Feature: document-schema-redesign, Property 3: Batch insert count invariant

For any list of N chunks passed to batch_insert_chunks, exactly N rows SHALL be
inserted into `document_chunks` and exactly N chunk_ids SHALL be returned.

**Validates: Requirement 8.4**
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from app.services.batch_ops import batch_insert_chunks


# ─────────────────────────────────────────────────────────────────────────────
#  Strategies
# ─────────────────────────────────────────────────────────────────────────────

# Non-empty chunk content
chunk_content_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Zs")),
    min_size=1,
    max_size=200,
).filter(lambda s: s.strip())

# Optional page number
page_no_st = st.one_of(st.none(), st.integers(min_value=1, max_value=500))

# Optional created_by
created_by_st = st.one_of(
    st.none(),
    st.text(
        alphabet=st.characters(whitelist_categories=("L", "N")),
        min_size=3,
        max_size=30,
    ).filter(lambda s: s.strip()),
)


# Strategy for a single chunk dict
chunk_st = st.builds(
    lambda content, page_no, created_by: {
        k: v
        for k, v in {
            "content": content,
            "page_no": page_no,
            "created_by": created_by,
        }.items()
        if v is not None
    },
    content=chunk_content_st,
    page_no=page_no_st,
    created_by=created_by_st,
)

# Strategy for a list of chunks (1 to 50 chunks)
chunks_list_st = st.lists(chunk_st, min_size=1, max_size=50)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _build_mock_session() -> MagicMock:
    """Build a mock SQLAlchemy session that accepts execute() calls.

    Since batch_insert_chunks only calls session.execute(stmt), we just need
    a mock that doesn't raise on execute.
    """
    session = MagicMock()
    session.execute = MagicMock(return_value=None)
    return session


# ─────────────────────────────────────────────────────────────────────────────
#  Property Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBatchInsertCountInvariant:
    """For any list of N chunks passed to batch_insert_chunks, exactly N
    chunk_ids SHALL be returned.

    **Validates: Requirement 8.4**
    """

    @given(chunks=chunks_list_st)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_returns_exactly_n_chunk_ids(self, chunks: list[dict]) -> None:
        """WHEN batch_insert_chunks is called with N chunks, THE function SHALL
        return exactly N chunk_ids.

        **Validates: Requirement 8.4**
        """
        # Arrange
        session = _build_mock_session()
        doc_id = uuid.uuid4()
        n = len(chunks)

        # Act
        result = batch_insert_chunks(session, doc_id, chunks)

        # Assert: exactly N chunk_ids returned
        assert len(result) == n, (
            f"Expected {n} chunk_ids, got {len(result)}"
        )

    @given(chunks=chunks_list_st)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_all_returned_ids_are_unique(self, chunks: list[dict]) -> None:
        """All chunk_ids returned by batch_insert_chunks SHALL be unique UUIDs.

        **Validates: Requirement 8.4**
        """
        # Arrange
        session = _build_mock_session()
        doc_id = uuid.uuid4()

        # Act
        result = batch_insert_chunks(session, doc_id, chunks)

        # Assert: all IDs are unique
        assert len(set(result)) == len(result), (
            "Returned chunk_ids contain duplicates"
        )

    @given(chunks=chunks_list_st)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_all_returned_ids_are_valid_uuids(self, chunks: list[dict]) -> None:
        """All returned values SHALL be valid UUID instances.

        **Validates: Requirement 8.4**
        """
        # Arrange
        session = _build_mock_session()
        doc_id = uuid.uuid4()

        # Act
        result = batch_insert_chunks(session, doc_id, chunks)

        # Assert: all returned values are UUID instances
        for chunk_id in result:
            assert isinstance(chunk_id, uuid.UUID), (
                f"Expected UUID, got {type(chunk_id)}: {chunk_id}"
            )

    @given(chunks=chunks_list_st)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_session_execute_called_once(self, chunks: list[dict]) -> None:
        """batch_insert_chunks SHALL execute a single batch INSERT statement
        (session.execute called exactly once).

        **Validates: Requirement 8.4**
        """
        # Arrange
        session = _build_mock_session()
        doc_id = uuid.uuid4()

        # Act
        batch_insert_chunks(session, doc_id, chunks)

        # Assert: single batch insert (one execute call)
        assert session.execute.call_count == 1, (
            f"Expected 1 execute call, got {session.execute.call_count}"
        )
