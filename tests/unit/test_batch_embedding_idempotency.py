"""Property-based tests for Batch Embedding Insert Idempotency.

# Feature: document-schema-redesign, Property 7: Batch embedding insert idempotency

For any set of embeddings where some chunk_ids already have embeddings, calling
batch_insert_embeddings SHALL not raise an error and SHALL silently skip
duplicates while inserting new ones.

**Validates: Requirement 8.3**
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from app.services.batch_ops import batch_insert_embeddings


# ─────────────────────────────────────────────────────────────────────────────
#  Strategies
# ─────────────────────────────────────────────────────────────────────────────

# Strategy for a 1024-dimensional vector.
# We use a fixed base vector and vary only a few elements for performance,
# since the property under test is idempotency (not vector content).
@st.composite
def vector_st(draw):
    """Generate a 1024-dim vector efficiently for property testing."""
    base_value = draw(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    return [base_value] * 1024


# Strategy for a chunk_id (UUID)
chunk_id_st = st.uuids()

# Optional created_by
created_by_st = st.one_of(
    st.none(),
    st.text(
        alphabet=st.characters(whitelist_categories=("L", "N")),
        min_size=3,
        max_size=30,
    ).filter(lambda s: s.strip()),
)


# Strategy for a single embedding dict
@st.composite
def embedding_st(draw):
    """Generate a single embedding dict with chunk_id, vector, and optional created_by."""
    chunk_id = draw(chunk_id_st)
    vector = draw(vector_st())
    created_by = draw(created_by_st)
    result = {"chunk_id": chunk_id, "vector": vector}
    if created_by is not None:
        result["created_by"] = created_by
    return result


# Strategy for a list of embeddings (1 to 10 embeddings)
embeddings_list_st = st.lists(embedding_st(), min_size=1, max_size=10)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _build_mock_session() -> MagicMock:
    """Build a mock SQLAlchemy session that accepts execute() calls.

    Since batch_insert_embeddings only calls session.execute(stmt), we just
    need a mock that doesn't raise on execute.
    """
    session = MagicMock()
    session.execute = MagicMock(return_value=None)
    return session


def _create_embeddings_with_duplicates(
    unique_embeddings: list[dict], duplicate_indices: list[int]
) -> list[dict]:
    """Create an embedding list that contains duplicates of certain chunk_ids.

    Takes a list of unique embeddings and a list of indices to duplicate.
    Returns a new list with duplicated chunk_ids appended.
    """
    result = list(unique_embeddings)
    for idx in duplicate_indices:
        if idx < len(unique_embeddings):
            # Create a new embedding dict with the same chunk_id
            dup = {
                "chunk_id": unique_embeddings[idx]["chunk_id"],
                "vector": unique_embeddings[idx]["vector"],
            }
            if "created_by" in unique_embeddings[idx]:
                dup["created_by"] = unique_embeddings[idx]["created_by"]
            result.append(dup)
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Property Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBatchEmbeddingIdempotency:
    """For any set of embeddings where some chunk_ids already have embeddings,
    calling batch_insert_embeddings SHALL not raise an error and SHALL silently
    skip duplicates while inserting new ones.

    **Validates: Requirement 8.3**
    """

    @given(embeddings=embeddings_list_st)
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
    )
    def test_no_error_with_duplicate_chunk_ids(
        self, embeddings: list[dict]
    ) -> None:
        """WHEN batch_insert_embeddings is called with embeddings containing
        duplicate chunk_ids, THE function SHALL not raise any error.

        **Validates: Requirement 8.3**
        """
        # Arrange: create duplicates by repeating some chunk_ids
        if len(embeddings) >= 2:
            # Duplicate the first embedding's chunk_id
            dup_embedding = {
                "chunk_id": embeddings[0]["chunk_id"],
                "vector": embeddings[1]["vector"],
            }
            if "created_by" in embeddings[0]:
                dup_embedding["created_by"] = embeddings[0]["created_by"]
            embeddings_with_dups = embeddings + [dup_embedding]
        else:
            # Even with a single embedding, duplicate it
            dup_embedding = {
                "chunk_id": embeddings[0]["chunk_id"],
                "vector": embeddings[0]["vector"],
            }
            if "created_by" in embeddings[0]:
                dup_embedding["created_by"] = embeddings[0]["created_by"]
            embeddings_with_dups = embeddings + [dup_embedding]

        session = _build_mock_session()

        # Act & Assert: no exception should be raised
        batch_insert_embeddings(session, embeddings_with_dups)

    @given(
        embeddings=embeddings_list_st,
        duplicate_indices=st.lists(
            st.integers(min_value=0, max_value=9), min_size=1, max_size=5
        ),
    )
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
    )
    def test_idempotent_with_arbitrary_duplicates(
        self, embeddings: list[dict], duplicate_indices: list[int]
    ) -> None:
        """WHEN batch_insert_embeddings is called with an arbitrary number of
        duplicate chunk_ids, THE function SHALL execute without error and call
        session.execute exactly once (single batch statement).

        **Validates: Requirement 8.3**
        """
        # Arrange: create embeddings with duplicates
        embeddings_with_dups = _create_embeddings_with_duplicates(
            embeddings, duplicate_indices
        )
        session = _build_mock_session()

        # Act: should not raise
        batch_insert_embeddings(session, embeddings_with_dups)

        # Assert: session.execute is called exactly once (single batch INSERT)
        assert session.execute.call_count == 1, (
            f"Expected 1 execute call, got {session.execute.call_count}"
        )

    @given(embeddings=embeddings_list_st)
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
    )
    def test_on_conflict_do_nothing_in_statement(
        self, embeddings: list[dict]
    ) -> None:
        """WHEN batch_insert_embeddings is called, THE generated SQL statement
        SHALL include ON CONFLICT DO NOTHING on chunk_id for idempotency.

        **Validates: Requirement 8.3**
        """
        # Arrange
        session = _build_mock_session()

        # Act
        batch_insert_embeddings(session, embeddings)

        # Assert: verify the statement passed to execute has on_conflict_do_nothing
        call_args = session.execute.call_args
        stmt = call_args[0][0]

        # The statement should be an Insert with on_conflict_do_nothing applied
        # SQLAlchemy's Insert with on_conflict_do_nothing sets _post_values_clause
        assert hasattr(stmt, "_post_values_clause"), (
            "Statement should have _post_values_clause (ON CONFLICT clause)"
        )
        assert stmt._post_values_clause is not None, (
            "ON CONFLICT DO NOTHING clause should be set on the statement"
        )
