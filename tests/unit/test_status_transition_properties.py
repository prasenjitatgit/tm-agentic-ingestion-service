"""Property-based tests for Status Transition Validity.

# Feature: document-schema-redesign, Property 4: Status transition validity

For any document, status transitions SHALL only follow valid paths:
TO BE INGESTED → CHUNKED → EMBEDDED → INGESTED, or
INGESTED → TO BE DELETED → DELETED.
No other transitions are permitted.

**Validates: Requirement 2.8**
"""

from __future__ import annotations

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from app.models.db_models import (
    STATUS_TO_BE_INGESTED,
    STATUS_CHUNKED,
    STATUS_EMBEDDED,
    STATUS_INGESTED,
    STATUS_TO_BE_DELETED,
    STATUS_DELETED,
    ALLOWED_STATUSES,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Valid Transition Map
# ─────────────────────────────────────────────────────────────────────────────

# The only valid transitions per the design document:
#   Ingestion path: TO BE INGESTED → CHUNKED → EMBEDDED → INGESTED
#   Deletion path:  INGESTED → TO BE DELETED → DELETED
VALID_TRANSITIONS: dict[str, str] = {
    STATUS_TO_BE_INGESTED: STATUS_CHUNKED,
    STATUS_CHUNKED: STATUS_EMBEDDED,
    STATUS_EMBEDDED: STATUS_INGESTED,
    STATUS_INGESTED: STATUS_TO_BE_DELETED,
    STATUS_TO_BE_DELETED: STATUS_DELETED,
}


def is_valid_transition(current_status: str, target_status: str) -> bool:
    """Check whether a status transition from current_status to target_status is valid.

    Only the following transitions are permitted:
        TO BE INGESTED → CHUNKED
        CHUNKED → EMBEDDED
        EMBEDDED → INGESTED
        INGESTED → TO BE DELETED
        TO BE DELETED → DELETED

    All other transitions (including self-transitions) are invalid.
    """
    return VALID_TRANSITIONS.get(current_status) == target_status


# ─────────────────────────────────────────────────────────────────────────────
#  Strategies
# ─────────────────────────────────────────────────────────────────────────────

all_statuses_st = st.sampled_from(list(ALLOWED_STATUSES))

# Strategy for generating a valid (current, target) pair
valid_transition_st = st.sampled_from(list(VALID_TRANSITIONS.items()))

# Strategy for generating any (current, target) pair
any_transition_st = st.tuples(all_statuses_st, all_statuses_st)


# ─────────────────────────────────────────────────────────────────────────────
#  Property Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestStatusTransitionValidity:
    """For any document, status transitions SHALL only follow valid paths:
    TO BE INGESTED → CHUNKED → EMBEDDED → INGESTED, or
    INGESTED → TO BE DELETED → DELETED. No other transitions are permitted.

    **Validates: Requirement 2.8**
    """

    @given(transition=valid_transition_st)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_valid_transitions_are_accepted(
        self, transition: tuple[str, str]
    ) -> None:
        """All transitions along the valid paths SHALL be accepted.

        **Validates: Requirement 2.8**
        """
        current_status, target_status = transition
        assert is_valid_transition(current_status, target_status), (
            f"Expected transition {current_status!r} → {target_status!r} to be valid"
        )

    @given(current_status=all_statuses_st, target_status=all_statuses_st)
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_invalid_transitions_are_rejected(
        self, current_status: str, target_status: str
    ) -> None:
        """Any transition NOT on the valid paths SHALL be rejected.

        **Validates: Requirement 2.8**
        """
        expected_valid = VALID_TRANSITIONS.get(current_status) == target_status
        actual = is_valid_transition(current_status, target_status)
        assert actual == expected_valid, (
            f"Transition {current_status!r} → {target_status!r}: "
            f"expected valid={expected_valid}, got valid={actual}"
        )

    @given(current_status=all_statuses_st)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_self_transitions_are_invalid(
        self, current_status: str
    ) -> None:
        """No status may transition to itself — self-transitions are never valid.

        **Validates: Requirement 2.8**
        """
        assert not is_valid_transition(current_status, current_status), (
            f"Self-transition {current_status!r} → {current_status!r} should be invalid"
        )

    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_deleted_is_terminal(self, data: st.DataObject) -> None:
        """DELETED is a terminal state — no transitions out of DELETED are valid.

        **Validates: Requirement 2.8**
        """
        target = data.draw(all_statuses_st)
        assert not is_valid_transition(STATUS_DELETED, target), (
            f"Transition from DELETED → {target!r} should be invalid (DELETED is terminal)"
        )

    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_each_status_has_at_most_one_successor(
        self, data: st.DataObject
    ) -> None:
        """Each status has at most one valid successor — the transition graph is linear.

        **Validates: Requirement 2.8**
        """
        current = data.draw(all_statuses_st)
        valid_targets = [s for s in ALLOWED_STATUSES if is_valid_transition(current, s)]
        assert len(valid_targets) <= 1, (
            f"Status {current!r} has multiple valid successors: {valid_targets}"
        )

    @given(
        transitions=st.lists(
            st.sampled_from(list(ALLOWED_STATUSES)),
            min_size=2,
            max_size=8,
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_random_transition_sequences(
        self, transitions: list[str]
    ) -> None:
        """For random sequences of statuses, verify that only consecutive valid
        transitions succeed and all others fail.

        **Validates: Requirement 2.8**
        """
        for i in range(len(transitions) - 1):
            current = transitions[i]
            target = transitions[i + 1]
            result = is_valid_transition(current, target)
            expected = VALID_TRANSITIONS.get(current) == target
            assert result == expected, (
                f"In sequence {transitions}, transition [{i}] "
                f"{current!r} → {target!r}: expected valid={expected}, got valid={result}"
            )
