"""Property-based tests for error classification determinism.

Feature: docling-migration, Property 1: Error Classification Determinism

For any exception raised during Docling conversion, the error classifier SHALL
deterministically categorize it as either retryable or non-retryable, and the
classification SHALL be consistent across repeated invocations with the same
exception type.

**Validates: Requirements 1.5, 6.3**
"""

from __future__ import annotations

import httpx
import pybreaker
from botocore.exceptions import BotoCoreError, ClientError
from hypothesis import given, settings, strategies as st
from openai import APIError, APITimeoutError
from sqlalchemy.exc import DBAPIError, OperationalError

from app.graph.nodes.status import classify


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers: Simulate Docling ConversionError via duck-typing
# ─────────────────────────────────────────────────────────────────────────────


class _FakeDoclingModule:
    """Namespace to simulate docling.exceptions module path."""

    pass


def _make_conversion_error(msg: str = "conversion failed") -> BaseException:
    """Create a fake ConversionError that looks like docling's."""

    class ConversionError(Exception):
        __module__ = "docling.exceptions"

    return ConversionError(msg)


# ─────────────────────────────────────────────────────────────────────────────
#  Strategies: Generate exceptions from known categories
# ─────────────────────────────────────────────────────────────────────────────

_non_retryable_factories = [
    lambda msg: pybreaker.CircuitBreakerError("open"),
    lambda msg: ValueError(msg),
    lambda msg: TypeError(msg),
    lambda msg: KeyError(msg),
    lambda msg: NotImplementedError(msg),
    lambda msg: FileNotFoundError(msg),
    lambda msg: MemoryError(msg),
    lambda msg: RuntimeError(f"model download failed: {msg}"),
]

_retryable_factories = [
    lambda msg: httpx.TimeoutException(msg),
    lambda msg: httpx.ConnectError(msg),
    lambda msg: httpx.ReadError(msg),
    lambda msg: httpx.RemoteProtocolError(msg),
    lambda msg: APITimeoutError(request=None),
    lambda msg: ConnectionError(msg),
    lambda msg: TimeoutError(msg),
    lambda msg: _make_conversion_error(msg),
]


@st.composite
def non_retryable_exceptions(draw):
    """Generate exceptions that should be classified as non-retryable."""
    msg = draw(st.text(min_size=1, max_size=50))
    factory = draw(st.sampled_from(_non_retryable_factories))
    return factory(msg)


@st.composite
def retryable_exceptions(draw):
    """Generate exceptions that should be classified as retryable."""
    msg = draw(st.text(min_size=1, max_size=50))
    factory = draw(st.sampled_from(_retryable_factories))
    return factory(msg)


@st.composite
def all_classified_exceptions(draw):
    """Generate any exception from both retryable and non-retryable categories."""
    factory = draw(st.sampled_from(_non_retryable_factories + _retryable_factories))
    msg = draw(st.text(min_size=1, max_size=50))
    return factory(msg)


# ─────────────────────────────────────────────────────────────────────────────
#  Property Tests
# ─────────────────────────────────────────────────────────────────────────────


@settings(max_examples=100)
@given(exc=all_classified_exceptions())
def test_classification_is_deterministic(exc: BaseException):
    """classify() returns the same boolean for the same exception instance
    across repeated invocations.

    Property 1: Error Classification Determinism
    **Validates: Requirements 1.5, 6.3**
    """
    result1 = classify(exc)
    result2 = classify(exc)
    result3 = classify(exc)
    assert result1 == result2 == result3, (
        f"Non-deterministic classification for {type(exc).__name__}: "
        f"got {result1}, {result2}, {result3}"
    )


@settings(max_examples=100)
@given(exc=non_retryable_exceptions())
def test_non_retryable_exceptions_classified_correctly(exc: BaseException):
    """All known non-retryable exceptions are classified as non-retryable (False).

    Property 1: Error Classification Determinism
    **Validates: Requirements 1.5, 6.3**
    """
    assert classify(exc) is False, (
        f"Expected non-retryable (False) for {type(exc).__name__}({exc}), got True"
    )


@settings(max_examples=100)
@given(exc=retryable_exceptions())
def test_retryable_exceptions_classified_correctly(exc: BaseException):
    """All known retryable exceptions are classified as retryable (True).

    Property 1: Error Classification Determinism
    **Validates: Requirements 1.5, 6.3**
    """
    assert classify(exc) is True, (
        f"Expected retryable (True) for {type(exc).__name__}({exc}), got False"
    )


@settings(max_examples=100)
@given(msg=st.text(min_size=1, max_size=100))
def test_docling_conversion_error_always_retryable(msg: str):
    """Docling ConversionError is always classified as retryable regardless of message.

    Property 1: Error Classification Determinism
    **Validates: Requirements 1.5, 6.3**
    """
    exc = _make_conversion_error(msg)
    assert classify(exc) is True, (
        f"Docling ConversionError should be retryable, got False for message: {msg}"
    )


@settings(max_examples=100)
@given(msg=st.text(min_size=1, max_size=100).filter(lambda s: "model" not in s.lower()))
def test_runtime_error_without_model_is_not_classified(msg: str):
    """RuntimeError without 'model' in message falls through to default (False).

    Property 1: Error Classification Determinism
    **Validates: Requirements 1.5, 6.3**
    """
    exc = RuntimeError(msg)
    # RuntimeError without "model" keyword is not in _NON_RETRYABLE or _RETRYABLE,
    # so it falls through to the default return False
    assert classify(exc) is False


@settings(max_examples=100)
@given(msg=st.from_regex(r".*[Mm]odel.*", fullmatch=True))
def test_runtime_error_with_model_keyword_non_retryable(msg: str):
    """RuntimeError with 'model' in message is non-retryable (model download failure).

    Property 1: Error Classification Determinism
    **Validates: Requirements 1.5, 6.3**
    """
    exc = RuntimeError(msg)
    assert classify(exc) is False, (
        f"RuntimeError with 'model' keyword should be non-retryable, got True for: {msg}"
    )
