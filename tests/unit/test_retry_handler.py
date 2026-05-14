"""Tests for the centralized retry handler + classifier + node wrapper."""

from __future__ import annotations

import httpx
import pybreaker
from sqlalchemy.exc import OperationalError

from app.graph.nodes.status import (
    classify,
    node,
    retry_handler_node,
)
from app.graph.state import AgentState, ErrorInfo, RetryContext
from app.graph.workflow import route_after_retry


# ── classify ────────────────────────────────────────────────────────────────


def test_classify_retryable_network():
    assert classify(httpx.TimeoutException("boom")) is True
    assert classify(httpx.ConnectError("boom")) is True


def test_classify_retryable_db():
    err = OperationalError("stmt", {}, Exception("conn lost"))
    assert classify(err) is True


def test_classify_non_retryable():
    assert classify(ValueError("bad input")) is False
    assert classify(pybreaker.CircuitBreakerError("open")) is False


# ── @node wrapper ───────────────────────────────────────────────────────────


def test_node_decorator_captures_exception_into_state():
    @node("test_node")
    def fn(state):
        raise httpx.TimeoutException("upstream timeout")

    delta = fn(AgentState(retry_context=RetryContext(retry_count=0, max_retries=3)))
    assert delta["error"] is not None
    assert delta["retry_context"].failed_node == "test_node"
    assert delta["retry_context"].is_retryable is True
    assert delta["error"].type == "TimeoutException"


def test_node_decorator_marks_value_error_non_retryable():
    @node("test_node")
    def fn(state):
        raise ValueError("schema mismatch")

    delta = fn(AgentState(retry_context=RetryContext(retry_count=0, max_retries=3)))
    assert delta["retry_context"].is_retryable is False


def test_node_decorator_clears_error_on_success():
    @node("test_node")
    def fn(state):
        return {"foo": "bar"}

    delta = fn(AgentState(retry_context=RetryContext(retry_count=0, max_retries=3)))
    assert delta["foo"] == "bar"
    assert delta["error"] is None
    assert delta["retry_context"].failed_node is None


# ── retry_handler_node ──────────────────────────────────────────────────────


def test_retry_handler_schedules_retry_when_under_budget(monkeypatch):
    monkeypatch.setattr("app.graph.nodes.status.time.sleep", lambda *_: None)
    state: AgentState = {
        "retry_context": RetryContext(
            retry_count=1, max_retries=3, failed_node="parse_pdf", is_retryable=True
        ),
        "error": _make_error(),
    }
    delta = retry_handler_node(state)
    assert delta["error"] is None
    rc = delta["retry_context"]
    assert rc.retry_count == 2
    assert rc.is_retryable is False  # cleared so the next attempt re-classifies
    assert rc.failed_node == "parse_pdf"


def test_retry_handler_gives_up_when_budget_exhausted(monkeypatch):
    monkeypatch.setattr("app.graph.nodes.status.time.sleep", lambda *_: None)
    state: AgentState = {
        "retry_context": RetryContext(
            retry_count=3, max_retries=3, failed_node="embed_chunks", is_retryable=True
        ),
        "error": _make_error(),
    }
    delta = retry_handler_node(state)
    # No state mutation → router will send to failure_handler.
    assert delta == {}


def test_route_after_retry_routes_to_failed_node(monkeypatch):
    monkeypatch.setattr("app.graph.nodes.status.time.sleep", lambda *_: None)
    state: AgentState = {
        "retry_context": RetryContext(
            retry_count=0, max_retries=3, failed_node="parse_pdf", is_retryable=True
        ),
        "error": _make_error(),
    }
    state.update(retry_handler_node(state))
    assert route_after_retry(state) == "parse_pdf"


def test_route_after_retry_routes_to_failure_handler():
    state: AgentState = {
        "retry_context": RetryContext(
            retry_count=3, max_retries=3, failed_node="embed_chunks", is_retryable=True
        ),
        "error": _make_error(),
    }
    assert route_after_retry(state) == "failure_handler"


def test_route_after_retry_handles_non_retryable():
    state: AgentState = {
        "retry_context": RetryContext(
            retry_count=0, max_retries=3, failed_node="select_parser", is_retryable=False
        ),
        "error": _make_error(),
    }
    assert route_after_retry(state) == "failure_handler"


# ── helpers ─────────────────────────────────────────────────────────────────


def _make_error():
    return ErrorInfo(type="TimeoutException", message="boom", traceback=None)
