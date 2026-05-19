"""Lifecycle / resilience nodes.

Hosts:

* `node(name)` decorator            — structured logging + exception capture into state.
* `classify(exc)`                    — transient vs. permanent error classifier.
* `update_status_chunked_node`       — mark `documents.status = CHUNKED`.
* `update_status_embedded_node`      — mark `documents.status = EMBEDDED`.
* `update_status_node`               — mark `documents.status = INGESTED` (final step).
* `retry_handler_node`               — central backoff + retry router.
* `failure_handler_node`             — sink for non-retryable / exhausted failures.
"""

from __future__ import annotations

import functools
import os
import random
import time
import traceback
import uuid
from typing import Any, Callable

import httpx
import pybreaker
from botocore.exceptions import BotoCoreError, ClientError
from openai import APIError, APITimeoutError
from sqlalchemy import update
from sqlalchemy.exc import DBAPIError, OperationalError

from app.core.config import get_settings
from app.graph.state import AgentState, ErrorInfo, RetryContext
from app.models.db_models import (
    STATUS_CHUNKED,
    STATUS_EMBEDDED,
    STATUS_INGESTED,
    Document,
)
from app.services.db_service import session_scope
from app.utils.logger import get_logger

log = get_logger(__name__)
_settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
#  RETRY CLASSIFIER + NODE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

_NON_RETRYABLE: tuple[type[BaseException], ...] = (
    pybreaker.CircuitBreakerError,
    ValueError,
    TypeError,
    KeyError,
    NotImplementedError,
    FileNotFoundError,
    MemoryError,
)
_RETRYABLE: tuple[type[BaseException], ...] = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
    APITimeoutError,
    APIError,
    BotoCoreError,
    ClientError,
    OperationalError,
    DBAPIError,
    ConnectionError,
    TimeoutError,
)


def classify(exc: BaseException) -> bool:
    """Return True iff `exc` is considered transient (i.e. retryable)."""
    if isinstance(exc, _NON_RETRYABLE):
        return False
    if isinstance(exc, _RETRYABLE):
        return True

    # Check for Docling ConversionError (retryable - may be transient)
    if type(exc).__name__ == "ConversionError" and "docling" in getattr(
        type(exc), "__module__", ""
    ):
        return True

    # RuntimeError from model download is non-retryable
    if isinstance(exc, RuntimeError) and "model" in str(exc).lower():
        return False

    return False


def node(
    name: str,
) -> Callable[[Callable[[AgentState], dict[str, Any]]], Callable[[AgentState], dict[str, Any]]]:
    """Decorator: structured logging + exception capture into state.

    Successful nodes return their state delta directly. Raised exceptions
    are translated into `{"error": ErrorInfo, "retry_context": ...}` so
    the graph can route to `retry_handler` without an exception escaping.
    """

    def decorator(fn: Callable[[AgentState], dict[str, Any]]):
        @functools.wraps(fn)
        def wrapper(state: AgentState) -> dict[str, Any]:
            doc_meta = state.get("doc_metadata")
            doc_id = getattr(doc_meta, "doc_id", None) if doc_meta else None
            rc = state.get("retry_context") or RetryContext()
            log.info(
                "node_start",
                node=name,
                doc_id=doc_id,
                retry_count=rc.retry_count,
            )
            t0 = time.perf_counter()
            try:
                delta = fn(state) or {}
                delta.setdefault("error", None)
                # Clear failed_node on successful execution.
                if "retry_context" not in delta:
                    delta["retry_context"] = rc.model_copy(update={"failed_node": None})
                log.info(
                    "node_end",
                    node=name,
                    doc_id=doc_id,
                    elapsed_ms=int((time.perf_counter() - t0) * 1000),
                )
                return delta
            except BaseException as exc:  # noqa: BLE001 — we capture everything
                retryable = classify(exc)
                tb = traceback.format_exc()
                log.error(
                    "node_error",
                    node=name,
                    doc_id=doc_id,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    retryable=retryable,
                    elapsed_ms=int((time.perf_counter() - t0) * 1000),
                )
                return {
                    "error": ErrorInfo(type=type(exc).__name__, message=str(exc), traceback=tb),
                    "retry_context": rc.model_copy(
                        update={"failed_node": name, "is_retryable": retryable}
                    ),
                }

        return wrapper

    return decorator


# ─────────────────────────────────────────────────────────────────────────────
#  STATUS SINKS
# ─────────────────────────────────────────────────────────────────────────────


@node("update_status_chunked")
def update_status_chunked_node(state: AgentState) -> dict[str, Any]:
    """Set `documents.status = CHUNKED` after chunks are persisted."""
    meta = state["doc_metadata"]

    with session_scope() as session:
        session.execute(
            update(Document)
            .where(Document.doc_id == meta.doc_id)
            .values(status=STATUS_CHUNKED, updated_by="ingestion-service")
        )

    log.info("status_updated", doc_id=meta.doc_id, status=STATUS_CHUNKED)
    return {}


@node("update_status_embedded")
def update_status_embedded_node(state: AgentState) -> dict[str, Any]:
    """Set `documents.status = EMBEDDED` after embeddings are persisted."""
    meta = state["doc_metadata"]

    with session_scope() as session:
        session.execute(
            update(Document)
            .where(Document.doc_id == meta.doc_id)
            .values(status=STATUS_EMBEDDED, updated_by="ingestion-service")
        )

    log.info("status_updated", doc_id=meta.doc_id, status=STATUS_EMBEDDED)
    return {}


@node("update_status")
def update_status_node(state: AgentState) -> dict[str, Any]:
    """Sync `documents.status = INGESTED` (final ingestion step) and emit summary."""
    meta = state["doc_metadata"]
    chunks = state.get("chunks") or []
    images = state.get("extracted_image_items") or []

    with session_scope() as session:
        session.execute(
            update(Document)
            .where(Document.doc_id == meta.doc_id)
            .values(status=STATUS_INGESTED, error=None, updated_by="ingestion-service")
        )

    local_path = state.get("local_path")
    if local_path and os.path.exists(local_path):
        try:
            os.remove(local_path)
        except OSError:
            pass

    summary = {
        "doc_id": meta.doc_id,
        "status": STATUS_INGESTED,
        "chunks": len(chunks),
        "images": len(images),
    }
    log.info("ingestion_completed", **summary)
    return {"result": summary}


def failure_handler_node(state: AgentState) -> dict[str, Any]:
    """Sink for non-retryable / exhausted failures.

    Since STATUS_FAILED is removed from the lifecycle, the document remains
    in its current status (allowing retry) and error details are stored in
    the `error` JSONB column.
    """
    meta = state.get("doc_metadata")
    err = state.get("error")
    rc = state.get("retry_context") or RetryContext()

    payload: dict[str, Any] = {}
    if meta is not None and err is not None:
        error_details = {
            "type": err.type,
            "message": err.message,
            "failed_node": rc.failed_node,
            "retry_count": rc.retry_count,
        }
        with session_scope() as session:
            session.execute(
                update(Document)
                .where(Document.doc_id == meta.doc_id)
                .values(
                    error=error_details,
                    updated_by="ingestion-service",
                )
            )
        payload = {
            "doc_id": meta.doc_id,
            "failed_node": rc.failed_node,
            "error_type": err.type,
            "error_message": err.message,
        }
        log.error("ingestion_failed", **payload)

    local_path = state.get("local_path")
    if local_path and os.path.exists(local_path):
        try:
            os.remove(local_path)
        except OSError:
            pass

    return {"result": payload}


def retry_handler_node(state: AgentState) -> dict[str, Any]:
    """Centralized retry router.

    If the captured error is retryable AND `retry_count < max_retries`,
    bump the count, sleep with exponential backoff + jitter, clear the
    error so the failed node can re-execute, and let the routing edge
    send execution back to `failed_node`. Otherwise fall through to
    `failure_handler`.
    """
    rc = state.get("retry_context") or RetryContext()
    err = state.get("error")

    log.warning(
        "retry_handler_invoked",
        failed_node=rc.failed_node,
        retry_count=rc.retry_count,
        max_retries=rc.max_retries,
        is_retryable=rc.is_retryable,
        error_type=getattr(err, "type", None),
        error_message=getattr(err, "message", None),
    )

    if rc.is_retryable and rc.retry_count < rc.max_retries:
        next_attempt = rc.retry_count + 1
        delay = _backoff_seconds(next_attempt)
        log.info(
            "retry_scheduled",
            failed_node=rc.failed_node,
            attempt=next_attempt,
            delay_seconds=round(delay, 2),
        )
        time.sleep(delay)
        return {
            "error": None,
            "retry_context": rc.model_copy(
                update={"retry_count": next_attempt, "is_retryable": False}
            ),
        }

    return {}


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff with jitter, capped at `retry_backoff_cap_seconds`."""
    base = _settings.retry_backoff_base_seconds
    cap = _settings.retry_backoff_cap_seconds
    raw = min(cap, base * (2 ** max(0, attempt - 1)))
    return raw * (0.5 + random.random() / 2)


# Re-exported so older imports (`from app.graph import functools`) keep working
# without surprising side effects. New code should import `functools` directly.
_ = functools
