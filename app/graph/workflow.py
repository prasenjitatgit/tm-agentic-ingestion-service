"""Graph construction, edge routing, and the per-job driver.

Topology
--------
Ingestion path:
    fetch_metadata → route_by_status
       ↓
       ├─ TO BE INGESTED → select_parser → docling_convert
       │                                        ↓
       │                          (has_images?) ──► summarize_images → assemble
       │                                    ↘ (none) ─────────────────╮
       │                                                              ▼
       │                                                    generate_chunks
       │                                                              ↓
       ├─ CHUNKED ──────────────────────────────────────► embed_chunks
       │                                                              ↓
       ├─ EMBEDDED ─────────────────────────────────► update_status_ingested → END
       │
       └─ no doc → END

Deletion path:
    fetch_deletion → delete_document → END

All document formats are converted via IBM Docling. Any node exception is
captured by the `@node(...)` wrapper into `state.error /
state.retry_context.failed_node` and routed to `retry_handler`, which
decides whether to re-enter `failed_node` (with exponential backoff +
jitter) or hand off to `failure_handler`.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable

import structlog
from langgraph.graph import END, StateGraph

from app.graph.nodes import (
    assemble_node,
    delete_document_node,
    docling_convert_node,
    embed_chunks_node,
    failure_handler_node,
    fetch_deletion_node,
    fetch_metadata_node,
    generate_chunks_node,
    retry_handler_node,
    select_parser_node,
    summarize_images_node,
    update_status_node,
)
from app.graph.state import AgentState, RetryContext, initial_state
from app.models.db_models import (
    STATUS_CHUNKED,
    STATUS_EMBEDDED,
    STATUS_INGESTED,
    STATUS_TO_BE_INGESTED,
)
from app.utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTING
# ─────────────────────────────────────────────────────────────────────────────


_STATUS_RESUME_MAP: dict[str, str] = {
    STATUS_TO_BE_INGESTED: "select_parser",
    STATUS_CHUNKED: "embed_chunks",
    STATUS_EMBEDDED: "update_status_ingested",
}


def route_after_fetch(state: AgentState) -> str:
    """Route based on document status for resumability."""
    if state.get("error") is not None:
        return "retry_handler"
    meta = state.get("doc_metadata")
    if meta is None:
        return END
    return _STATUS_RESUME_MAP.get(meta.status, "failure_handler")


def route_after_select_parser(state: AgentState) -> str:
    """Dispatch to Docling converter after file download and hash verification."""
    if state.get("error") is not None:
        return "retry_handler"
    return "docling_convert"


def route_after_parser(state: AgentState) -> str:
    """Skip the image branch when the parser produced no images."""
    if state.get("error") is not None:
        return "retry_handler"
    if state.get("extracted_image_items"):
        return "summarize_images"
    return "generate_chunks"


def route_after_node(state: AgentState, success_target: str) -> str:
    """Generic post-node router used by simple linear edges."""
    if state.get("error") is not None:
        return "retry_handler"
    return success_target


def route_after_retry(state: AgentState) -> str:
    """Send execution back to `failed_node` when retrying, else to failure_handler."""
    rc = state.get("retry_context") or RetryContext()
    err = state.get("error")
    # If retry_handler cleared the error, error is now None and we re-enter failed_node.
    if err is None and rc.failed_node and rc.retry_count <= rc.max_retries:
        return rc.failed_node
    return "failure_handler"


# ─────────────────────────────────────────────────────────────────────────────
#  GRAPH BUILDER
# ─────────────────────────────────────────────────────────────────────────────


def build_graph():
    """Compile and return the LangGraph `StateGraph` for one ingestion run."""
    g = StateGraph(AgentState)

    g.add_node("fetch_metadata", fetch_metadata_node)
    g.add_node("select_parser", select_parser_node)
    g.add_node("docling_convert", docling_convert_node)
    g.add_node("summarize_images", summarize_images_node)
    g.add_node("assemble", assemble_node)
    g.add_node("generate_chunks", generate_chunks_node)
    g.add_node("embed_chunks", embed_chunks_node)
    g.add_node("update_status_ingested", update_status_node)
    g.add_node("delete_document", delete_document_node)
    g.add_node("retry_handler", retry_handler_node)
    g.add_node("failure_handler", failure_handler_node)

    g.set_entry_point("fetch_metadata")

    g.add_conditional_edges(
        "fetch_metadata",
        route_after_fetch,
        {
            "select_parser": "select_parser",
            "embed_chunks": "embed_chunks",
            "update_status_ingested": "update_status_ingested",
            "failure_handler": "failure_handler",
            "retry_handler": "retry_handler",
            END: END,
        },
    )
    g.add_conditional_edges(
        "select_parser",
        route_after_select_parser,
        {
            "docling_convert": "docling_convert",
            "retry_handler": "retry_handler",
        },
    )

    g.add_conditional_edges(
        "docling_convert",
        route_after_parser,
        {
            "summarize_images": "summarize_images",
            "generate_chunks": "generate_chunks",
            "retry_handler": "retry_handler",
        },
    )

    g.add_conditional_edges(
        "summarize_images",
        functools.partial(route_after_node, success_target="assemble"),
        {"assemble": "assemble", "retry_handler": "retry_handler"},
    )
    g.add_conditional_edges(
        "assemble",
        functools.partial(route_after_node, success_target="generate_chunks"),
        {"generate_chunks": "generate_chunks", "retry_handler": "retry_handler"},
    )
    g.add_conditional_edges(
        "generate_chunks",
        functools.partial(route_after_node, success_target="embed_chunks"),
        {"embed_chunks": "embed_chunks", "retry_handler": "retry_handler"},
    )
    g.add_conditional_edges(
        "embed_chunks",
        functools.partial(route_after_node, success_target="update_status_ingested"),
        {"update_status_ingested": "update_status_ingested", "retry_handler": "retry_handler"},
    )
    g.add_edge("update_status_ingested", END)

    # Deletion path
    g.add_conditional_edges(
        "delete_document",
        functools.partial(route_after_node, success_target=END),
        {END: END, "retry_handler": "retry_handler"},
    )

    g.add_conditional_edges(
        "retry_handler",
        route_after_retry,
        {
            "fetch_metadata": "fetch_metadata",
            "select_parser": "select_parser",
            "docling_convert": "docling_convert",
            "summarize_images": "summarize_images",
            "assemble": "assemble",
            "generate_chunks": "generate_chunks",
            "embed_chunks": "embed_chunks",
            "update_status": "update_status_ingested",
            "update_status_ingested": "update_status_ingested",
            "delete_document": "delete_document",
            "failure_handler": "failure_handler",
        },
    )

    g.add_edge("failure_handler", END)
    return g.compile()


_GRAPH = build_graph()


# ─────────────────────────────────────────────────────────────────────────────
#  DELETION GRAPH
# ─────────────────────────────────────────────────────────────────────────────


def route_after_fetch_deletion(state: AgentState) -> str:
    """No TO BE DELETED docs → end gracefully; error → retry; else → delete_document."""
    if state.get("error") is not None:
        return "retry_handler"
    if state.get("doc_metadata") is None:
        return END
    return "delete_document"


def build_deletion_graph():
    """Compile and return the LangGraph `StateGraph` for one deletion run.

    Topology: fetch_deletion → delete_document → END
    """
    g = StateGraph(AgentState)

    g.add_node("fetch_deletion", fetch_deletion_node)
    g.add_node("delete_document", delete_document_node)
    g.add_node("retry_handler", retry_handler_node)
    g.add_node("failure_handler", failure_handler_node)

    g.set_entry_point("fetch_deletion")

    g.add_conditional_edges(
        "fetch_deletion",
        route_after_fetch_deletion,
        {"delete_document": "delete_document", "retry_handler": "retry_handler", END: END},
    )
    g.add_conditional_edges(
        "delete_document",
        functools.partial(route_after_node, success_target=END),
        {END: END, "retry_handler": "retry_handler"},
    )
    g.add_conditional_edges(
        "retry_handler",
        route_after_retry,
        {
            "fetch_deletion": "fetch_deletion",
            "delete_document": "delete_document",
            "failure_handler": "failure_handler",
        },
    )
    g.add_edge("failure_handler", END)
    return g.compile()


_DELETION_GRAPH = build_deletion_graph()


# ─────────────────────────────────────────────────────────────────────────────
#  JOB DRIVER (called from FastAPI BackgroundTasks)
# ─────────────────────────────────────────────────────────────────────────────


def run_ingestion_job(
    request_filter: dict[str, Any],
    max_documents: int,
    job_id: str,
    on_complete: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Sequentially process up to `max_documents` documents through the graph.

    Intended to be called from a FastAPI `BackgroundTasks` callable so the
    HTTP response returns immediately with a `job_id`.

    `on_complete(job_id, summary)` is invoked at the end so the API layer
    can update its in-memory job registry without coupling this module to it.
    """
    structlog.contextvars.bind_contextvars(job_id=job_id)
    log.info("ingestion_job_started", job_id=job_id, max_documents=max_documents)

    processed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    skipped = 0
    t0 = time.perf_counter()

    try:
        # Sequential processing per spec (constraint #5).
        for _ in range(max_documents):
            state = initial_state(request_filter=request_filter, correlation_id=job_id)
            final_state = _GRAPH.invoke(state)

            if final_state.get("doc_metadata") is None:
                skipped += 1
                break

            res = final_state.get("result") or {}
            if res.get("status") == STATUS_INGESTED:
                processed.append(res)
            else:
                failed.append(res)

        summary = {
            "job_id": job_id,
            "status": "COMPLETED",
            "processed": processed,
            "failed": failed,
            "skipped": skipped,
            "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        }
    except BaseException as exc:  # noqa: BLE001
        log.exception("ingestion_job_crashed", job_id=job_id)
        summary = {
            "job_id": job_id,
            "status": "CRASHED",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "processed": processed,
            "failed": failed,
            "skipped": skipped,
            "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        }
    finally:
        structlog.contextvars.unbind_contextvars("job_id")

    log.info("ingestion_job_finished", **summary)
    if on_complete is not None:
        on_complete(job_id, summary)
    return summary


def run_deletion_job(
    max_documents: int,
    job_id: str,
    on_complete: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Sequentially process up to `max_documents` TO BE DELETED documents.

    Intended to be called from a FastAPI `BackgroundTasks` callable so the
    HTTP response returns immediately with a `job_id`.

    `on_complete(job_id, summary)` is invoked at the end so the API layer
    can update its in-memory job registry without coupling this module to it.
    """
    from app.models.db_models import STATUS_DELETED

    structlog.contextvars.bind_contextvars(job_id=job_id)
    log.info("deletion_job_started", job_id=job_id, max_documents=max_documents)

    deleted: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    skipped = 0
    t0 = time.perf_counter()

    try:
        for _ in range(max_documents):
            state = initial_state(request_filter={}, correlation_id=job_id)
            final_state = _DELETION_GRAPH.invoke(state)

            if final_state.get("doc_metadata") is None:
                skipped += 1
                break

            res = final_state.get("result") or {}
            if res.get("status") == STATUS_DELETED:
                deleted.append(res)
            else:
                failed.append(res)

        summary = {
            "job_id": job_id,
            "status": "COMPLETED",
            "deleted": deleted,
            "failed": failed,
            "skipped": skipped,
            "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        }
    except BaseException as exc:  # noqa: BLE001
        log.exception("deletion_job_crashed", job_id=job_id)
        summary = {
            "job_id": job_id,
            "status": "CRASHED",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "deleted": deleted,
            "failed": failed,
            "skipped": skipped,
            "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        }
    finally:
        structlog.contextvars.unbind_contextvars("job_id")

    log.info("deletion_job_finished", **summary)
    if on_complete is not None:
        on_complete(job_id, summary)
    return summary
