"""Authentication / middleware helpers.

Currently exposes:

* `correlation_id_middleware` — assigns or echoes an `x-correlation-id`
  header for every request and binds it into the structlog contextvars
  for the duration of the request.

Auth (e.g. JWT, API-key, IAM-SigV4) is intentionally a no-op stub here;
plug in your provider of choice and add it via `app.add_middleware(...)`.
"""

from __future__ import annotations

import time
import uuid

import structlog
from fastapi import Request

from app.utils.logger import get_logger

log = get_logger(__name__)


async def correlation_id_middleware(request: Request, call_next):
    """Bind/propagate `x-correlation-id` and log basic request metrics."""
    cid = request.headers.get("x-correlation-id") or str(uuid.uuid4())
    structlog.contextvars.bind_contextvars(correlation_id=cid)
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        log.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            elapsed_ms=int((time.perf_counter() - t0) * 1000),
        )
        structlog.contextvars.unbind_contextvars("correlation_id")
    response.headers["x-correlation-id"] = cid
    return response
