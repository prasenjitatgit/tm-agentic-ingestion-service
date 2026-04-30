"""FastAPI application entry point.

The HTTP surface is implemented in `app.api.v1.ingestion`; this module
just constructs the `FastAPI` instance, applies middleware, mounts the
router, and wires lifecycle hooks.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import ingestion_router
from app.core.config import get_settings
from app.core.security import correlation_id_middleware
from app.utils.logger import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)
_settings = get_settings()

app = FastAPI(
    title=_settings.service_name,
    version="0.1.0",
    docs_url="/docs",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.middleware("http")(correlation_id_middleware)

app.include_router(ingestion_router)


@app.on_event("startup")
def _on_startup() -> None:
    log.info("service_starting", service=_settings.service_name)


@app.on_event("shutdown")
def _on_shutdown() -> None:
    log.info("service_stopping", service=_settings.service_name)
