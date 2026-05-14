"""v1 API surface."""

from app.api.v1.ingestion import router as ingestion_router

__all__ = ["ingestion_router"]
