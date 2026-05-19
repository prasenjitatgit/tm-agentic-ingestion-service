"""Ingestion API (v1).

Endpoints
---------
* POST /v1/ingestion/documents      → submit a document for ingestion
* POST /v1/ingestion/run            → start a background job (returns job_id)
* GET  /healthz                     → liveness
* GET  /readyz                      → readiness (DB + S3 + NIM key)
"""

from __future__ import annotations

import uuid
from typing import Any, Literal, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.graph import run_ingestion_job
from app.models.db_models import Document, STATUS_TO_BE_INGESTED
from app.services.db_service import ping as db_ping, session_scope
from app.services.s3_service import S3Client
from app.utils.logger import get_logger

log = get_logger(__name__)
_settings = get_settings()
_s3 = S3Client()

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
#  API SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

DocType = Literal["PDF", "EXCEL", "PPT", "DOCX"]
KnowledgeBaseType = Literal["Maintenance", "Construction", "BusinessIntelligence"]


class DocumentSubmitRequest(BaseModel):
    """Payload for submitting a document for ingestion."""

    doc_type: DocType
    doc_hash: str = Field(..., min_length=1, max_length=100)
    doc_name: str = Field(..., min_length=1, max_length=200)
    source: str = Field(..., min_length=1, max_length=500)
    s3_url: str = Field(..., min_length=1, max_length=500)
    knowledge_base_type: KnowledgeBaseType
    author: Optional[str] = Field(default=None, max_length=100)
    version: Optional[str] = Field(default=None, max_length=20)
    created_by: str = Field(..., min_length=1, max_length=100)


class DocumentSubmitResponse(BaseModel):
    """Response after a document is accepted for ingestion."""

    doc_id: str
    status: str
    message: str


class IngestionRunRequest(BaseModel):
    """Filter criteria for selecting TO BE INGESTED documents."""

    knowledge_base_type: Optional[KnowledgeBaseType] = None
    max_documents: int = Field(default=1, ge=1, le=100)


class IngestionRunResponse(BaseModel):
    """Returned immediately after the background task is enqueued."""

    job_id: str
    status: Literal["STARTED"] = "STARTED"


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    checks: dict[str, bool] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────


@router.post(
    "/v1/ingestion/documents",
    response_model=DocumentSubmitResponse,
    status_code=status.HTTP_201_CREATED,
)
def submit_document(payload: DocumentSubmitRequest) -> DocumentSubmitResponse:
    """Submit a document for ingestion.

    Inserts the document with status TO BE INGESTED for the pipeline to pick up.
    """
    with session_scope() as session:
        new_doc = Document(
            doc_type=payload.doc_type,
            doc_hash=payload.doc_hash,
            doc_name=payload.doc_name,
            source=payload.source,
            s3_url=payload.s3_url,
            knowledge_base_type=payload.knowledge_base_type,
            author=payload.author,
            version=payload.version,
            status=STATUS_TO_BE_INGESTED,
            created_by=payload.created_by,
            updated_by=payload.created_by,
        )
        session.add(new_doc)
        session.flush()  # Populate server-generated doc_id

        doc_id = str(new_doc.doc_id)
        log.info(
            "document_submitted",
            doc_id=doc_id,
            doc_name=payload.doc_name,
        )

    return DocumentSubmitResponse(
        doc_id=doc_id,
        status=STATUS_TO_BE_INGESTED,
        message="Document accepted for ingestion",
    )


@router.post(
    "/v1/ingestion/run",
    response_model=IngestionRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def trigger_ingestion(
    payload: IngestionRunRequest, background_tasks: BackgroundTasks
) -> IngestionRunResponse:
    """Kick off the LangGraph ingestion in a background task and return a job id."""
    job_id = str(uuid.uuid4())
    request_filter: dict[str, Any] = {}
    if payload.knowledge_base_type:
        request_filter["knowledge_base_type"] = payload.knowledge_base_type

    background_tasks.add_task(
        run_ingestion_job,
        request_filter=request_filter,
        max_documents=payload.max_documents,
        job_id=job_id,
    )
    log.info("ingestion_job_accepted", job_id=job_id, request_filter=request_filter)
    return IngestionRunResponse(job_id=job_id)


@router.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    """Liveness probe."""
    return HealthResponse(status="ok", checks={"alive": True})


@router.get("/readyz", response_model=HealthResponse)
def readyz() -> HealthResponse:
    """Readiness probe: DB + S3 reachable, NIM key configured."""
    checks: dict[str, bool] = {}
    overall_ok = True

    try:
        checks["db"] = db_ping()
    except Exception as exc:  # noqa: BLE001
        log.warning("readyz_db_failed", error=str(exc))
        checks["db"] = False
        overall_ok = False

    try:
        checks["s3"] = _s3.ping()
    except Exception as exc:  # noqa: BLE001
        log.warning("readyz_s3_failed", error=str(exc))
        checks["s3"] = False
        overall_ok = False

    checks["nim_configured"] = bool(_settings.nvidia_api_key)
    overall_ok = overall_ok and checks["nim_configured"]

    if not overall_ok:
        raise HTTPException(status_code=503, detail={"status": "degraded", "checks": checks})
    return HealthResponse(status="ok", checks=checks)
