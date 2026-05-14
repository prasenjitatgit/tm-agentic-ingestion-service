# syntax=docker/dockerfile:1.7

# ── Builder ──────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY app ./app

RUN pip install --upgrade pip wheel \
    && pip install --prefix=/install .


# ── Runtime ──────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    DOCLING_MODELS_PATH=/app/.cache/docling

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app \
    && useradd --system --gid app --home /app --shell /usr/sbin/nologin app

WORKDIR /app

COPY --from=builder /install /usr/local
COPY app ./app
COPY docker ./docker

# Pre-download Docling model weights during build so they are cached in the
# image and available at runtime without network access.
RUN python -c "\
from docling.document_converter import DocumentConverter; \
from docling.datamodel.pipeline_options import PdfPipelineOptions; \
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline; \
from docling.datamodel.base_models import InputFormat; \
opts = PdfPipelineOptions(); \
opts.do_table_structure = True; \
DocumentConverter( \
    allowed_formats=[InputFormat.PDF], \
    format_options={InputFormat.PDF: {'pipeline_cls': StandardPdfPipeline, 'pipeline_options': opts}} \
)" \
    && chown -R app:app /app/.cache

USER app

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --silent --fail http://localhost:8080/healthz || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers"]
