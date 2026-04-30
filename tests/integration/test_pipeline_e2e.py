"""End-to-end integration test of the LangGraph ingestion pipeline.

Uses a **real Postgres + pgvector** (via `docker compose up -d postgres`)
but stubs the external dependencies that don't belong in CI:

    * S3            → in-memory blob store + local-file download
    * VLM           → returns a deterministic summary
    * Embeddings    → returns deterministic 1536-dim float vectors

The test inserts a real `documents` row, builds a real PDF with PyMuPDF,
runs the compiled LangGraph, and asserts that the document ends in
`EMBEDDED` status with chunks + matching embeddings persisted.
"""

from __future__ import annotations

import hashlib
import shutil
import uuid
from pathlib import Path

import fitz  # PyMuPDF
import pytest
from sqlalchemy import text

# The conftest sets PG_DSN env, rebuilds the engine, and skips the suite
# when Postgres is unreachable. We re-evaluate the same skip condition
# here so collection-time failures are surfaced clearly per test file.
from app.services import db_service as _db  # noqa: E402


def _postgres_reachable() -> bool:
    try:
        with _db.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_reachable(),
    reason="Postgres + pgvector not reachable (run `docker compose up -d postgres`).",
)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _build_pdf_with_image(path: Path) -> str:
    """Create a tiny PDF containing text + one embedded raster image.

    Returns the file's SHA-256 (used as `doc_hash` in `documents`).
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello world. This is a test page.")
    page.insert_text((72, 120), "Another sentence to keep things small.")

    # Make a tiny 50x50 red image and embed it on the page.
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 50, 50))
    pix.set_rect(pix.irect, (255, 0, 0))
    img_bytes = pix.tobytes("png")
    page.insert_image(fitz.Rect(100, 200, 200, 300), stream=img_bytes)

    doc.save(str(path))
    doc.close()

    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _build_pdf_text_only(path: Path) -> str:
    """Build a text-only PDF and return its SHA-256 (no images)."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Plain text only document. " * 30)
    doc.save(str(path))
    doc.close()

    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


class _FakeS3:
    """In-memory replacement for `app.graph.nodes.parsers._s3`."""

    def __init__(self, local_pdf: Path) -> None:
        self._local_pdf = local_pdf
        self.uploads: dict[str, bytes] = {}

    def download_to_path(self, s3_url: str, dest_path: str) -> None:
        shutil.copyfile(self._local_pdf, dest_path)

    def download_bytes(self, s3_url: str) -> bytes:
        return self.uploads[s3_url]

    def upload_image(
        self,
        knowledge_base_type: str,
        doc_id: str,
        page: int,
        image_index: int,
        data: bytes,
        ext: str = "png",
    ) -> str:
        url = (
            f"s3://charter-rag-test/raw/"
            f"{knowledge_base_type.lower()}/images/{doc_id}/{page}_{image_index}.{ext}"
        )
        self.uploads[url] = data
        return url


def _insert_document(SessionLocal, doc_hash: str, kb_type: str = "Maintenance") -> str:
    """Insert a NEW Documents row and return its doc_id."""
    from app.models.db_models import Document

    doc_id = uuid.uuid4()
    with SessionLocal() as session:
        session.add(
            Document(
                doc_id=doc_id,
                doc_type="PDF",
                doc_hash=doc_hash,
                doc_name="sample.pdf",
                s3_url=f"s3://charter-rag-test/raw/{kb_type.lower()}/documents/{doc_hash}.pdf",
                knowledge_base_type=kb_type,
                status="NEW",
            )
        )
        session.commit()
    return str(doc_id)


def _row_count(engine, table: str, where_sql: str = "") -> int:
    with engine.begin() as conn:
        return conn.execute(text(f"SELECT count(*) FROM {table} {where_sql}")).scalar() or 0


def _patch_externals(monkeypatch, fake_s3: _FakeS3) -> None:
    """Replace S3, VLM, and embedding dependencies inside the graph nodes."""
    from app.graph.nodes import embedder, parsers, processor

    monkeypatch.setattr(parsers, "_s3", fake_s3)
    monkeypatch.setattr(processor, "summarize_image", lambda _b: "FAKE SUMMARY")
    monkeypatch.setattr(
        embedder,
        "embed_texts",
        lambda texts: [[0.1] * 1536 for _ in texts],
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_pipeline_with_image_marks_document_embedded(tmp_path, engine, SessionLocal, monkeypatch):
    pdf_path = tmp_path / "sample.pdf"
    doc_hash = _build_pdf_with_image(pdf_path)
    doc_id = _insert_document(SessionLocal, doc_hash, kb_type="Maintenance")

    fake_s3 = _FakeS3(pdf_path)
    _patch_externals(monkeypatch, fake_s3)

    # Fresh graph instance per test to avoid LangGraph internal state reuse.
    from app.graph import build_graph, initial_state

    graph = build_graph()
    final = graph.invoke(initial_state({"knowledge_base_type": "Maintenance"}, "test-job-1"))

    # ── Assertions on graph result ─────────────────────────────────────────
    result = final.get("result") or {}
    assert result.get("status") == "EMBEDDED"
    assert result.get("doc_id") == doc_id
    assert result.get("chunks", 0) > 0
    assert result.get("images", 0) == 1

    # ── Assertions on the database ─────────────────────────────────────────
    with engine.begin() as conn:
        status = conn.execute(
            text("SELECT status FROM documents WHERE doc_id = :did"), {"did": doc_id}
        ).scalar()
    assert status == "EMBEDDED"

    chunk_count = _row_count(engine, "document_chunks", f"WHERE doc_id = '{doc_id}'")
    embed_count = _row_count(
        engine,
        "embeddings",
        f"WHERE chunk_id IN (SELECT chunk_id FROM document_chunks WHERE doc_id = '{doc_id}')",
    )
    assert chunk_count > 0
    assert embed_count == chunk_count

    # ── Image was uploaded via the fake S3 ─────────────────────────────────
    assert any("/images/" in k for k in fake_s3.uploads), "image was not uploaded"


def test_pipeline_text_only_skips_image_branch(tmp_path, engine, SessionLocal, monkeypatch):
    pdf_path = tmp_path / "plain.pdf"
    doc_hash = _build_pdf_text_only(pdf_path)
    doc_id = _insert_document(SessionLocal, doc_hash, kb_type="Construction")

    fake_s3 = _FakeS3(pdf_path)
    _patch_externals(monkeypatch, fake_s3)

    # Spy on summarize_image to confirm it is NEVER called.
    called: list[bytes] = []

    from app.graph.nodes import processor

    def _spy(_blob):  # noqa: ANN001
        called.append(_blob)
        return "should-not-be-called"

    monkeypatch.setattr(processor, "summarize_image", _spy)

    from app.graph import build_graph, initial_state

    final = build_graph().invoke(
        initial_state({"knowledge_base_type": "Construction"}, "test-job-2")
    )

    result = final.get("result") or {}
    assert result.get("status") == "EMBEDDED"
    assert result.get("images", 0) == 0
    assert called == [], "summarize_image must be skipped when no images are present"

    chunk_count = _row_count(engine, "document_chunks", f"WHERE doc_id = '{doc_id}'")
    assert chunk_count > 0


def test_pipeline_marks_document_failed_when_embeddings_break(
    tmp_path, engine, SessionLocal, monkeypatch
):
    pdf_path = tmp_path / "plain.pdf"
    doc_hash = _build_pdf_text_only(pdf_path)
    doc_id = _insert_document(SessionLocal, doc_hash, kb_type="BusinessIntelligence")

    fake_s3 = _FakeS3(pdf_path)
    _patch_externals(monkeypatch, fake_s3)

    # Force embed_chunks to raise a NON-retryable error so we fail fast.
    from app.graph.nodes import embedder

    def _broken_embed(_texts):  # noqa: ANN001
        raise ValueError("embedding service contract violation")

    monkeypatch.setattr(embedder, "embed_texts", _broken_embed)

    from app.graph import build_graph, initial_state

    final = build_graph().invoke(
        initial_state({"knowledge_base_type": "BusinessIntelligence"}, "test-job-3")
    )

    result = final.get("result") or {}
    assert result.get("status") == "FAILED"
    assert result.get("failed_node") == "embed_chunks"

    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT status, last_error FROM documents WHERE doc_id = :did"),
            {"did": doc_id},
        ).one()
    assert row.status == "FAILED"
    assert row.last_error is not None
    assert row.last_error.get("failed_node") == "embed_chunks"


def test_pipeline_no_pending_docs_returns_skipped(tmp_path, monkeypatch):
    # Don't insert anything → fetch_metadata returns None → graph ends gracefully.
    fake_s3 = _FakeS3(tmp_path / "nonexistent.pdf")
    _patch_externals(monkeypatch, fake_s3)

    from app.graph import build_graph, initial_state

    final = build_graph().invoke(initial_state({}, "test-job-empty"))
    assert final.get("doc_metadata") is None
    assert final.get("result") in (None, {})
