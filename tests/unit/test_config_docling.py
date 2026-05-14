"""Tests for Docling configuration defaults and routing.

Validates: Requirements 8.2, 8.3
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.graph.state import DocMetadata
from app.graph.workflow import route_after_select_parser


class TestDoclingConfigDefaults:
    """Verify Docling-related configuration defaults."""

    def test_docling_ocr_enabled_defaults_to_false(self, monkeypatch):
        """Requirement 8.2: DOCLING_OCR_ENABLED defaults to false."""
        monkeypatch.delenv("DOCLING_OCR_ENABLED", raising=False)
        settings = Settings()
        assert settings.docling_ocr_enabled is False

    def test_docling_device_defaults_to_cpu(self, monkeypatch):
        """Requirement 8.3: DOCLING_DEVICE defaults to 'cpu'."""
        monkeypatch.delenv("DOCLING_DEVICE", raising=False)
        settings = Settings()
        assert settings.docling_device == "cpu"


class TestDoclingRouting:
    """Verify that all documents route to docling_convert."""

    def _make_state(self, doc_type: str) -> dict:
        """Build a minimal AgentState-like dict for routing."""
        return {
            "doc_metadata": DocMetadata(
                doc_id="test-id",
                doc_hash="abc123",
                doc_name="test.pdf",
                s3_url="s3://bucket/key",
                doc_type=doc_type,
                knowledge_base_type="Maintenance",
            ),
            "error": None,
        }

    def test_pdf_routes_to_docling_convert(self):
        state = self._make_state("PDF")
        assert route_after_select_parser(state) == "docling_convert"

    def test_docx_routes_to_docling_convert(self):
        state = self._make_state("DOCX")
        assert route_after_select_parser(state) == "docling_convert"

    def test_pptx_routes_to_docling_convert(self):
        state = self._make_state("PPT")
        assert route_after_select_parser(state) == "docling_convert"

    def test_excel_routes_to_docling_convert(self):
        state = self._make_state("EXCEL")
        assert route_after_select_parser(state) == "docling_convert"

    def test_error_state_routes_to_retry_handler(self):
        state = self._make_state("PDF")
        state["error"] = {"type": "RuntimeError", "message": "something broke"}
        assert route_after_select_parser(state) == "retry_handler"
