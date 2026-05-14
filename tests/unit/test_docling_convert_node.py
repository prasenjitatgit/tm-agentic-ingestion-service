"""Unit tests for the docling_convert_node function.

Tests image extraction and S3 upload path format, speaker notes fallback
for PPTX, VLM circuit breaker fallback text, and empty document handling.

Validates: Requirements 3.1, 3.5, 5.2, 6.1
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from app.graph.state import (
    AgentState,
    DocMetadata,
    ImageItem,
    RetryContext,
    TextBlock,
)
from app.services.docling_service import ImageData


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_state(
    doc_type: str = "PDF",
    knowledge_base_type: str = "Maintenance",
    doc_id: str = "doc-abc-123",
) -> AgentState:
    """Build a minimal AgentState for testing."""
    return AgentState(
        doc_metadata=DocMetadata(
            doc_id=doc_id,
            doc_hash="sha256_abc",
            doc_name="test_document.pdf",
            s3_url="s3://bucket/raw/Maintenance/documents/test_document.pdf",
            doc_type=doc_type,
            knowledge_base_type=knowledge_base_type,
        ),
        local_path="/tmp/test_document.pdf",
        extracted_image_items=[],
        extracted_text_blocks=[],
        assembled_texts=[],
        chunks=[],
        error=None,
        retry_context=RetryContext(),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Test: Image extraction and S3 upload path format
#  Validates: Requirement 3.1
# ─────────────────────────────────────────────────────────────────────────────


class TestImageExtractionAndS3Upload:
    """Verify that images are extracted and uploaded to S3 with correct path format."""

    @patch("app.graph.nodes.docling_converter._s3")
    @patch("app.graph.nodes.docling_converter.DoclingConverterService")
    @patch("app.graph.nodes.docling_converter.get_settings")
    def test_images_uploaded_with_correct_path_format(
        self, mock_get_settings, mock_service_cls, mock_s3
    ):
        """Requirement 3.1: Images uploaded to S3 using path convention
        raw/{kb}/images/{doc_id}/{page}_{index}.{ext}."""
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings

        mock_service = MagicMock()
        mock_service_cls.return_value = mock_service

        # Mock conversion result
        mock_result = MagicMock()
        mock_service.convert.return_value = mock_result
        mock_service.extract_markdown.return_value = "# Test Document\n\nSome content here."

        # Mock image extraction - two images
        mock_service.extract_images.return_value = [
            ImageData(
                page=1,
                image_index=1,
                image_bytes=b"\x89PNG\r\n\x1a\n" + b"\x00" * 100,
                format="png",
                position_in_markdown=5,
            ),
            ImageData(
                page=2,
                image_index=1,
                image_bytes=b"\xff\xd8\xff\xe0" + b"\x00" * 50,
                format="jpeg",
                position_in_markdown=20,
            ),
        ]

        # Mock S3 upload returning expected URLs
        mock_s3.upload_image.side_effect = [
            "s3://bucket/raw/Maintenance/images/doc-abc-123/1_1.png",
            "s3://bucket/raw/Maintenance/images/doc-abc-123/2_1.jpeg",
        ]

        state = _make_state()

        from app.graph.nodes.docling_converter import docling_convert_node

        result = docling_convert_node(state)

        # Verify S3 upload was called with correct arguments
        calls = mock_s3.upload_image.call_args_list
        assert len(calls) == 2

        # First image: page=1, index=1, png
        # upload_image(kb_type, doc_id, page, image_index, data, ext=...)
        assert calls[0].args[0] == "Maintenance"
        assert calls[0].args[1] == "doc-abc-123"
        assert calls[0].args[2] == 1  # page
        assert calls[0].args[3] == 1  # image_index
        assert calls[0].kwargs["ext"] == "png"

        # Second image: page=2, index=1, jpeg
        assert calls[1].args[0] == "Maintenance"
        assert calls[1].args[1] == "doc-abc-123"
        assert calls[1].args[2] == 2  # page
        assert calls[1].args[3] == 1  # image_index
        assert calls[1].kwargs["ext"] == "jpeg"

        # Verify ImageItems in result
        image_items = result["extracted_image_items"]
        assert len(image_items) == 2
        assert image_items[0].s3_url == "s3://bucket/raw/Maintenance/images/doc-abc-123/1_1.png"
        assert image_items[0].page == 1
        assert image_items[0].image_index == 1
        assert image_items[1].s3_url == "s3://bucket/raw/Maintenance/images/doc-abc-123/2_1.jpeg"
        assert image_items[1].page == 2
        assert image_items[1].image_index == 1

    @patch("app.graph.nodes.docling_converter._s3")
    @patch("app.graph.nodes.docling_converter.DoclingConverterService")
    @patch("app.graph.nodes.docling_converter.get_settings")
    def test_empty_image_bytes_skipped(
        self, mock_get_settings, mock_service_cls, mock_s3
    ):
        """Images with empty bytes are skipped and not uploaded."""
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings

        mock_service = MagicMock()
        mock_service_cls.return_value = mock_service

        mock_result = MagicMock()
        mock_service.convert.return_value = mock_result
        mock_service.extract_markdown.return_value = "# Test\n\nContent."

        # One image with empty bytes, one with valid bytes
        mock_service.extract_images.return_value = [
            ImageData(
                page=1,
                image_index=1,
                image_bytes=b"",  # Empty - should be skipped
                format="png",
                position_in_markdown=5,
            ),
            ImageData(
                page=1,
                image_index=2,
                image_bytes=b"\x89PNG" + b"\x00" * 50,
                format="png",
                position_in_markdown=10,
            ),
        ]

        mock_s3.upload_image.return_value = (
            "s3://bucket/raw/Maintenance/images/doc-abc-123/1_2.png"
        )

        state = _make_state()

        from app.graph.nodes.docling_converter import docling_convert_node

        result = docling_convert_node(state)

        # Only one image should be uploaded (the non-empty one)
        assert mock_s3.upload_image.call_count == 1
        assert len(result["extracted_image_items"]) == 1


# ─────────────────────────────────────────────────────────────────────────────
#  Test: Speaker notes fallback for PPTX
#  Validates: Requirement 5.2
# ─────────────────────────────────────────────────────────────────────────────


class TestSpeakerNotesFallback:
    """Verify that speaker notes are extracted via python-pptx fallback for PPTX."""

    @patch("app.graph.nodes.docling_converter._extract_speaker_notes_fallback")
    @patch("app.graph.nodes.docling_converter._s3")
    @patch("app.graph.nodes.docling_converter.DoclingConverterService")
    @patch("app.graph.nodes.docling_converter.get_settings")
    def test_speaker_notes_fallback_triggered_for_pptx(
        self, mock_get_settings, mock_service_cls, mock_s3, mock_fallback
    ):
        """Requirement 5.2: If Docling does not extract speaker notes from PPTX,
        fall back to python-pptx extraction."""
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings

        mock_service = MagicMock()
        mock_service_cls.return_value = mock_service

        mock_result = MagicMock()
        mock_service.convert.return_value = mock_result
        # Markdown without speaker notes markers
        mock_service.extract_markdown.return_value = (
            "# Slide 1\n\nContent of slide 1.\n\n# Slide 2\n\nContent of slide 2."
        )
        mock_service.extract_images.return_value = []

        # Fallback returns notes for slides
        mock_fallback.return_value = {
            1: "These are the notes for slide 1.",
            2: "Notes for slide 2.",
        }

        state = _make_state(doc_type="PPT")

        from app.graph.nodes.docling_converter import docling_convert_node

        result = docling_convert_node(state)

        # Verify fallback was called with the local path
        mock_fallback.assert_called_once_with("/tmp/test_document.pdf")

        # Verify speaker notes appear in the text blocks
        all_text = " ".join(block.text for block in result["extracted_text_blocks"])
        assert "These are the notes for slide 1." in all_text
        assert "Notes for slide 2." in all_text

    @patch("app.graph.nodes.docling_converter._extract_speaker_notes_fallback")
    @patch("app.graph.nodes.docling_converter._s3")
    @patch("app.graph.nodes.docling_converter.DoclingConverterService")
    @patch("app.graph.nodes.docling_converter.get_settings")
    def test_speaker_notes_fallback_not_triggered_when_notes_present(
        self, mock_get_settings, mock_service_cls, mock_s3, mock_fallback
    ):
        """Requirement 5.2: Fallback is NOT triggered when Docling already
        extracted speaker notes."""
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings

        mock_service = MagicMock()
        mock_service_cls.return_value = mock_service

        mock_result = MagicMock()
        mock_service.convert.return_value = mock_result
        # Markdown WITH speaker notes markers (Docling extracted them)
        mock_service.extract_markdown.return_value = (
            "# Slide 1\n\nContent.\n\n"
            "<!-- SPEAKER_NOTES slide=1 -->\n"
            "Notes from Docling.\n"
            "<!-- /SPEAKER_NOTES -->"
        )
        mock_service.extract_images.return_value = []

        state = _make_state(doc_type="PPT")

        from app.graph.nodes.docling_converter import docling_convert_node

        result = docling_convert_node(state)

        # Fallback should NOT be called
        mock_fallback.assert_not_called()

    @patch("app.graph.nodes.docling_converter._extract_speaker_notes_fallback")
    @patch("app.graph.nodes.docling_converter._s3")
    @patch("app.graph.nodes.docling_converter.DoclingConverterService")
    @patch("app.graph.nodes.docling_converter.get_settings")
    def test_speaker_notes_fallback_not_triggered_for_non_pptx(
        self, mock_get_settings, mock_service_cls, mock_s3, mock_fallback
    ):
        """Speaker notes fallback is only for PPT doc_type."""
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings

        mock_service = MagicMock()
        mock_service_cls.return_value = mock_service

        mock_result = MagicMock()
        mock_service.convert.return_value = mock_result
        mock_service.extract_markdown.return_value = "# Document\n\nContent."
        mock_service.extract_images.return_value = []

        state = _make_state(doc_type="PDF")

        from app.graph.nodes.docling_converter import docling_convert_node

        result = docling_convert_node(state)

        # Fallback should NOT be called for PDF
        mock_fallback.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
#  Test: VLM circuit breaker fallback text
#  Validates: Requirement 3.5
# ─────────────────────────────────────────────────────────────────────────────


class TestVLMCircuitBreakerFallback:
    """Verify that the VLM fallback text is used when circuit breaker is open."""

    @patch("app.graph.nodes.docling_converter._s3")
    @patch("app.graph.nodes.docling_converter.DoclingConverterService")
    @patch("app.graph.nodes.docling_converter.get_settings")
    def test_image_reference_contains_summary_placeholder(
        self, mock_get_settings, mock_service_cls, mock_s3
    ):
        """Requirement 3.5: Image references include a summary field that can
        hold '[image summary unavailable]' when VLM circuit breaker is open.

        The docling_convert_node inserts image reference markers with an empty
        summary field. The VLM summarization step fills it in later. When the
        circuit breaker is open, the downstream node uses the fallback text.
        Here we verify the marker structure supports the fallback pattern.
        """
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings

        mock_service = MagicMock()
        mock_service_cls.return_value = mock_service

        mock_result = MagicMock()
        mock_service.convert.return_value = mock_result
        mock_service.extract_markdown.return_value = "# Test\n\nSome content here."

        mock_service.extract_images.return_value = [
            ImageData(
                page=1,
                image_index=1,
                image_bytes=b"\x89PNG" + b"\x00" * 50,
                format="png",
                position_in_markdown=5,
            ),
        ]

        mock_s3.upload_image.return_value = (
            "s3://bucket/raw/Maintenance/images/doc-abc-123/1_1.png"
        )

        state = _make_state()

        from app.graph.nodes.docling_converter import docling_convert_node

        result = docling_convert_node(state)

        # Verify the text blocks contain the image reference with Summary field
        all_text = " ".join(block.text for block in result["extracted_text_blocks"])
        assert "**Summary:**" in all_text, (
            "Image reference should contain a Summary field for VLM or fallback text"
        )

        # The ImageItem should have an empty summary (to be filled by VLM or fallback)
        image_items = result["extracted_image_items"]
        assert len(image_items) == 1
        assert image_items[0].summary == "", (
            "ImageItem summary should be empty at this stage (filled by VLM node)"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Test: Empty document handling
#  Validates: Requirement 6.1
# ─────────────────────────────────────────────────────────────────────────────


class TestEmptyDocumentHandling:
    """Verify handling of documents with no text or image-only content."""

    @patch("app.graph.nodes.docling_converter._s3")
    @patch("app.graph.nodes.docling_converter.DoclingConverterService")
    @patch("app.graph.nodes.docling_converter.get_settings")
    def test_empty_document_no_text_no_images(
        self, mock_get_settings, mock_service_cls, mock_s3
    ):
        """Requirement 6.1: Empty document produces empty text blocks and image items."""
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings

        mock_service = MagicMock()
        mock_service_cls.return_value = mock_service

        mock_result = MagicMock()
        mock_service.convert.return_value = mock_result
        mock_service.extract_markdown.return_value = ""  # Empty markdown
        mock_service.extract_images.return_value = []

        state = _make_state()

        from app.graph.nodes.docling_converter import docling_convert_node

        result = docling_convert_node(state)

        assert result["extracted_text_blocks"] == []
        assert result["extracted_image_items"] == []

    @patch("app.graph.nodes.docling_converter._s3")
    @patch("app.graph.nodes.docling_converter.DoclingConverterService")
    @patch("app.graph.nodes.docling_converter.get_settings")
    def test_whitespace_only_document(
        self, mock_get_settings, mock_service_cls, mock_s3
    ):
        """Requirement 6.1: Document with only whitespace produces no text blocks."""
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings

        mock_service = MagicMock()
        mock_service_cls.return_value = mock_service

        mock_result = MagicMock()
        mock_service.convert.return_value = mock_result
        mock_service.extract_markdown.return_value = "   \n\n   \t  "
        mock_service.extract_images.return_value = []

        state = _make_state()

        from app.graph.nodes.docling_converter import docling_convert_node

        result = docling_convert_node(state)

        assert result["extracted_text_blocks"] == []
        assert result["extracted_image_items"] == []

    @patch("app.graph.nodes.docling_converter._s3")
    @patch("app.graph.nodes.docling_converter.DoclingConverterService")
    @patch("app.graph.nodes.docling_converter.get_settings")
    def test_image_only_document(
        self, mock_get_settings, mock_service_cls, mock_s3
    ):
        """Requirement 6.1: Image-only document produces image items and
        text blocks containing image references."""
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings

        mock_service = MagicMock()
        mock_service_cls.return_value = mock_service

        mock_result = MagicMock()
        mock_service.convert.return_value = mock_result
        # Minimal markdown (just a newline) but with images
        mock_service.extract_markdown.return_value = "\n"

        mock_service.extract_images.return_value = [
            ImageData(
                page=1,
                image_index=1,
                image_bytes=b"\x89PNG" + b"\x00" * 50,
                format="png",
                position_in_markdown=0,
            ),
        ]

        mock_s3.upload_image.return_value = (
            "s3://bucket/raw/Maintenance/images/doc-abc-123/1_1.png"
        )

        state = _make_state()

        from app.graph.nodes.docling_converter import docling_convert_node

        result = docling_convert_node(state)

        # Should have image items
        assert len(result["extracted_image_items"]) == 1
        assert result["extracted_image_items"][0].page == 1

        # Should have text blocks containing the image reference
        assert len(result["extracted_text_blocks"]) >= 1


# ─────────────────────────────────────────────────────────────────────────────
#  Test: Correct output schema
# ─────────────────────────────────────────────────────────────────────────────


class TestOutputSchema:
    """Verify the node returns the correct output schema."""

    @patch("app.graph.nodes.docling_converter._s3")
    @patch("app.graph.nodes.docling_converter.DoclingConverterService")
    @patch("app.graph.nodes.docling_converter.get_settings")
    def test_output_contains_required_keys(
        self, mock_get_settings, mock_service_cls, mock_s3
    ):
        """The node returns dict with extracted_text_blocks and extracted_image_items."""
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings

        mock_service = MagicMock()
        mock_service_cls.return_value = mock_service

        mock_result = MagicMock()
        mock_service.convert.return_value = mock_result
        mock_service.extract_markdown.return_value = "# Title\n\nParagraph text."
        mock_service.extract_images.return_value = []

        state = _make_state()

        from app.graph.nodes.docling_converter import docling_convert_node

        result = docling_convert_node(state)

        assert "extracted_text_blocks" in result
        assert "extracted_image_items" in result
        assert isinstance(result["extracted_text_blocks"], list)
        assert isinstance(result["extracted_image_items"], list)

    @patch("app.graph.nodes.docling_converter._s3")
    @patch("app.graph.nodes.docling_converter.DoclingConverterService")
    @patch("app.graph.nodes.docling_converter.get_settings")
    def test_text_blocks_have_correct_fields(
        self, mock_get_settings, mock_service_cls, mock_s3
    ):
        """TextBlock objects have page, section, and text fields."""
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings

        mock_service = MagicMock()
        mock_service_cls.return_value = mock_service

        mock_result = MagicMock()
        mock_service.convert.return_value = mock_result
        mock_service.extract_markdown.return_value = (
            "# Introduction\n\nThis is the intro.\n\n"
            "## Methods\n\nMethodology section."
        )
        mock_service.extract_images.return_value = []

        state = _make_state()

        from app.graph.nodes.docling_converter import docling_convert_node

        result = docling_convert_node(state)

        text_blocks = result["extracted_text_blocks"]
        assert len(text_blocks) >= 1

        for block in text_blocks:
            assert isinstance(block, TextBlock)
            assert isinstance(block.page, int)
            assert block.page >= 1
            assert isinstance(block.text, str)
            assert len(block.text) > 0

    @patch("app.graph.nodes.docling_converter._s3")
    @patch("app.graph.nodes.docling_converter.DoclingConverterService")
    @patch("app.graph.nodes.docling_converter.get_settings")
    def test_service_called_with_local_path(
        self, mock_get_settings, mock_service_cls, mock_s3
    ):
        """The service.convert() is called with the state's local_path."""
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings

        mock_service = MagicMock()
        mock_service_cls.return_value = mock_service

        mock_result = MagicMock()
        mock_service.convert.return_value = mock_result
        mock_service.extract_markdown.return_value = "# Doc\n\nText."
        mock_service.extract_images.return_value = []

        state = _make_state()

        from app.graph.nodes.docling_converter import docling_convert_node

        docling_convert_node(state)

        mock_service.convert.assert_called_once_with("/tmp/test_document.pdf")
