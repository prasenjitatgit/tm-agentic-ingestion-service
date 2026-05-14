"""Tests for DoclingConverterService lazy initialization and configuration.

Validates: Requirements 1.6
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from app.core.config import Settings


class TestLazyInitialization:
    """Verify that the converter is not created until first use."""

    def test_converter_is_none_after_instantiation(self):
        """Requirement 1.6: Converter is only instantiated on first use."""
        settings = Settings()

        from app.services.docling_service import DoclingConverterService

        service = DoclingConverterService(settings)
        assert service._converter is None

    def test_build_converter_called_on_first_property_access(self):
        """Requirement 1.6: Accessing .converter triggers _build_converter()."""
        settings = Settings()

        from app.services.docling_service import DoclingConverterService

        service = DoclingConverterService(settings)

        mock_converter = MagicMock()
        with patch.object(service, "_build_converter", return_value=mock_converter) as mock_build:
            result = service.converter
            mock_build.assert_called_once()
            assert result is mock_converter

    def test_build_converter_not_called_twice(self):
        """Requirement 1.6: Subsequent .converter accesses reuse the instance."""
        settings = Settings()

        from app.services.docling_service import DoclingConverterService

        service = DoclingConverterService(settings)

        mock_converter = MagicMock()
        with patch.object(service, "_build_converter", return_value=mock_converter) as mock_build:
            _ = service.converter
            _ = service.converter
            mock_build.assert_called_once()


class TestConfigurationPropagation:
    """Verify that OCR, device, and table mode settings are passed to Docling."""

    def test_ocr_enabled_propagated(self, monkeypatch):
        """Requirement 1.6: OCR setting is passed to PdfPipelineOptions."""
        monkeypatch.setenv("DOCLING_OCR_ENABLED", "true")
        settings = Settings()

        mock_pipeline_options = MagicMock()
        mock_pipeline_options_cls = MagicMock(return_value=mock_pipeline_options)
        mock_converter_cls = MagicMock()
        mock_input_format = MagicMock()
        mock_standard_pipeline = MagicMock()
        mock_table_former_mode = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "docling": MagicMock(),
                "docling.document_converter": MagicMock(DocumentConverter=mock_converter_cls),
                "docling.datamodel": MagicMock(),
                "docling.datamodel.pipeline_options": MagicMock(
                    PdfPipelineOptions=mock_pipeline_options_cls,
                    TableFormerMode=mock_table_former_mode,
                ),
                "docling.datamodel.base_models": MagicMock(InputFormat=mock_input_format),
                "docling.pipeline": MagicMock(),
                "docling.pipeline.standard_pdf_pipeline": MagicMock(
                    StandardPdfPipeline=mock_standard_pipeline
                ),
            },
        ):
            from app.services.docling_service import DoclingConverterService

            service = DoclingConverterService(settings)
            service._build_converter()

            assert mock_pipeline_options.do_ocr is True

    def test_ocr_disabled_propagated(self, monkeypatch):
        """Requirement 1.6: OCR disabled setting is passed to PdfPipelineOptions."""
        monkeypatch.setenv("DOCLING_OCR_ENABLED", "false")
        settings = Settings()

        mock_pipeline_options = MagicMock()
        mock_pipeline_options_cls = MagicMock(return_value=mock_pipeline_options)
        mock_converter_cls = MagicMock()
        mock_input_format = MagicMock()
        mock_standard_pipeline = MagicMock()
        mock_table_former_mode = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "docling": MagicMock(),
                "docling.document_converter": MagicMock(DocumentConverter=mock_converter_cls),
                "docling.datamodel": MagicMock(),
                "docling.datamodel.pipeline_options": MagicMock(
                    PdfPipelineOptions=mock_pipeline_options_cls,
                    TableFormerMode=mock_table_former_mode,
                ),
                "docling.datamodel.base_models": MagicMock(InputFormat=mock_input_format),
                "docling.pipeline": MagicMock(),
                "docling.pipeline.standard_pdf_pipeline": MagicMock(
                    StandardPdfPipeline=mock_standard_pipeline
                ),
            },
        ):
            from app.services.docling_service import DoclingConverterService

            service = DoclingConverterService(settings)
            service._build_converter()

            assert mock_pipeline_options.do_ocr is False

    def test_table_mode_fast_propagated(self, monkeypatch):
        """Requirement 1.6: Table mode 'fast' sets TableFormerMode.FAST."""
        monkeypatch.setenv("DOCLING_TABLE_MODE", "fast")
        settings = Settings()

        mock_pipeline_options = MagicMock()
        mock_pipeline_options_cls = MagicMock(return_value=mock_pipeline_options)
        mock_converter_cls = MagicMock()
        mock_input_format = MagicMock()
        mock_standard_pipeline = MagicMock()
        mock_table_former_mode = MagicMock()
        mock_table_former_mode.FAST = "FAST"
        mock_table_former_mode.ACCURATE = "ACCURATE"

        with patch.dict(
            sys.modules,
            {
                "docling": MagicMock(),
                "docling.document_converter": MagicMock(DocumentConverter=mock_converter_cls),
                "docling.datamodel": MagicMock(),
                "docling.datamodel.pipeline_options": MagicMock(
                    PdfPipelineOptions=mock_pipeline_options_cls,
                    TableFormerMode=mock_table_former_mode,
                ),
                "docling.datamodel.base_models": MagicMock(InputFormat=mock_input_format),
                "docling.pipeline": MagicMock(),
                "docling.pipeline.standard_pdf_pipeline": MagicMock(
                    StandardPdfPipeline=mock_standard_pipeline
                ),
            },
        ):
            from app.services.docling_service import DoclingConverterService

            service = DoclingConverterService(settings)
            service._build_converter()

            assert (
                mock_pipeline_options.table_structure_options.mode
                == mock_table_former_mode.FAST
            )

    def test_table_mode_accurate_propagated(self, monkeypatch):
        """Requirement 1.6: Table mode 'accurate' sets TableFormerMode.ACCURATE."""
        monkeypatch.setenv("DOCLING_TABLE_MODE", "accurate")
        settings = Settings()

        mock_pipeline_options = MagicMock()
        mock_pipeline_options_cls = MagicMock(return_value=mock_pipeline_options)
        mock_converter_cls = MagicMock()
        mock_input_format = MagicMock()
        mock_standard_pipeline = MagicMock()
        mock_table_former_mode = MagicMock()
        mock_table_former_mode.FAST = "FAST"
        mock_table_former_mode.ACCURATE = "ACCURATE"

        with patch.dict(
            sys.modules,
            {
                "docling": MagicMock(),
                "docling.document_converter": MagicMock(DocumentConverter=mock_converter_cls),
                "docling.datamodel": MagicMock(),
                "docling.datamodel.pipeline_options": MagicMock(
                    PdfPipelineOptions=mock_pipeline_options_cls,
                    TableFormerMode=mock_table_former_mode,
                ),
                "docling.datamodel.base_models": MagicMock(InputFormat=mock_input_format),
                "docling.pipeline": MagicMock(),
                "docling.pipeline.standard_pdf_pipeline": MagicMock(
                    StandardPdfPipeline=mock_standard_pipeline
                ),
            },
        ):
            from app.services.docling_service import DoclingConverterService

            service = DoclingConverterService(settings)
            service._build_converter()

            assert (
                mock_pipeline_options.table_structure_options.mode
                == mock_table_former_mode.ACCURATE
            )

    def test_document_converter_receives_allowed_formats(self):
        """Requirement 1.6: DocumentConverter is configured with PDF, DOCX, PPTX, XLSX."""
        settings = Settings()

        mock_pipeline_options = MagicMock()
        mock_pipeline_options_cls = MagicMock(return_value=mock_pipeline_options)
        mock_converter_cls = MagicMock()
        mock_input_format = MagicMock()
        mock_input_format.PDF = "PDF"
        mock_input_format.DOCX = "DOCX"
        mock_input_format.PPTX = "PPTX"
        mock_input_format.XLSX = "XLSX"
        mock_standard_pipeline = MagicMock()
        mock_table_former_mode = MagicMock()
        mock_table_former_mode.ACCURATE = "ACCURATE"

        with patch.dict(
            sys.modules,
            {
                "docling": MagicMock(),
                "docling.document_converter": MagicMock(DocumentConverter=mock_converter_cls),
                "docling.datamodel": MagicMock(),
                "docling.datamodel.pipeline_options": MagicMock(
                    PdfPipelineOptions=mock_pipeline_options_cls,
                    TableFormerMode=mock_table_former_mode,
                ),
                "docling.datamodel.base_models": MagicMock(InputFormat=mock_input_format),
                "docling.pipeline": MagicMock(),
                "docling.pipeline.standard_pdf_pipeline": MagicMock(
                    StandardPdfPipeline=mock_standard_pipeline
                ),
            },
        ):
            from app.services.docling_service import DoclingConverterService

            service = DoclingConverterService(settings)
            service._build_converter()

            call_kwargs = mock_converter_cls.call_args
            allowed = call_kwargs[1]["allowed_formats"] if call_kwargs[1] else call_kwargs[0][0]
            assert "PDF" in allowed
            assert "DOCX" in allowed
            assert "PPTX" in allowed
            assert "XLSX" in allowed


class TestLazyImport:
    """Verify that Docling modules are not imported at service instantiation time."""

    def test_service_instantiation_does_not_import_docling(self, monkeypatch):
        """Creating the service does not trigger docling imports."""
        settings = Settings()

        # Remove any cached docling modules to ensure clean state
        docling_modules = [key for key in sys.modules if key.startswith("docling")]
        for mod in docling_modules:
            monkeypatch.delitem(sys.modules, mod, raising=False)

        from app.services.docling_service import DoclingConverterService

        service = DoclingConverterService(settings)

        # Verify no docling runtime modules were imported (TYPE_CHECKING imports don't count)
        runtime_docling_modules = [
            key for key in sys.modules
            if key.startswith("docling") and sys.modules[key] is not None
        ]
        assert service._converter is None
        assert len(runtime_docling_modules) == 0, (
            f"Docling modules should not be imported at instantiation: {runtime_docling_modules}"
        )
