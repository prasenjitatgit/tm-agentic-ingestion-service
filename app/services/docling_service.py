"""Docling document conversion service.

Wraps IBM Docling's DocumentConverter with lazy initialization and
configurable pipeline options (OCR, device, table mode). Docling modules
are only imported when the converter is first used (lazy loading) to
keep startup fast.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from app.core.config import Settings
from app.utils.logger import get_logger

if TYPE_CHECKING:
    from docling.document_converter import ConversionResult, DocumentConverter

log = get_logger(__name__)


@dataclass
class ImageData:
    """Image extracted by Docling with position metadata."""

    page: int
    image_index: int
    image_bytes: bytes
    format: str  # "png", "jpeg", etc.
    position_in_markdown: int  # character offset in the Markdown output


class DoclingConverterService:
    """Lazy-initialized Docling converter with configurable pipeline options.

    The underlying DocumentConverter is only instantiated on first use,
    keeping memory footprint low when the service is constructed but not
    yet needed (e.g. during app startup health checks).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._converter: DocumentConverter | None = None

    @property
    def converter(self) -> "DocumentConverter":
        """Lazy initialization — only loads models when first used."""
        if self._converter is None:
            self._converter = self._build_converter()
        return self._converter

    def _build_converter(self) -> "DocumentConverter":
        """Configure and return a DocumentConverter based on settings."""
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
        from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

        log.info(
            "building_docling_converter",
            ocr_enabled=self._settings.docling_ocr_enabled,
            device=self._settings.docling_device,
            table_mode=self._settings.docling_table_mode,
        )

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = self._settings.docling_ocr_enabled
        pipeline_options.do_table_structure = True

        if self._settings.docling_table_mode == "fast":
            from docling.datamodel.pipeline_options import TableFormerMode

            pipeline_options.table_structure_options.mode = TableFormerMode.FAST
        else:
            from docling.datamodel.pipeline_options import TableFormerMode

            pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

        pipeline_options.images_scale = 1.0
        pipeline_options.generate_picture_images = True

        converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.DOCX,
                InputFormat.PPTX,
                InputFormat.XLSX,
            ],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=StandardPdfPipeline,
                    pipeline_options=pipeline_options,
                ),
            },
        )

        log.info("docling_converter_ready")
        return converter

    def convert(self, file_path: str) -> "ConversionResult":
        """Convert a document file to DoclingDocument.

        Args:
            file_path: Path to the local document file.

        Returns:
            A ConversionResult containing the parsed document.

        Raises:
            FileNotFoundError: If the file does not exist.
            Various Docling exceptions on conversion failure.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        log.info("docling_convert_start", file_path=file_path)
        result = self.converter.convert(str(path))
        log.info("docling_convert_complete", file_path=file_path)
        return result

    def extract_markdown(self, result: "ConversionResult") -> str:
        """Export conversion result to Markdown with image placeholders.

        Args:
            result: A ConversionResult from a prior convert() call.

        Returns:
            Markdown string representation of the document.
        """
        markdown = result.document.export_to_markdown()
        log.debug("markdown_extracted", length=len(markdown))
        return markdown

    def extract_images(self, result: "ConversionResult") -> list[ImageData]:
        """Extract image binary data with page/position metadata.

        Iterates over PictureItems in the Docling document and exports
        each as PNG bytes along with page number and positional info.

        Args:
            result: A ConversionResult from a prior convert() call.

        Returns:
            List of ImageData with binary content and metadata.
        """
        from docling.datamodel.document import PictureItem

        images: list[ImageData] = []
        markdown = result.document.export_to_markdown()

        page_image_counts: dict[int, int] = {}

        for element, _level in result.document.iterate_items():
            if not isinstance(element, PictureItem):
                continue

            # Determine page number from provenance
            page = 1
            if element.prov and len(element.prov) > 0:
                page = element.prov[0].page_no

            # Track image index per page
            page_image_counts[page] = page_image_counts.get(page, 0) + 1
            image_index = page_image_counts[page]

            # Export image to bytes
            image_bytes = b""
            image_format = "png"
            if element.image is not None:
                import io

                buf = io.BytesIO()
                element.image.pil_image.save(buf, format="PNG")
                image_bytes = buf.getvalue()

            # Estimate position in markdown based on element text/caption
            position = 0
            if element.caption_text(result.document):
                caption = element.caption_text(result.document)
                pos = markdown.find(caption)
                if pos >= 0:
                    position = pos

            images.append(
                ImageData(
                    page=page,
                    image_index=image_index,
                    image_bytes=image_bytes,
                    format=image_format,
                    position_in_markdown=position,
                )
            )

        log.info("images_extracted", count=len(images))
        return images
