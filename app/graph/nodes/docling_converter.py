"""Docling conversion node: unified document → Markdown → TextBlocks + ImageItems.

Converts all supported document formats (PDF, DOCX, PPTX, XLSX) via IBM
Docling, extracts images to S3, handles speaker notes fallback for PPTX,
and splits the resulting Markdown into TextBlock objects.
"""

from __future__ import annotations

import re
from typing import Any

from app.core.config import get_settings
from app.graph.nodes.status import node
from app.graph.state import AgentState, ImageItem, TextBlock
from app.services.docling_service import DoclingConverterService
from app.services.s3_service import S3Client
from app.utils.logger import get_logger

log = get_logger(__name__)

_s3 = S3Client()

# Regex to detect speaker notes markers in Markdown output from Docling.
_SPEAKER_NOTES_RE = re.compile(
    r"<!-- SPEAKER_NOTES slide=\d+ -->", flags=re.IGNORECASE
)

# Heading pattern for splitting Markdown into sections.
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", flags=re.MULTILINE)

# Page break marker that Docling may insert.
_PAGE_BREAK_RE = re.compile(r"---\s*\n|<!-- page \d+ -->", flags=re.IGNORECASE)


def _extract_speaker_notes_fallback(pptx_path: str) -> dict[int, str]:
    """Use python-pptx to extract speaker notes as fallback.

    Returns a mapping of slide number (1-based) → notes text.
    """
    from pptx import Presentation

    notes: dict[int, str] = {}
    prs = Presentation(pptx_path)
    for slide_no, slide in enumerate(prs.slides, start=1):
        if slide.has_notes_slide:
            text = slide.notes_slide.notes_text_frame.text.strip()
            if text:
                notes[slide_no] = text
    return notes


def _insert_image_references(
    markdown: str,
    image_items: list[ImageItem],
    image_positions: list[int],
) -> str:
    """Insert HTML comment image reference markers into Markdown at original positions.

    Inserts in reverse order so earlier offsets remain valid.
    """
    # Pair items with positions and sort by position descending.
    pairs = sorted(
        zip(image_positions, image_items),
        key=lambda x: x[0],
        reverse=True,
    )

    for position, item in pairs:
        marker = (
            f"<!-- IMAGE_REF: {item.s3_url} page={item.page} index={item.image_index} -->\n"
            f"![Image]({item.s3_url})\n"
            f"**Summary:** \n"
            f"<!-- /IMAGE_REF -->"
        )
        # Insert at the position (or end if position exceeds length).
        pos = min(position, len(markdown))
        # Find a good insertion point (after the current line).
        newline_pos = markdown.find("\n", pos)
        if newline_pos == -1:
            insert_at = len(markdown)
        else:
            insert_at = newline_pos + 1

        markdown = markdown[:insert_at] + marker + "\n" + markdown[insert_at:]

    return markdown


def _append_speaker_notes(markdown: str, notes: dict[int, str]) -> str:
    """Append speaker notes blocks to the Markdown using HTML comment markers."""
    for slide_no, text in sorted(notes.items()):
        block = (
            f"\n<!-- SPEAKER_NOTES slide={slide_no} -->\n"
            f"{text}\n"
            f"<!-- /SPEAKER_NOTES -->\n"
        )
        markdown += block
    return markdown


def _split_markdown_to_text_blocks(markdown: str) -> list[TextBlock]:
    """Split Markdown into TextBlock objects by heading-based sections.

    Each section is delimited by a heading (H1-H3). Content before the
    first heading goes into a default section. Page numbers are estimated
    from page break markers or section index.
    """
    if not markdown.strip():
        return []

    blocks: list[TextBlock] = []

    # Find all headings with their positions.
    headings = list(_HEADING_RE.finditer(markdown))

    if not headings:
        # No headings — emit the entire content as one block.
        blocks.append(TextBlock(page=1, section="content", text=markdown.strip()))
        return blocks

    # Content before the first heading.
    pre_heading_text = markdown[: headings[0].start()].strip()
    if pre_heading_text:
        blocks.append(TextBlock(page=1, section="preamble", text=pre_heading_text))

    # Process each heading section.
    for i, match in enumerate(headings):
        section_name = match.group(2).strip()
        start = match.start()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(markdown)
        section_text = markdown[start:end].strip()

        if section_text:
            # Estimate page number from section index (1-based).
            page = i + 1
            blocks.append(TextBlock(page=page, section=section_name, text=section_text))

    return blocks


@node("docling_convert")
def docling_convert_node(state: AgentState) -> dict[str, Any]:
    """Convert document via Docling, extract images, produce text blocks.

    Steps:
    1. Convert document to DoclingDocument via DoclingConverterService
    2. Export to Markdown (with image placeholders)
    3. Extract images → upload to S3 → record ImageItems
    4. Insert contextual image references into Markdown
    5. Handle speaker notes fallback for PPTX
    6. Split Markdown into TextBlocks by page/section
    7. Return extracted_text_blocks + extracted_image_items
    """
    settings = get_settings()
    meta = state["doc_metadata"]
    local_path = state["local_path"]

    log.info(
        "docling_convert_start",
        doc_id=meta.doc_id,
        doc_type=meta.doc_type,
        file_path=local_path,
    )

    # 1. Convert document.
    service = DoclingConverterService(settings)
    result = service.convert(local_path)

    # 2. Export to Markdown.
    markdown = service.extract_markdown(result)

    # 3. Extract images and upload to S3.
    raw_images = service.extract_images(result)
    image_items: list[ImageItem] = []
    image_positions: list[int] = []

    for img_data in raw_images:
        if not img_data.image_bytes:
            log.warning(
                "empty_image_skipped",
                doc_id=meta.doc_id,
                page=img_data.page,
                image_index=img_data.image_index,
            )
            continue

        s3_url = _s3.upload_image(
            meta.knowledge_base_type,
            meta.doc_id,
            img_data.page,
            img_data.image_index,
            img_data.image_bytes,
            ext=img_data.format,
        )

        image_items.append(
            ImageItem(
                page=img_data.page,
                image_index=img_data.image_index,
                s3_url=s3_url,
            )
        )
        image_positions.append(img_data.position_in_markdown)

    # 4. Insert image references into Markdown at original positions.
    if image_items:
        markdown = _insert_image_references(markdown, image_items, image_positions)

    # 5. Handle speaker notes fallback for PPTX.
    if meta.doc_type == "PPT":
        has_speaker_notes = bool(_SPEAKER_NOTES_RE.search(markdown))
        if not has_speaker_notes:
            log.info(
                "speaker_notes_fallback",
                doc_id=meta.doc_id,
                reason="docling_did_not_extract_notes",
            )
            notes = _extract_speaker_notes_fallback(local_path)
            if notes:
                markdown = _append_speaker_notes(markdown, notes)
                log.info(
                    "speaker_notes_appended",
                    doc_id=meta.doc_id,
                    slide_count=len(notes),
                )

    # 6. Split Markdown into TextBlocks.
    text_blocks = _split_markdown_to_text_blocks(markdown)

    log.info(
        "docling_convert_complete",
        doc_id=meta.doc_id,
        text_blocks=len(text_blocks),
        images=len(image_items),
    )

    return {
        "extracted_text_blocks": text_blocks,
        "extracted_image_items": image_items,
    }
