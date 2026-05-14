"""Image processing nodes: VLM summarization + assembly into text blocks."""

from __future__ import annotations

import re
from typing import Any

from app.graph.nodes.status import node
from app.graph.state import AgentState, AssembledText, ImageItem
from app.services.llm_service import VLM_FALLBACK_SUMMARY, summarize_image
from app.services.s3_service import S3Client
from app.utils.logger import get_logger

log = get_logger(__name__)

_s3 = S3Client()

_PLACEHOLDER_RE = re.compile(
    r'<IMAGE_REFERENCE\s+src="(?P<src>[^"]+)"\s+page="(?P<page>\d+)"\s+index="(?P<idx>\d+)">'
    r"\s*</IMAGE_REFERENCE>",
    flags=re.DOTALL,
)


@node("summarize_images")
def summarize_images_node(state: AgentState) -> dict[str, Any]:
    """Generate a VLM summary for each extracted image."""
    items = list(state.get("extracted_image_items") or [])
    if not items:
        return {}

    updated: list[ImageItem] = []
    for item in items:
        try:
            blob = _s3.download_bytes(item.s3_url)
            summary = summarize_image(blob)
        except Exception as exc:  # noqa: BLE001 — soft-fail per image
            log.warning(
                "summarize_image_failed",
                s3_url=item.s3_url,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            summary = VLM_FALLBACK_SUMMARY
        updated.append(item.model_copy(update={"summary": summary}))

    return {"extracted_image_items": updated}


@node("assemble")
def assemble_node(state: AgentState) -> dict[str, Any]:
    """Splice image summaries into their `<IMAGE_REFERENCE>` placeholders."""
    image_items = state.get("extracted_image_items") or []
    summaries = {(i.page, i.image_index): (i.summary or VLM_FALLBACK_SUMMARY) for i in image_items}

    def _replace(match: re.Match[str]) -> str:
        src = match.group("src")
        page = int(match.group("page"))
        idx = int(match.group("idx"))
        summary = summaries.get((page, idx), VLM_FALLBACK_SUMMARY)
        return (
            f'<IMAGE_REFERENCE src="{src}" page="{page}" index="{idx}">\n'
            f"{summary}\n"
            f"</IMAGE_REFERENCE>"
        )

    blocks = state.get("extracted_text_blocks") or []
    assembled = [
        AssembledText(section=b.section, page=b.page, text=_PLACEHOLDER_RE.sub(_replace, b.text))
        for b in blocks
    ]
    return {"assembled_texts": assembled}
