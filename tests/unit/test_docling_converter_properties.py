"""Property-based tests for Docling converter image reference insertion.

Feature: docling-migration, Property 7: Image Reference Positional Integrity
**Validates: Requirements 3.2, 3.4**

For any image extracted from a document, the Image_Handler SHALL insert the
image reference (placeholder + VLM summary) into the Markdown at the position
corresponding to where the image appeared in the original document, and the
summary text SHALL be adjacent to the image placeholder within the same
protected block.
"""

from __future__ import annotations

import re

from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from app.graph.nodes.docling_converter import _insert_image_references
from app.graph.state import ImageItem


# ─────────────────────────────────────────────────────────────────────────────
#  Strategies
# ─────────────────────────────────────────────────────────────────────────────

# Strategy for generating paragraph text (no special markers)
paragraph_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "S", "Zs"),
        blacklist_characters="<>!\n\r#",
    ),
    min_size=10,
    max_size=100,
).filter(lambda t: t.strip())

# Strategy for generating multi-line markdown content with newlines
# (the function inserts at newline boundaries, so we need content with newlines)
markdown_content = st.builds(
    lambda parts: "\n".join(parts),
    st.lists(paragraph_text, min_size=4, max_size=10),
)

# Strategy for generating image page numbers
page_number = st.integers(min_value=1, max_value=20)


@st.composite
def markdown_and_images(draw):
    """Generate markdown content along with image items at distinct positions.

    Ensures positions are unique and well-separated so that insertions
    don't interfere with each other.
    """
    md = draw(markdown_content)
    assume(len(md) > 50)

    # Find all newline positions in the markdown (insertion points)
    newline_positions = [i for i, c in enumerate(md) if c == "\n"]
    assume(len(newline_positions) >= 2)

    num_images = draw(st.integers(min_value=1, max_value=min(4, len(newline_positions))))

    # Draw distinct positions from the available newline positions
    chosen_indices = draw(
        st.lists(
            st.sampled_from(range(len(newline_positions))),
            min_size=num_images,
            max_size=num_images,
            unique=True,
        )
    )
    positions = sorted([newline_positions[i] for i in chosen_indices])

    items: list[ImageItem] = []
    for i, pos in enumerate(positions):
        page = draw(page_number)
        idx = i + 1
        s3_url = f"s3://bucket/raw/Maintenance/images/doc123/{page}_{idx}.png"

        item = ImageItem(
            page=page,
            image_index=idx,
            s3_url=s3_url,
        )
        items.append(item)

    return md, items, positions


@st.composite
def markdown_and_single_image(draw):
    """Generate markdown with a single image for simpler property checks."""
    md = draw(markdown_content)
    assume(len(md) > 30)

    newline_positions = [i for i, c in enumerate(md) if c == "\n"]
    assume(len(newline_positions) >= 1)

    pos_idx = draw(st.sampled_from(range(len(newline_positions))))
    position = newline_positions[pos_idx]

    page = draw(page_number)
    s3_url = f"s3://bucket/raw/Maintenance/images/doc123/{page}_1.png"

    item = ImageItem(
        page=page,
        image_index=1,
        s3_url=s3_url,
    )

    return md, [item], [position]


# ─────────────────────────────────────────────────────────────────────────────
#  Property Tests
# ─────────────────────────────────────────────────────────────────────────────

# Regex patterns for image reference blocks
IMAGE_REF_START = re.compile(r"<!-- IMAGE_REF:.*?-->")
IMAGE_REF_END = re.compile(r"<!-- /IMAGE_REF -->")
IMAGE_PLACEHOLDER = re.compile(r"!\[Image\]\(.*?\)")
SUMMARY_LINE = re.compile(r"\*\*Summary:\*\*")


# Feature: docling-migration, Property 7: Image Reference Positional Integrity
# **Validates: Requirements 3.2, 3.4**
class TestImageReferencePositionalIntegrity:
    """For any image extracted from a document, the Image_Handler SHALL insert
    the image reference (placeholder + VLM summary) into the Markdown at the
    position corresponding to where the image appeared in the original document,
    and the summary text SHALL be adjacent to the image placeholder within the
    same protected block.
    """

    @given(data=markdown_and_images())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_image_reference_contains_placeholder_and_summary(
        self, data: tuple[str, list[ImageItem], list[int]]
    ) -> None:
        """Each inserted image reference block contains both the image
        placeholder and the summary text adjacent to each other."""
        md, items, positions = data
        assume(len(items) > 0)

        result = _insert_image_references(md, items, positions)

        # For each image item, verify its reference block is complete
        for item in items:
            # The block should contain the IMAGE_REF start marker with the s3_url
            assert f"<!-- IMAGE_REF: {item.s3_url}" in result, (
                f"IMAGE_REF start marker not found for {item.s3_url}"
            )
            # The block should contain the image placeholder
            assert f"![Image]({item.s3_url})" in result, (
                f"Image placeholder not found for {item.s3_url}"
            )
            # The block should contain the summary marker
            assert "**Summary:**" in result, (
                "Summary marker not found in result"
            )
            # The block should contain the IMAGE_REF end marker
            assert "<!-- /IMAGE_REF -->" in result, (
                "IMAGE_REF end marker not found in result"
            )

    @given(data=markdown_and_single_image())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_summary_adjacent_to_placeholder_within_block(
        self, data: tuple[str, list[ImageItem], list[int]]
    ) -> None:
        """The summary text SHALL be adjacent to the image placeholder
        within the same protected block."""
        md, items, positions = data
        assume(len(items) > 0)

        result = _insert_image_references(md, items, positions)

        # Find all complete IMAGE_REF blocks
        block_pattern = re.compile(
            r"(<!-- IMAGE_REF:.*?-->.*?<!-- /IMAGE_REF -->)", re.DOTALL
        )
        blocks = block_pattern.findall(result)

        assert len(blocks) == len(items), (
            f"Expected {len(items)} image ref blocks, found {len(blocks)}"
        )

        for block in blocks:
            # Within each block, verify the placeholder and summary are present
            assert IMAGE_PLACEHOLDER.search(block), (
                f"Image placeholder missing in block: {block[:100]!r}"
            )
            assert SUMMARY_LINE.search(block), (
                f"Summary line missing in block: {block[:100]!r}"
            )

            # Verify adjacency: the placeholder line should be immediately
            # followed by the summary line (with no other content between them)
            lines = block.strip().split("\n")
            placeholder_idx = None
            summary_idx = None
            for i, line in enumerate(lines):
                if IMAGE_PLACEHOLDER.search(line):
                    placeholder_idx = i
                if SUMMARY_LINE.search(line):
                    summary_idx = i

            assert placeholder_idx is not None, "Placeholder line not found"
            assert summary_idx is not None, "Summary line not found"
            assert summary_idx == placeholder_idx + 1, (
                f"Summary not adjacent to placeholder: "
                f"placeholder at line {placeholder_idx}, summary at line {summary_idx}"
            )

    @given(data=markdown_and_images())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_image_references_in_correct_relative_order(
        self, data: tuple[str, list[ImageItem], list[int]]
    ) -> None:
        """Image references appear in the output in the order corresponding
        to their original positions in the document (ascending position)."""
        md, items, positions = data
        assume(len(items) > 1)

        result = _insert_image_references(md, items, positions)

        # Sort items by their original position to get expected order
        sorted_pairs = sorted(zip(positions, items), key=lambda x: x[0])
        expected_order_urls = [item.s3_url for _, item in sorted_pairs]

        # Find the order of s3_urls as they appear in the result
        found_positions: list[tuple[int, str]] = []
        for item in items:
            marker = f"<!-- IMAGE_REF: {item.s3_url}"
            pos = result.find(marker)
            if pos >= 0:
                found_positions.append((pos, item.s3_url))

        found_positions.sort(key=lambda x: x[0])
        actual_order_urls = [url for _, url in found_positions]

        assert actual_order_urls == expected_order_urls, (
            f"Image references not in correct positional order.\n"
            f"Expected: {expected_order_urls}\n"
            f"Actual:   {actual_order_urls}"
        )

    @given(data=markdown_and_single_image())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_original_content_preserved(
        self, data: tuple[str, list[ImageItem], list[int]]
    ) -> None:
        """The original markdown content is preserved after image reference insertion."""
        md, items, positions = data
        assume(len(items) > 0)

        result = _insert_image_references(md, items, positions)

        # Remove all inserted image reference blocks from the result
        cleaned = re.sub(
            r"<!-- IMAGE_REF:.*?<!-- /IMAGE_REF -->\n?",
            "",
            result,
            flags=re.DOTALL,
        )

        # The original markdown content should be fully contained in the cleaned result
        # (the function only inserts content, never removes)
        # Check that each original line is present in the cleaned output
        original_lines = [line for line in md.split("\n") if line.strip()]
        for line in original_lines:
            assert line in cleaned, (
                f"Original line lost after insertion: {line!r}"
            )
