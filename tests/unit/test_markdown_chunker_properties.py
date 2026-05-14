"""Property-based tests for MarkdownChunker using Hypothesis.

Tests validate universal correctness properties of the MarkdownChunker
across randomly generated Markdown inputs.
"""

from __future__ import annotations

import re

from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from app.graph.nodes.markdown_chunker import MarkdownChunker


# ─────────────────────────────────────────────────────────────────────────────
#  Strategies
# ─────────────────────────────────────────────────────────────────────────────

# Strategy for generating heading levels (H1, H2, H3)
heading_level = st.sampled_from(["#", "##", "###"])

# Strategy for generating non-empty text lines (no heading markers or HTML comments)
text_line = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "S", "Zs"),
        blacklist_characters="#<>!\n\r",
    ),
    min_size=5,
    max_size=80,
).filter(lambda t: t.strip() and not t.strip().startswith("#"))

# Strategy for generating paragraph text (multiple sentences)
paragraph = st.builds(
    lambda parts: ". ".join(parts) + ".",
    st.lists(text_line, min_size=1, max_size=5),
)

# Strategy for generating a heading line
heading_line = st.builds(
    lambda level, text: f"{level} {text}",
    heading_level,
    st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "Zs"), blacklist_characters="\n\r#<>"),
        min_size=3,
        max_size=30,
    ).filter(lambda t: t.strip()),
)

# Strategy for generating a section (heading + content)
section = st.builds(
    lambda h, p: f"{h}\n{p}",
    heading_line,
    paragraph,
)

# Strategy for generating Markdown with multiple headed sections
markdown_with_headings = st.builds(
    lambda sections: "\n\n".join(sections),
    st.lists(section, min_size=2, max_size=6),
)

# Strategy for generating chunk_size values
chunk_size_strategy = st.integers(min_value=100, max_value=2000)

# Strategy for generating chunk_overlap values (relative to chunk_size)
chunk_overlap_strategy = st.integers(min_value=0, max_value=50)

# Strategy for generating Unicode text with various scripts
unicode_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "S", "Zs"),
        blacklist_characters="<>!\n\r#",
    ),
    min_size=10,
    max_size=100,
).filter(lambda t: t.strip())

# Strategy for generating image reference blocks
image_ref_block = st.builds(
    lambda page, idx, summary: (
        f"<!-- IMAGE_REF: s3://bucket/key page={page} index={idx} -->\n"
        f"![Image](s3://bucket/path/{page}_{idx}.png)\n"
        f"**Summary:** {summary}\n"
        f"<!-- /IMAGE_REF -->"
    ),
    st.integers(min_value=1, max_value=10),
    st.integers(min_value=1, max_value=5),
    st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "Zs"), blacklist_characters="\n\r<>"),
        min_size=10,
        max_size=60,
    ).filter(lambda t: t.strip()),
)

# Strategy for generating table blocks
table_block = st.builds(
    lambda cols, rows: (
        "<!-- TABLE_BLOCK -->\n"
        + "| " + " | ".join(f"Col{i}" for i in range(1, cols + 1)) + " |\n"
        + "|" + "|".join("---" for _ in range(cols)) + "|\n"
        + "\n".join(
            "| " + " | ".join(f"r{r}c{c}" for c in range(1, cols + 1)) + " |"
            for r in range(1, rows + 1)
        )
        + "\n<!-- /TABLE_BLOCK -->"
    ),
    st.integers(min_value=2, max_value=5),
    st.integers(min_value=1, max_value=8),
)

# Strategy for generating speaker notes blocks
speaker_notes_block = st.builds(
    lambda slide, text: (
        f"<!-- SPEAKER_NOTES slide={slide} -->\n"
        f"{text}\n"
        f"<!-- /SPEAKER_NOTES -->"
    ),
    st.integers(min_value=1, max_value=20),
    st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P", "Zs"), blacklist_characters="\n\r<>"),
        min_size=10,
        max_size=100,
    ).filter(lambda t: t.strip()),
)

# Strategy for any protected block
protected_block = st.one_of(image_ref_block, table_block, speaker_notes_block)

# Strategy for mixed content (text interleaved with protected blocks)
mixed_content_element = st.one_of(
    paragraph.map(lambda p: ("text", p)),
    table_block.map(lambda t: ("table", t)),
    image_ref_block.map(lambda i: ("image", i)),
)


# ─────────────────────────────────────────────────────────────────────────────
#  Property Tests
# ─────────────────────────────────────────────────────────────────────────────


# Feature: docling-migration, Property 2: Heading Boundary Splitting
# **Validates: Requirements 2.1**
class TestHeadingBoundarySplitting:
    """For any Markdown content containing heading markers (H1, H2, H3),
    the chunker SHALL produce chunks where each chunk begins with at most
    one heading, and no heading marker appears in the middle of a chunk
    (except when the entire section fits within chunk_size).
    """

    @given(
        md=markdown_with_headings,
        chunk_size=st.integers(min_value=200, max_value=2000),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_no_heading_in_middle_of_chunk(self, md: str, chunk_size: int) -> None:
        """Each chunk has at most one heading, and it appears at the start."""
        chunker = MarkdownChunker(chunk_size=chunk_size, chunk_overlap=0)
        chunks = chunker.chunk(md)

        assume(len(chunks) > 0)

        heading_pattern = re.compile(r"^(#{1,3})\s+\S", re.MULTILINE)

        for chunk in chunks:
            # Find all heading occurrences in this chunk
            headings_in_chunk = list(heading_pattern.finditer(chunk))

            if not headings_in_chunk:
                continue

            # If the chunk is small enough to be a single section, it's fine
            # to have one heading at the start
            if len(headings_in_chunk) == 1:
                # The single heading should be at the start of the chunk
                assert headings_in_chunk[0].start() == 0, (
                    f"Single heading not at start of chunk: "
                    f"found at position {headings_in_chunk[0].start()} in chunk: {chunk[:100]!r}"
                )
            else:
                # Multiple headings in one chunk: this is only acceptable if
                # the entire content fits within chunk_size (i.e., the chunker
                # didn't need to split further)
                assert len(chunk) <= chunk_size, (
                    f"Multiple headings found in chunk exceeding chunk_size: "
                    f"chunk length={len(chunk)}, chunk_size={chunk_size}"
                )


# Feature: docling-migration, Property 3: Chunk Size Invariant
# **Validates: Requirements 2.3**
class TestChunkSizeInvariant:
    """For any Markdown input, all chunks produced SHALL have length ≤ chunk_size,
    with the sole exception of protected blocks (tables, image references,
    speaker notes) whose content inherently exceeds chunk_size.
    """

    @given(
        md=markdown_with_headings,
        chunk_size=st.integers(min_value=100, max_value=2000),
    )
    @settings(max_examples=100)
    def test_chunks_respect_size_limit(self, md: str, chunk_size: int) -> None:
        """All chunks are within chunk_size unless they contain a protected block."""
        chunker = MarkdownChunker(chunk_size=chunk_size, chunk_overlap=0)
        chunks = chunker.chunk(md)

        for chunk in chunks:
            if len(chunk) > chunk_size:
                # Only acceptable if chunk contains a protected block
                has_protected = (
                    "<!-- IMAGE_REF" in chunk
                    or "<!-- TABLE_BLOCK -->" in chunk
                    or "<!-- SPEAKER_NOTES" in chunk
                )
                assert has_protected, (
                    f"Chunk exceeds chunk_size ({len(chunk)} > {chunk_size}) "
                    f"without containing a protected block: {chunk[:100]!r}"
                )

    @given(
        blocks=st.lists(protected_block, min_size=1, max_size=3),
        text_parts=st.lists(paragraph, min_size=1, max_size=3),
    )
    @settings(max_examples=100)
    def test_protected_blocks_allowed_to_exceed(
        self, blocks: list[str], text_parts: list[str]
    ) -> None:
        """Protected blocks that exceed chunk_size are still emitted as single chunks."""
        # Interleave text and blocks
        parts: list[str] = []
        for i, text in enumerate(text_parts):
            parts.append(text)
            if i < len(blocks):
                parts.append(blocks[i])
        md = "\n\n".join(parts)

        # Use a small chunk_size to force protected blocks to exceed it
        chunker = MarkdownChunker(chunk_size=50, chunk_overlap=0)
        chunks = chunker.chunk(md)

        for chunk in chunks:
            if len(chunk) > 50:
                # Must contain a protected block marker
                has_protected = (
                    "<!-- IMAGE_REF" in chunk
                    or "<!-- TABLE_BLOCK -->" in chunk
                    or "<!-- SPEAKER_NOTES" in chunk
                )
                assert has_protected, (
                    f"Chunk exceeds chunk_size=50 without protected block: {chunk[:100]!r}"
                )


# Feature: docling-migration, Property 4: Overlap Correctness with Protected Block Exclusion
# **Validates: Requirements 2.4**
class TestOverlapCorrectnessWithProtectedBlockExclusion:
    """For any pair of adjacent chunks from the same section, the overlap region
    SHALL contain approximately chunk_overlap characters of shared text, AND no
    protected block content SHALL appear in the overlap region.
    """

    @given(
        md=st.builds(
            lambda parts: "\n\n".join(parts),
            st.lists(paragraph, min_size=3, max_size=6),
        ),
        chunk_size=st.integers(min_value=100, max_value=500),
        chunk_overlap=st.integers(min_value=10, max_value=50),
    )
    @settings(max_examples=100)
    def test_no_protected_block_in_overlap(
        self, md: str, chunk_size: int, chunk_overlap: int
    ) -> None:
        """Protected block markers never appear in overlap regions between chunks."""
        assume(chunk_overlap < chunk_size)

        # Add a protected block in the middle
        md_with_block = (
            md[:len(md) // 2]
            + "\n\n<!-- TABLE_BLOCK -->\n| A | B |\n|---|---|\n| 1 | 2 |\n<!-- /TABLE_BLOCK -->\n\n"
            + md[len(md) // 2:]
        )

        chunker = MarkdownChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = chunker.chunk(md_with_block)

        assume(len(chunks) > 1)

        # For each pair of adjacent chunks, check that overlap text
        # does not contain protected block markers
        for i in range(1, len(chunks)):
            chunk = chunks[i]
            prev_chunk = chunks[i - 1]

            # If current chunk is a protected block itself, skip
            if "<!-- TABLE_BLOCK -->" in chunk and "<!-- /TABLE_BLOCK -->" in chunk:
                continue

            # If previous chunk is a protected block, the current chunk
            # should not start with table content from the protected block
            if "<!-- TABLE_BLOCK -->" in prev_chunk:
                # The overlap should not contain table markers
                assert not chunk.startswith("<!-- TABLE_BLOCK"), (
                    f"Protected block content leaked into overlap: {chunk[:100]!r}"
                )
                assert not chunk.startswith("| A | B |"), (
                    f"Table content leaked into overlap region: {chunk[:100]!r}"
                )


# Feature: docling-migration, Property 5: Small Section Single Chunk
# **Validates: Requirements 2.5**
class TestSmallSectionSingleChunk:
    """For any Markdown section whose total length (including heading) is less
    than or equal to chunk_size, the chunker SHALL emit it as exactly one chunk
    without further splitting.
    """

    @given(
        heading=heading_line,
        body=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P", "Zs"),
                blacklist_characters="#<>!\n\r",
            ),
            min_size=5,
            max_size=50,
        ).filter(lambda t: t.strip()),
    )
    @settings(max_examples=100)
    def test_small_section_not_split(self, heading: str, body: str) -> None:
        """A section smaller than chunk_size is emitted as exactly one chunk."""
        section_text = f"{heading}\n{body}"
        # Use a chunk_size larger than the section
        chunk_size = len(section_text) + 100

        chunker = MarkdownChunker(chunk_size=chunk_size, chunk_overlap=0)
        chunks = chunker.chunk(section_text)

        assert len(chunks) == 1, (
            f"Expected 1 chunk for section of length {len(section_text)} "
            f"with chunk_size={chunk_size}, got {len(chunks)} chunks"
        )
        assert heading in chunks[0]
        assert body.strip() in chunks[0]

    @given(
        body=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P", "Zs"),
                blacklist_characters="#<>!\n\r",
            ),
            min_size=5,
            max_size=100,
        ).filter(lambda t: t.strip()),
    )
    @settings(max_examples=100)
    def test_small_text_without_heading_single_chunk(self, body: str) -> None:
        """Text without headings that fits in chunk_size is a single chunk."""
        chunk_size = len(body) + 100
        chunker = MarkdownChunker(chunk_size=chunk_size, chunk_overlap=0)
        chunks = chunker.chunk(body)

        assert len(chunks) == 1, (
            f"Expected 1 chunk for text of length {len(body)} "
            f"with chunk_size={chunk_size}, got {len(chunks)} chunks"
        )


# Feature: docling-migration, Property 6: Protected Blocks Never Split
# **Validates: Requirements 3.6, 4.2, 5.3**
class TestProtectedBlocksNeverSplit:
    """For any protected block (image reference with summary, Markdown table,
    or speaker notes block), the chunker SHALL emit it as a single atomic unit
    within one chunk — the block SHALL never be split across two or more chunks.
    """

    @given(block=image_ref_block)
    @settings(max_examples=100)
    def test_image_ref_never_split(self, block: str) -> None:
        """Image reference blocks are never split across chunks."""
        md = f"Some intro text.\n\n{block}\n\nSome outro text."
        # Use small chunk_size to force splitting of surrounding text
        chunker = MarkdownChunker(chunk_size=50, chunk_overlap=0)
        chunks = chunker.chunk(md)

        # Find the chunk containing the image ref start
        start_chunks = [c for c in chunks if "<!-- IMAGE_REF" in c]
        end_chunks = [c for c in chunks if "<!-- /IMAGE_REF -->" in c]

        assert len(start_chunks) >= 1, "Image ref start marker not found in any chunk"
        assert len(end_chunks) >= 1, "Image ref end marker not found in any chunk"

        # The start and end must be in the same chunk
        for chunk in start_chunks:
            assert "<!-- /IMAGE_REF -->" in chunk, (
                f"Image ref block split across chunks: start found but end missing in: {chunk[:100]!r}"
            )

    @given(block=table_block)
    @settings(max_examples=100)
    def test_table_block_never_split(self, block: str) -> None:
        """Table blocks are never split across chunks."""
        md = f"Some intro text.\n\n{block}\n\nSome outro text."
        chunker = MarkdownChunker(chunk_size=50, chunk_overlap=0)
        chunks = chunker.chunk(md)

        start_chunks = [c for c in chunks if "<!-- TABLE_BLOCK -->" in c]
        end_chunks = [c for c in chunks if "<!-- /TABLE_BLOCK -->" in c]

        assert len(start_chunks) >= 1, "Table block start marker not found"
        assert len(end_chunks) >= 1, "Table block end marker not found"

        for chunk in start_chunks:
            assert "<!-- /TABLE_BLOCK -->" in chunk, (
                f"Table block split across chunks: start found but end missing in: {chunk[:100]!r}"
            )

    @given(block=speaker_notes_block)
    @settings(max_examples=100)
    def test_speaker_notes_never_split(self, block: str) -> None:
        """Speaker notes blocks are never split across chunks."""
        md = f"Some intro text.\n\n{block}\n\nSome outro text."
        chunker = MarkdownChunker(chunk_size=50, chunk_overlap=0)
        chunks = chunker.chunk(md)

        start_chunks = [c for c in chunks if "<!-- SPEAKER_NOTES" in c]
        end_chunks = [c for c in chunks if "<!-- /SPEAKER_NOTES -->" in c]

        assert len(start_chunks) >= 1, "Speaker notes start marker not found"
        assert len(end_chunks) >= 1, "Speaker notes end marker not found"

        for chunk in start_chunks:
            assert "<!-- /SPEAKER_NOTES -->" in chunk, (
                f"Speaker notes split across chunks: start found but end missing in: {chunk[:100]!r}"
            )


# Feature: docling-migration, Property 10: Content Ordering Preservation
# **Validates: Requirements 6.4**
class TestContentOrderingPreservation:
    """For any document containing mixed content (text, tables, images interleaved),
    the chunker SHALL produce chunks where the relative ordering of content elements
    within each chunk matches their original ordering in the source Markdown.
    """

    @given(
        elements=st.lists(mixed_content_element, min_size=3, max_size=8),
    )
    @settings(max_examples=100)
    def test_content_ordering_preserved(
        self, elements: list[tuple[str, str]]
    ) -> None:
        """The relative ordering of content elements is preserved across chunks."""
        # Tag each element with a unique marker to track ordering
        tagged_elements: list[tuple[str, str, str]] = []
        for i, (etype, content) in enumerate(elements):
            marker = f"MARKER_{i:04d}"
            if etype == "text":
                tagged_content = f"{marker} {content}"
            elif etype == "table":
                # Insert marker before the table block
                tagged_content = f"{marker}\n\n{content}"
            elif etype == "image":
                # Insert marker before the image block
                tagged_content = f"{marker}\n\n{content}"
            else:
                tagged_content = f"{marker} {content}"
            tagged_elements.append((etype, tagged_content, marker))

        # Build the markdown from tagged elements
        md = "\n\n".join(content for _, content, _ in tagged_elements)

        chunker = MarkdownChunker(chunk_size=800, chunk_overlap=0)
        chunks = chunker.chunk(md)

        assume(len(chunks) > 0)

        # Concatenate all chunks to get the full output
        all_output = "\n".join(chunks)

        # Collect positions of markers in the output
        positions: list[tuple[int, int]] = []  # (position, element_index)
        for i, (_, _, marker) in enumerate(tagged_elements):
            pos = all_output.find(marker)
            if pos != -1:
                positions.append((pos, i))

        # Verify that found markers appear in order
        for j in range(len(positions) - 1):
            pos_a, idx_a = positions[j]
            pos_b, idx_b = positions[j + 1]
            assert idx_a < idx_b, (
                f"Content ordering violated: element {idx_a} (pos={pos_a}) "
                f"appears after element {idx_b} (pos={pos_b})"
            )


# Feature: docling-migration, Property 11: Unicode Preservation Through Chunking
# **Validates: Requirements 6.7**
class TestUnicodePreservationThroughChunking:
    """For any Markdown content containing Unicode characters, special symbols,
    or non-Latin scripts, the chunking process SHALL preserve all characters
    without corruption — the concatenation of all chunk contents (minus overlap)
    SHALL contain every character from the original input.
    """

    @given(
        parts=st.lists(
            st.text(
                alphabet=st.characters(
                    whitelist_categories=("L", "N", "P", "S"),
                    blacklist_characters="<>!\n\r#",
                ),
                min_size=10,
                max_size=80,
            ).filter(lambda t: t.strip()),
            min_size=2,
            max_size=5,
        ),
    )
    @settings(max_examples=100)
    def test_unicode_characters_preserved(self, parts: list[str]) -> None:
        """All Unicode characters from the input appear in the chunked output."""
        md = "\n\n".join(parts)

        # Use no overlap to simplify verification
        chunker = MarkdownChunker(chunk_size=200, chunk_overlap=0)
        chunks = chunker.chunk(md)

        assume(len(chunks) > 0)

        # Concatenate all chunks
        all_output = " ".join(chunks)

        # Every non-whitespace character from the original should appear in output
        original_chars = set(md) - {"\n", "\r", " ", "\t", "\xa0"}
        output_chars = set(all_output)

        missing = original_chars - output_chars
        assert not missing, (
            f"Unicode characters lost during chunking: {missing!r}"
        )

    @given(
        text=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P", "S", "Zs"),
                blacklist_characters="<>!\n\r#",
            ),
            min_size=20,
            max_size=300,
        ).filter(lambda t: t.strip() and " " in t.strip()),
        chunk_size=st.integers(min_value=50, max_value=200),
    )
    @settings(max_examples=100)
    def test_no_character_corruption(self, text: str, chunk_size: int) -> None:
        """Chunking does not corrupt any characters (no mojibake, no replacements)."""
        chunker = MarkdownChunker(chunk_size=chunk_size, chunk_overlap=0)
        chunks = chunker.chunk(text)

        assume(len(chunks) > 0)

        # Rejoin chunks and verify all original characters are present
        all_output = " ".join(chunks)

        # Every unique character from the original should appear in the output
        # (characters are preserved, not corrupted into different characters)
        original_chars = set(text) - {"\n", "\r"}
        output_chars = set(all_output)

        missing = original_chars - output_chars
        # Filter out whitespace-only differences (spaces may be trimmed)
        missing = {c for c in missing if c.strip()}
        assert not missing, (
            f"Characters corrupted or lost during chunking: {missing!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Property 8 & 9: Table Handling Properties
# ─────────────────────────────────────────────────────────────────────────────


# Strategy for generating table column counts
table_col_count = st.integers(min_value=2, max_value=6)

# Strategy for generating table row counts (enough to force splitting)
table_row_count = st.integers(min_value=3, max_value=20)

# Strategy for generating cell content
cell_content = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), blacklist_characters="|<>!\n\r#"),
    min_size=2,
    max_size=10,
).filter(lambda t: t.strip())


@st.composite
def markdown_table(draw, min_cols=2, max_cols=6, min_rows=3, max_rows=20):
    """Generate a Markdown table with given column/row constraints."""
    num_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))
    num_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))

    # Generate header cells
    headers = [draw(cell_content) for _ in range(num_cols)]
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "|" + "|".join("---" for _ in range(num_cols)) + "|"

    # Generate data rows
    data_rows = []
    for _ in range(num_rows):
        cells = [draw(cell_content) for _ in range(num_cols)]
        data_rows.append("| " + " | ".join(cells) + " |")

    table_str = "\n".join([header_line, separator_line] + data_rows)
    return table_str, headers, separator_line, data_rows


@st.composite
def wrapped_markdown_table(draw, min_cols=2, max_cols=6, min_rows=3, max_rows=20):
    """Generate a TABLE_BLOCK-wrapped Markdown table."""
    table_str, headers, separator_line, data_rows = draw(
        markdown_table(min_cols=min_cols, max_cols=max_cols, min_rows=min_rows, max_rows=max_rows)
    )
    wrapped = f"<!-- TABLE_BLOCK -->\n{table_str}\n<!-- /TABLE_BLOCK -->"
    return wrapped, headers, separator_line, data_rows


# Feature: docling-migration, Property 8: Large Table Splitting Preserves Headers
# **Validates: Requirements 4.3, 4.5, 6.2**
class TestLargeTableSplittingPreservesHeaders:
    """For any Markdown table exceeding max_table_size characters (or any XLSX sheet
    exceeding the configured row limit), the Table_Handler SHALL split it into
    sub-tables where each sub-table preserves the original column headers as its
    first row, and each sub-table (except possibly single-row edge cases) is within
    max_table_size.
    """

    @given(
        data=st.data(),
        num_cols=st.integers(min_value=2, max_value=5),
        num_rows=st.integers(min_value=4, max_value=15),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_split_tables_preserve_headers(
        self, data: st.DataObject, num_cols: int, num_rows: int
    ) -> None:
        """Each sub-table after splitting preserves the original header row and separator."""
        # Generate table cells
        headers = [data.draw(cell_content) for _ in range(num_cols)]
        header_line = "| " + " | ".join(headers) + " |"
        separator_line = "|" + "|".join("---" for _ in range(num_cols)) + "|"

        rows = []
        for r in range(num_rows):
            cells = [data.draw(cell_content) for _ in range(num_cols)]
            rows.append("| " + " | ".join(cells) + " |")

        table_str = "\n".join([header_line, separator_line] + rows)
        wrapped = f"<!-- TABLE_BLOCK -->\n{table_str}\n<!-- /TABLE_BLOCK -->"

        # Use a small max_table_size to force splitting
        max_table_size = len(header_line) + len(separator_line) + 80
        assume(len(wrapped) > max_table_size)

        chunker = MarkdownChunker(
            chunk_size=2000, chunk_overlap=0, max_table_size=max_table_size
        )

        # Call _handle_large_tables directly
        result = chunker._handle_large_tables(wrapped)

        # Parse out all TABLE_BLOCK regions from the result
        table_block_pattern = re.compile(
            r"<!-- TABLE_BLOCK -->\n(.*?)\n<!-- /TABLE_BLOCK -->", re.DOTALL
        )
        sub_table_matches = table_block_pattern.findall(result)

        assert len(sub_table_matches) >= 1, "No sub-tables found after splitting"

        # Verify each sub-table has the original header and separator
        for i, sub_table_content in enumerate(sub_table_matches):
            lines = [l for l in sub_table_content.strip().split("\n") if l.strip()]
            assert len(lines) >= 2, (
                f"Sub-table {i} has fewer than 2 lines (no header+separator): {lines}"
            )
            # First line should be the header
            assert lines[0] == header_line, (
                f"Sub-table {i} header mismatch.\n"
                f"Expected: {header_line!r}\n"
                f"Got:      {lines[0]!r}"
            )
            # Second line should be the separator
            assert lines[1] == separator_line, (
                f"Sub-table {i} separator mismatch.\n"
                f"Expected: {separator_line!r}\n"
                f"Got:      {lines[1]!r}"
            )

    @given(
        data=st.data(),
        num_cols=st.integers(min_value=2, max_value=5),
        num_rows=st.integers(min_value=4, max_value=15),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_split_tables_preserve_all_data_rows(
        self, data: st.DataObject, num_cols: int, num_rows: int
    ) -> None:
        """All original data rows are present across the sub-tables after splitting."""
        headers = [data.draw(cell_content) for _ in range(num_cols)]
        header_line = "| " + " | ".join(headers) + " |"
        separator_line = "|" + "|".join("---" for _ in range(num_cols)) + "|"

        rows = []
        for r in range(num_rows):
            cells = [data.draw(cell_content) for _ in range(num_cols)]
            rows.append("| " + " | ".join(cells) + " |")

        table_str = "\n".join([header_line, separator_line] + rows)
        wrapped = f"<!-- TABLE_BLOCK -->\n{table_str}\n<!-- /TABLE_BLOCK -->"

        # Use a small max_table_size to force splitting
        max_table_size = len(header_line) + len(separator_line) + 80
        assume(len(wrapped) > max_table_size)

        chunker = MarkdownChunker(
            chunk_size=2000, chunk_overlap=0, max_table_size=max_table_size
        )

        result = chunker._handle_large_tables(wrapped)

        # Collect all data rows from all sub-tables
        table_block_pattern = re.compile(
            r"<!-- TABLE_BLOCK -->\n(.*?)\n<!-- /TABLE_BLOCK -->", re.DOTALL
        )
        sub_table_matches = table_block_pattern.findall(result)

        all_data_rows: list[str] = []
        for sub_table_content in sub_table_matches:
            lines = [l for l in sub_table_content.strip().split("\n") if l.strip()]
            # Skip header (line 0) and separator (line 1)
            all_data_rows.extend(lines[2:])

        # Verify all original rows are present
        assert sorted(all_data_rows) == sorted(rows), (
            f"Data rows mismatch after splitting.\n"
            f"Original rows ({len(rows)}): {rows[:3]}...\n"
            f"Found rows ({len(all_data_rows)}): {all_data_rows[:3]}..."
        )

    @given(
        data=st.data(),
        num_cols=st.integers(min_value=2, max_value=4),
        num_rows=st.integers(min_value=5, max_value=20),
        rows_per_block=st.integers(min_value=2, max_value=5),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_xlsx_split_preserves_headers(
        self, data: st.DataObject, num_cols: int, num_rows: int, rows_per_block: int
    ) -> None:
        """XLSX row-based splitting preserves headers in each sub-table."""
        assume(num_rows > rows_per_block)

        headers = [data.draw(cell_content) for _ in range(num_cols)]
        header_line = "| " + " | ".join(headers) + " |"
        separator_line = "|" + "|".join("---" for _ in range(num_cols)) + "|"

        rows = []
        for r in range(num_rows):
            cells = [data.draw(cell_content) for _ in range(num_cols)]
            rows.append("| " + " | ".join(cells) + " |")

        table_str = "\n".join([header_line, separator_line] + rows)
        wrapped = f"<!-- TABLE_BLOCK -->\n{table_str}\n<!-- /TABLE_BLOCK -->"

        chunker = MarkdownChunker(chunk_size=5000, chunk_overlap=0, max_table_size=5000)
        result = chunker.split_xlsx_tables(wrapped, rows_per_block=rows_per_block)

        # Parse sub-tables
        table_block_pattern = re.compile(
            r"<!-- TABLE_BLOCK -->\n(.*?)\n<!-- /TABLE_BLOCK -->", re.DOTALL
        )
        sub_table_matches = table_block_pattern.findall(result)

        assert len(sub_table_matches) >= 2, (
            f"Expected at least 2 sub-tables for {num_rows} rows with "
            f"rows_per_block={rows_per_block}, got {len(sub_table_matches)}"
        )

        # Each sub-table must have the original header and separator
        all_data_rows: list[str] = []
        for i, sub_table_content in enumerate(sub_table_matches):
            lines = [l for l in sub_table_content.strip().split("\n") if l.strip()]
            assert lines[0] == header_line, (
                f"XLSX sub-table {i} header mismatch: {lines[0]!r} != {header_line!r}"
            )
            assert lines[1] == separator_line, (
                f"XLSX sub-table {i} separator mismatch: {lines[1]!r} != {separator_line!r}"
            )
            # Each sub-table should have at most rows_per_block data rows
            data_in_sub = lines[2:]
            assert len(data_in_sub) <= rows_per_block, (
                f"XLSX sub-table {i} has {len(data_in_sub)} rows, "
                f"exceeds rows_per_block={rows_per_block}"
            )
            all_data_rows.extend(data_in_sub)

        # All original data rows must be present
        assert sorted(all_data_rows) == sorted(rows), (
            f"XLSX split lost data rows: expected {len(rows)}, got {len(all_data_rows)}"
        )


# Feature: docling-migration, Property 9: Table Context Adjacency
# **Validates: Requirements 4.4**
class TestTableContextAdjacency:
    """For any table with surrounding context (caption or preceding paragraph) where
    the combined size of context + table ≤ chunk_size, the Markdown_Chunker SHALL
    place both the context and the table within the same chunk.
    """

    @given(
        context_text=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P", "Zs"),
                blacklist_characters="#<>!\n\r|",
            ),
            min_size=10,
            max_size=60,
        ).filter(lambda t: t.strip()),
        num_cols=st.integers(min_value=2, max_value=3),
        num_rows=st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_context_and_table_in_same_chunk(
        self, context_text: str, num_cols: int, num_rows: int
    ) -> None:
        """When context + table fits within chunk_size, they share the same chunk."""
        # Build a small table
        header_line = "| " + " | ".join(f"H{i}" for i in range(num_cols)) + " |"
        separator_line = "|" + "|".join("---" for _ in range(num_cols)) + "|"
        data_rows = []
        for r in range(num_rows):
            data_rows.append("| " + " | ".join(f"d{r}{c}" for c in range(num_cols)) + " |")

        table_str = "\n".join([header_line, separator_line] + data_rows)
        table_block = f"<!-- TABLE_BLOCK -->\n{table_str}\n<!-- /TABLE_BLOCK -->"

        # Build markdown: context paragraph followed by table
        md = f"{context_text.strip()}\n\n{table_block}"

        # Calculate combined size and set chunk_size to accommodate both
        combined_size = len(context_text.strip()) + 1 + len(table_block)  # +1 for newline separator
        chunk_size = combined_size + 100  # ensure it fits

        assume(combined_size <= chunk_size)

        chunker = MarkdownChunker(
            chunk_size=chunk_size, chunk_overlap=0, max_table_size=5000
        )
        chunks = chunker.chunk(md)

        assume(len(chunks) > 0)

        # Find the chunk containing the table
        table_chunks = [c for c in chunks if "<!-- TABLE_BLOCK -->" in c]
        assert len(table_chunks) >= 1, "Table not found in any chunk"

        # The context text should be in the same chunk as the table
        for table_chunk in table_chunks:
            assert context_text.strip() in table_chunk, (
                f"Context text not adjacent to table in same chunk.\n"
                f"Context: {context_text.strip()!r}\n"
                f"Table chunk: {table_chunk[:200]!r}\n"
                f"All chunks: {[c[:80] for c in chunks]}"
            )

    @given(
        context_text=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P", "Zs"),
                blacklist_characters="#<>!\n\r|",
            ),
            min_size=5,
            max_size=40,
        ).filter(lambda t: t.strip()),
        num_cols=st.integers(min_value=2, max_value=3),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_caption_before_table_stays_adjacent(
        self, context_text: str, num_cols: int
    ) -> None:
        """A caption/paragraph immediately before a table stays in the same chunk."""
        # Build a minimal table (1 data row) to keep combined size small
        header_line = "| " + " | ".join(f"Col{i}" for i in range(num_cols)) + " |"
        separator_line = "|" + "|".join("---" for _ in range(num_cols)) + "|"
        data_row = "| " + " | ".join(f"val{c}" for c in range(num_cols)) + " |"

        table_str = "\n".join([header_line, separator_line, data_row])
        table_block = f"<!-- TABLE_BLOCK -->\n{table_str}\n<!-- /TABLE_BLOCK -->"

        # Context immediately before the table
        md = f"{context_text.strip()}\n\n{table_block}"

        # Set chunk_size large enough to hold both
        combined_size = len(context_text.strip()) + 1 + len(table_block)
        chunk_size = combined_size + 200

        chunker = MarkdownChunker(
            chunk_size=chunk_size, chunk_overlap=0, max_table_size=5000
        )
        chunks = chunker.chunk(md)

        assume(len(chunks) > 0)

        # Both context and table should be in the same chunk
        table_chunks = [c for c in chunks if "<!-- TABLE_BLOCK -->" in c]
        assert len(table_chunks) == 1, (
            f"Expected exactly 1 chunk with table, got {len(table_chunks)}"
        )
        assert context_text.strip() in table_chunks[0], (
            f"Caption not in same chunk as table.\n"
            f"Caption: {context_text.strip()!r}\n"
            f"Chunk: {table_chunks[0][:200]!r}"
        )
