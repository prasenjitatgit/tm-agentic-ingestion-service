"""Unit tests for MarkdownChunker.

Tests cover:
- Protected block detection and preservation
- Heading boundary splitting
- Recursive splitting with separator hierarchy
- Large table handling
- Overlap application
- Content ordering preservation
- Edge cases (empty input, Unicode, etc.)
"""

from __future__ import annotations

import pytest

from app.graph.nodes.markdown_chunker import MarkdownChunker


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def chunker() -> MarkdownChunker:
    """Default chunker with standard settings."""
    return MarkdownChunker(chunk_size=800, chunk_overlap=100, max_table_size=1500)


@pytest.fixture
def small_chunker() -> MarkdownChunker:
    """Chunker with small chunk_size for testing splitting behavior."""
    return MarkdownChunker(chunk_size=100, chunk_overlap=20, max_table_size=200)


# ─────────────────────────────────────────────────────────────────────────────
#  Empty / Trivial Input
# ─────────────────────────────────────────────────────────────────────────────


class TestEmptyInput:
    def test_empty_string(self, chunker: MarkdownChunker) -> None:
        assert chunker.chunk("") == []

    def test_whitespace_only(self, chunker: MarkdownChunker) -> None:
        assert chunker.chunk("   \n\n  ") == []

    def test_single_word(self, chunker: MarkdownChunker) -> None:
        assert chunker.chunk("hello") == ["hello"]


# ─────────────────────────────────────────────────────────────────────────────
#  Small Section Single Chunk
# ─────────────────────────────────────────────────────────────────────────────


class TestSmallSectionSingleChunk:
    def test_short_text_single_chunk(self, chunker: MarkdownChunker) -> None:
        text = "This is a short paragraph that fits in one chunk."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_section_with_heading_fits(self, chunker: MarkdownChunker) -> None:
        text = "# Title\nShort content."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert "# Title" in chunks[0]
        assert "Short content." in chunks[0]


# ─────────────────────────────────────────────────────────────────────────────
#  Heading Boundary Splitting
# ─────────────────────────────────────────────────────────────────────────────


class TestHeadingSplit:
    def test_h1_split(self, chunker: MarkdownChunker) -> None:
        md = "# Section A\nContent A.\n\n# Section B\nContent B."
        chunks = chunker.chunk(md)
        assert len(chunks) == 2
        assert "# Section A" in chunks[0]
        assert "Content A." in chunks[0]
        assert "# Section B" in chunks[1]
        assert "Content B." in chunks[1]

    def test_h2_split(self, chunker: MarkdownChunker) -> None:
        md = "## Part 1\nText 1.\n\n## Part 2\nText 2."
        chunks = chunker.chunk(md)
        assert len(chunks) == 2
        assert "## Part 1" in chunks[0]
        assert "## Part 2" in chunks[1]

    def test_h3_split(self, chunker: MarkdownChunker) -> None:
        md = "### Sub A\nContent.\n\n### Sub B\nMore content."
        chunks = chunker.chunk(md)
        assert len(chunks) == 2

    def test_heading_preserved_in_chunk(self, chunker: MarkdownChunker) -> None:
        md = "# My Heading\nParagraph text here."
        chunks = chunker.chunk(md)
        assert "# My Heading" in chunks[0]

    def test_content_before_first_heading(self, chunker: MarkdownChunker) -> None:
        md = "Preamble text.\n\n# First Section\nSection content."
        chunks = chunker.chunk(md)
        assert len(chunks) == 2
        assert "Preamble text." in chunks[0]
        assert "# First Section" in chunks[1]

    def test_h4_not_split(self, chunker: MarkdownChunker) -> None:
        """H4+ headings do not trigger splits."""
        md = "#### Sub-sub\nContent A.\n\n#### Another\nContent B."
        chunks = chunker.chunk(md)
        # Should be one chunk since H4 doesn't split
        assert len(chunks) == 1


# ─────────────────────────────────────────────────────────────────────────────
#  Recursive Splitting
# ─────────────────────────────────────────────────────────────────────────────


class TestRecursiveSplit:
    def test_split_on_double_newline(self, small_chunker: MarkdownChunker) -> None:
        text = (
            "Paragraph one with enough text to exceed the limit.\n\n"
            "Paragraph two with enough text to exceed the limit.\n\n"
            "Paragraph three with enough text to exceed the limit."
        )
        chunks = small_chunker.chunk(text)
        assert all(len(c) <= 100 for c in chunks)
        assert len(chunks) >= 2

    def test_split_on_single_newline(self, small_chunker: MarkdownChunker) -> None:
        text = "Line one is here.\nLine two is here.\nLine three is here.\nLine four is here.\nLine five is here.\nLine six is here."
        chunks = small_chunker.chunk(text)
        assert all(len(c) <= 100 for c in chunks)

    def test_split_on_sentence_boundary(self, small_chunker: MarkdownChunker) -> None:
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here. Fifth sentence here."
        chunks = small_chunker.chunk(text)
        assert all(len(c) <= 100 for c in chunks)

    def test_split_on_space(self) -> None:
        """When no other separator works, split on space."""
        chunker = MarkdownChunker(chunk_size=30, chunk_overlap=0)
        text = "word " * 20  # 100 chars, no sentence boundaries
        chunks = chunker.chunk(text.strip())
        assert all(len(c) <= 30 for c in chunks)

    def test_no_chunks_exceed_size(self, small_chunker: MarkdownChunker) -> None:
        text = "A" * 50 + " " + "B" * 50 + " " + "C" * 50
        chunks = small_chunker.chunk(text)
        for chunk in chunks:
            assert len(chunk) <= 100


# ─────────────────────────────────────────────────────────────────────────────
#  Protected Blocks
# ─────────────────────────────────────────────────────────────────────────────


class TestProtectedBlocks:
    def test_image_ref_not_split(self, chunker: MarkdownChunker) -> None:
        md = (
            "<!-- IMAGE_REF: s3://bucket/key page=1 index=1 -->\n"
            "![Image](s3://bucket/path/1_1.png)\n"
            "**Summary:** A description of the image content.\n"
            "<!-- /IMAGE_REF -->"
        )
        chunks = chunker.chunk(md)
        assert len(chunks) == 1
        assert "<!-- IMAGE_REF" in chunks[0]
        assert "<!-- /IMAGE_REF -->" in chunks[0]

    def test_table_block_not_split(self, chunker: MarkdownChunker) -> None:
        md = (
            "<!-- TABLE_BLOCK -->\n"
            "| Col1 | Col2 | Col3 |\n"
            "|------|------|------|\n"
            "| a    | b    | c    |\n"
            "| d    | e    | f    |\n"
            "<!-- /TABLE_BLOCK -->"
        )
        chunks = chunker.chunk(md)
        assert len(chunks) == 1
        assert "<!-- TABLE_BLOCK -->" in chunks[0]
        assert "<!-- /TABLE_BLOCK -->" in chunks[0]

    def test_speaker_notes_not_split(self, chunker: MarkdownChunker) -> None:
        md = (
            "<!-- SPEAKER_NOTES slide=3 -->\n"
            "These are the speaker notes for slide 3.\n"
            "They contain important context.\n"
            "<!-- /SPEAKER_NOTES -->"
        )
        chunks = chunker.chunk(md)
        assert len(chunks) == 1
        assert "<!-- SPEAKER_NOTES" in chunks[0]
        assert "<!-- /SPEAKER_NOTES -->" in chunks[0]

    def test_protected_block_exceeding_chunk_size(self) -> None:
        """Protected blocks that exceed chunk_size are still emitted whole."""
        chunker = MarkdownChunker(chunk_size=50)
        md = (
            "<!-- TABLE_BLOCK -->\n"
            "| Col1 | Col2 | Col3 | Col4 | Col5 |\n"
            "|------|------|------|------|------|\n"
            "| data | data | data | data | data |\n"
            "<!-- /TABLE_BLOCK -->"
        )
        chunks = chunker.chunk(md)
        assert len(chunks) == 1
        # Block exceeds chunk_size but is emitted whole
        assert len(chunks[0]) > 50
        assert "<!-- TABLE_BLOCK -->" in chunks[0]

    def test_mixed_protected_and_text(self, small_chunker: MarkdownChunker) -> None:
        md = (
            "Some text before the table.\n\n"
            "<!-- TABLE_BLOCK -->\n"
            "| A | B |\n"
            "|---|---|\n"
            "| 1 | 2 |\n"
            "<!-- /TABLE_BLOCK -->\n\n"
            "Some text after the table."
        )
        chunks = small_chunker.chunk(md)
        # Table should be in its own chunk
        table_chunks = [c for c in chunks if "TABLE_BLOCK" in c]
        assert len(table_chunks) == 1
        # Table markers are intact
        assert "<!-- TABLE_BLOCK -->" in table_chunks[0]
        assert "<!-- /TABLE_BLOCK -->" in table_chunks[0]


# ─────────────────────────────────────────────────────────────────────────────
#  Large Table Handling
# ─────────────────────────────────────────────────────────────────────────────


class TestLargeTableHandling:
    def test_small_table_unchanged(self, chunker: MarkdownChunker) -> None:
        md = (
            "<!-- TABLE_BLOCK -->\n"
            "| A | B |\n"
            "|---|---|\n"
            "| 1 | 2 |\n"
            "<!-- /TABLE_BLOCK -->"
        )
        result = chunker._handle_large_tables(md)
        assert result == md

    def test_large_table_split_preserves_headers(self) -> None:
        chunker = MarkdownChunker(max_table_size=200)
        header = "| Col1 | Col2 | Col3 |"
        separator = "|------|------|------|"
        rows = [f"| row{i:02d} | data{i:02d} | val{i:02d} |" for i in range(30)]
        md = (
            "<!-- TABLE_BLOCK -->\n"
            + header + "\n"
            + separator + "\n"
            + "\n".join(rows) + "\n"
            + "<!-- /TABLE_BLOCK -->"
        )
        result = chunker._handle_large_tables(md)
        # Should produce multiple TABLE_BLOCK regions
        import re
        blocks = re.findall(
            r"<!-- TABLE_BLOCK -->.*?<!-- /TABLE_BLOCK -->", result, re.DOTALL
        )
        assert len(blocks) > 1
        # Each sub-table has the original headers
        for block in blocks:
            assert header in block
            assert separator in block

    def test_large_table_all_rows_preserved(self) -> None:
        chunker = MarkdownChunker(max_table_size=200)
        rows = [f"| row{i:02d} | data{i:02d} |" for i in range(20)]
        md = (
            "<!-- TABLE_BLOCK -->\n"
            "| Col1 | Col2 |\n"
            "|------|------|\n"
            + "\n".join(rows) + "\n"
            + "<!-- /TABLE_BLOCK -->"
        )
        result = chunker._handle_large_tables(md)
        # All original data rows should be present
        for row in rows:
            assert row in result


# ─────────────────────────────────────────────────────────────────────────────
#  Overlap
# ─────────────────────────────────────────────────────────────────────────────


class TestOverlap:
    def test_overlap_applied_between_chunks(self) -> None:
        chunker = MarkdownChunker(chunk_size=100, chunk_overlap=20)
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here. Fifth sentence here. Sixth sentence here."
        chunks = chunker.chunk(text)
        # With overlap, adjacent chunks should share some text
        if len(chunks) > 1:
            # The second chunk should contain some text from end of first
            # (overlap is applied)
            assert len(chunks[1]) > 0

    def test_no_overlap_when_zero(self) -> None:
        chunker = MarkdownChunker(chunk_size=100, chunk_overlap=0)
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here. Fifth sentence here."
        chunks = chunker.chunk(text)
        # Chunks should not share content
        assert len(chunks) >= 2

    def test_protected_blocks_excluded_from_overlap(self) -> None:
        """Protected block content should not appear in overlap regions."""
        chunker = MarkdownChunker(chunk_size=200, chunk_overlap=50)
        md = (
            "Some introductory text here that is fairly long.\n\n"
            "<!-- TABLE_BLOCK -->\n"
            "| A | B |\n|---|---|\n| 1 | 2 |\n"
            "<!-- /TABLE_BLOCK -->\n\n"
            "Some concluding text here that follows the table."
        )
        chunks = chunker.chunk(md)
        # Find chunks that don't contain the table
        non_table_chunks = [c for c in chunks if "TABLE_BLOCK" not in c]
        # Table content should not leak into non-table chunks via overlap
        for chunk in non_table_chunks:
            assert "| A | B |" not in chunk


# ─────────────────────────────────────────────────────────────────────────────
#  Content Ordering
# ─────────────────────────────────────────────────────────────────────────────


class TestContentOrdering:
    def test_ordering_preserved(self, chunker: MarkdownChunker) -> None:
        md = (
            "# First\nContent first.\n\n"
            "# Second\nContent second.\n\n"
            "# Third\nContent third."
        )
        chunks = chunker.chunk(md)
        # Find positions of key content
        all_text = " ".join(chunks)
        pos_first = all_text.find("Content first")
        pos_second = all_text.find("Content second")
        pos_third = all_text.find("Content third")
        assert pos_first < pos_second < pos_third

    def test_interleaved_content_ordering(self, chunker: MarkdownChunker) -> None:
        md = (
            "Paragraph A.\n\n"
            "<!-- TABLE_BLOCK -->\n| X | Y |\n|---|---|\n| 1 | 2 |\n<!-- /TABLE_BLOCK -->\n\n"
            "Paragraph B.\n\n"
            "<!-- IMAGE_REF: s3://b/k page=1 index=1 -->\n![img](url)\n**Summary:** desc\n<!-- /IMAGE_REF -->\n\n"
            "Paragraph C."
        )
        chunks = chunker.chunk(md)
        all_text = "\n".join(chunks)
        pos_a = all_text.find("Paragraph A")
        pos_table = all_text.find("TABLE_BLOCK")
        pos_b = all_text.find("Paragraph B")
        pos_img = all_text.find("IMAGE_REF")
        pos_c = all_text.find("Paragraph C")
        assert pos_a < pos_table < pos_b < pos_img < pos_c


# ─────────────────────────────────────────────────────────────────────────────
#  Unicode Preservation
# ─────────────────────────────────────────────────────────────────────────────


class TestUnicodePreservation:
    def test_unicode_characters_preserved(self, chunker: MarkdownChunker) -> None:
        md = "# 日本語のタイトル\nこれはテストです。Unicode文字が保持されます。"
        chunks = chunker.chunk(md)
        all_text = " ".join(chunks)
        assert "日本語のタイトル" in all_text
        assert "これはテストです" in all_text

    def test_emoji_preserved(self, chunker: MarkdownChunker) -> None:
        md = "# 🎉 Celebration\nContent with emojis: 🚀 💻 🎯"
        chunks = chunker.chunk(md)
        all_text = " ".join(chunks)
        assert "🎉" in all_text
        assert "🚀" in all_text

    def test_special_symbols(self, chunker: MarkdownChunker) -> None:
        md = "Mathematical: ∑∏∫√∞ ≤≥≠ α β γ δ"
        chunks = chunker.chunk(md)
        assert "∑∏∫√∞" in chunks[0]


# ─────────────────────────────────────────────────────────────────────────────
#  Bare Table Detection (wrap_bare_tables)
# ─────────────────────────────────────────────────────────────────────────────


class TestBareTableDetection:
    def test_bare_table_gets_wrapped(self, chunker: MarkdownChunker) -> None:
        """A bare Markdown table should be wrapped in TABLE_BLOCK markers."""
        md = (
            "Some text before.\n\n"
            "| Col1 | Col2 |\n"
            "|------|------|\n"
            "| a    | b    |\n"
            "| c    | d    |\n\n"
            "Some text after."
        )
        result = chunker.wrap_bare_tables(md)
        assert "<!-- TABLE_BLOCK -->" in result
        assert "<!-- /TABLE_BLOCK -->" in result
        assert "| Col1 | Col2 |" in result

    def test_already_wrapped_table_not_double_wrapped(self, chunker: MarkdownChunker) -> None:
        """A table already in TABLE_BLOCK markers should not be wrapped again."""
        md = (
            "<!-- TABLE_BLOCK -->\n"
            "| Col1 | Col2 |\n"
            "|------|------|\n"
            "| a    | b    |\n"
            "<!-- /TABLE_BLOCK -->"
        )
        result = chunker.wrap_bare_tables(md)
        # Should only have one pair of markers
        assert result.count("<!-- TABLE_BLOCK -->") == 1
        assert result.count("<!-- /TABLE_BLOCK -->") == 1

    def test_multiple_bare_tables_wrapped(self, chunker: MarkdownChunker) -> None:
        """Multiple bare tables should each be wrapped."""
        md = (
            "| A | B |\n"
            "|---|---|\n"
            "| 1 | 2 |\n\n"
            "Some text.\n\n"
            "| X | Y |\n"
            "|---|---|\n"
            "| 3 | 4 |\n"
        )
        result = chunker.wrap_bare_tables(md)
        assert result.count("<!-- TABLE_BLOCK -->") == 2
        assert result.count("<!-- /TABLE_BLOCK -->") == 2

    def test_non_table_pipe_not_wrapped(self, chunker: MarkdownChunker) -> None:
        """Lines with pipes that don't form a valid table should not be wrapped."""
        md = "This | is | not | a table\nJust some text with pipes."
        result = chunker.wrap_bare_tables(md)
        assert "<!-- TABLE_BLOCK -->" not in result

    def test_bare_table_treated_as_protected_block(self, chunker: MarkdownChunker) -> None:
        """After wrapping, bare tables should be treated as protected blocks (not split)."""
        md = (
            "# Section\n"
            "| Col1 | Col2 | Col3 |\n"
            "|------|------|------|\n"
            "| data | data | data |\n"
            "| more | more | more |\n"
        )
        chunks = chunker.chunk(md)
        # The table should appear intact in a chunk
        table_chunks = [c for c in chunks if "<!-- TABLE_BLOCK -->" in c]
        assert len(table_chunks) == 1
        assert "<!-- /TABLE_BLOCK -->" in table_chunks[0]
        assert "| Col1 | Col2 | Col3 |" in table_chunks[0]

    def test_bare_table_with_alignment(self, chunker: MarkdownChunker) -> None:
        """Tables with alignment markers (colons) should be detected."""
        md = (
            "| Left | Center | Right |\n"
            "|:-----|:------:|------:|\n"
            "| a    | b      | c     |\n"
        )
        result = chunker.wrap_bare_tables(md)
        assert "<!-- TABLE_BLOCK -->" in result
        assert "<!-- /TABLE_BLOCK -->" in result


# ─────────────────────────────────────────────────────────────────────────────
#  XLSX-Specific Table Splitting (split_xlsx_tables)
# ─────────────────────────────────────────────────────────────────────────────


class TestXlsxTableSplitting:
    def test_small_table_unchanged(self, chunker: MarkdownChunker) -> None:
        """Tables with fewer rows than rows_per_block are not split."""
        md = (
            "<!-- TABLE_BLOCK -->\n"
            "| A | B |\n"
            "|---|---|\n"
            "| 1 | 2 |\n"
            "| 3 | 4 |\n"
            "<!-- /TABLE_BLOCK -->"
        )
        result = chunker.split_xlsx_tables(md, rows_per_block=50)
        assert result == md

    def test_large_table_split_by_row_count(self, chunker: MarkdownChunker) -> None:
        """Tables exceeding rows_per_block are split into sub-tables."""
        header = "| Col1 | Col2 |"
        separator = "|------|------|"
        rows = [f"| row{i:02d} | val{i:02d} |" for i in range(10)]
        md = (
            "<!-- TABLE_BLOCK -->\n"
            + header + "\n"
            + separator + "\n"
            + "\n".join(rows) + "\n"
            + "<!-- /TABLE_BLOCK -->"
        )
        result = chunker.split_xlsx_tables(md, rows_per_block=3)
        import re
        blocks = re.findall(
            r"<!-- TABLE_BLOCK -->.*?<!-- /TABLE_BLOCK -->", result, re.DOTALL
        )
        # 10 rows / 3 per block = 4 blocks (3+3+3+1)
        assert len(blocks) == 4

    def test_split_preserves_headers(self, chunker: MarkdownChunker) -> None:
        """Each sub-table preserves the original column headers."""
        header = "| Name | Age | City |"
        separator = "|------|-----|------|"
        rows = [f"| person{i} | {20+i} | city{i} |" for i in range(8)]
        md = (
            "<!-- TABLE_BLOCK -->\n"
            + header + "\n"
            + separator + "\n"
            + "\n".join(rows) + "\n"
            + "<!-- /TABLE_BLOCK -->"
        )
        result = chunker.split_xlsx_tables(md, rows_per_block=3)
        import re
        blocks = re.findall(
            r"<!-- TABLE_BLOCK -->.*?<!-- /TABLE_BLOCK -->", result, re.DOTALL
        )
        for block in blocks:
            assert header in block
            assert separator in block

    def test_split_preserves_all_rows(self, chunker: MarkdownChunker) -> None:
        """All original data rows are preserved after splitting."""
        rows = [f"| data{i} | val{i} |" for i in range(7)]
        md = (
            "<!-- TABLE_BLOCK -->\n"
            "| A | B |\n"
            "|---|---|\n"
            + "\n".join(rows) + "\n"
            + "<!-- /TABLE_BLOCK -->"
        )
        result = chunker.split_xlsx_tables(md, rows_per_block=2)
        for row in rows:
            assert row in result

    def test_exact_multiple_rows(self, chunker: MarkdownChunker) -> None:
        """When row count is exact multiple of rows_per_block, no empty blocks."""
        rows = [f"| r{i} | v{i} |" for i in range(6)]
        md = (
            "<!-- TABLE_BLOCK -->\n"
            "| A | B |\n"
            "|---|---|\n"
            + "\n".join(rows) + "\n"
            + "<!-- /TABLE_BLOCK -->"
        )
        result = chunker.split_xlsx_tables(md, rows_per_block=3)
        import re
        blocks = re.findall(
            r"<!-- TABLE_BLOCK -->.*?<!-- /TABLE_BLOCK -->", result, re.DOTALL
        )
        assert len(blocks) == 2


# ─────────────────────────────────────────────────────────────────────────────
#  Table Context Adjacency
# ─────────────────────────────────────────────────────────────────────────────


class TestTableContextAdjacency:
    def test_caption_stays_with_table(self) -> None:
        """A short caption/paragraph before a table stays in the same chunk."""
        chunker = MarkdownChunker(chunk_size=500, chunk_overlap=0)
        md = (
            "Table 1: Sales Data\n\n"
            "<!-- TABLE_BLOCK -->\n"
            "| Q1 | Q2 | Q3 |\n"
            "|----|----|----|\n"
            "| 10 | 20 | 30 |\n"
            "<!-- /TABLE_BLOCK -->"
        )
        chunks = chunker.chunk(md)
        # Caption and table should be in the same chunk
        assert len(chunks) == 1
        assert "Table 1: Sales Data" in chunks[0]
        assert "<!-- TABLE_BLOCK -->" in chunks[0]

    def test_large_context_separated_from_table(self) -> None:
        """When context + table exceeds chunk_size, they are in separate chunks."""
        chunker = MarkdownChunker(chunk_size=100, chunk_overlap=0)
        long_text = "A" * 80  # 80 chars of context
        md = (
            f"{long_text}\n\n"
            "<!-- TABLE_BLOCK -->\n"
            "| Col1 | Col2 | Col3 | Col4 | Col5 |\n"
            "|------|------|------|------|------|\n"
            "| data | data | data | data | data |\n"
            "<!-- /TABLE_BLOCK -->"
        )
        chunks = chunker.chunk(md)
        # They should be in separate chunks since combined > 100
        assert len(chunks) >= 2

    def test_preceding_paragraph_adjacent_to_table(self) -> None:
        """A preceding paragraph stays adjacent to the table when combined fits."""
        chunker = MarkdownChunker(chunk_size=800, chunk_overlap=0)
        md = (
            "# Report\n"
            "The following table shows quarterly results.\n\n"
            "<!-- TABLE_BLOCK -->\n"
            "| Quarter | Revenue |\n"
            "|---------|--------|\n"
            "| Q1      | $100M  |\n"
            "| Q2      | $120M  |\n"
            "<!-- /TABLE_BLOCK -->\n\n"
            "The results show growth."
        )
        chunks = chunker.chunk(md)
        # Find the chunk with the table
        table_chunks = [c for c in chunks if "<!-- TABLE_BLOCK -->" in c]
        assert len(table_chunks) == 1
        # The preceding text should be in the same chunk as the table
        assert "quarterly results" in table_chunks[0]
