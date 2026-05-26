"""`MarkdownChunker` — two-stage Markdown chunker: heading-split then semantic-split.

This module provides the chunking strategy for the Docling-based pipeline.
It understands Markdown structure (headings, tables, image references,
speaker notes) and guarantees:

    1. Protected blocks (tables, image refs, speaker notes) are NEVER split
       across chunks.
    2. Heading boundaries (H1, H2, H3) define natural split points.
    3. Oversized sections are split using embedding-based semantic chunking:
       sentences are grouped by cosine similarity of their embeddings.
    4. Overlap is applied between adjacent chunks from the same section,
       but protected blocks are excluded from the overlap window.
    5. Sections smaller than chunk_size are emitted as single chunks.
    6. Content ordering is always preserved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import structlog

log = structlog.get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Data Models
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ProtectedBlock:
    """An atomic block that must not be split across chunks."""

    content: str
    block_type: str  # "table" | "image_ref" | "speaker_notes"
    placeholder: str  # unique placeholder inserted into text during extraction


@dataclass
class Section:
    """A heading-delimited section of Markdown."""

    heading: str | None
    level: int  # 0 = no heading, 1 = H1, 2 = H2, 3 = H3
    content: str


# ─────────────────────────────────────────────────────────────────────────────
#  Protected Block Patterns
# ─────────────────────────────────────────────────────────────────────────────

# Matches <!-- IMAGE_REF: ... --> ... <!-- /IMAGE_REF -->
_IMAGE_REF_PATTERN = re.compile(
    r"<!-- IMAGE_REF:.*?-->.*?<!-- /IMAGE_REF -->",
    flags=re.DOTALL,
)

# Matches <!-- TABLE_BLOCK --> ... <!-- /TABLE_BLOCK -->
_TABLE_BLOCK_PATTERN = re.compile(
    r"<!-- TABLE_BLOCK -->.*?<!-- /TABLE_BLOCK -->",
    flags=re.DOTALL,
)

# Matches <!-- SPEAKER_NOTES slide=N --> ... <!-- /SPEAKER_NOTES -->
_SPEAKER_NOTES_PATTERN = re.compile(
    r"<!-- SPEAKER_NOTES\s+slide=\d+\s*-->.*?<!-- /SPEAKER_NOTES -->",
    flags=re.DOTALL,
)

# Bare Markdown table pattern: detects tables NOT already wrapped in TABLE_BLOCK markers.
# A table is: header row (|...|), separator row (|---|...), and one or more data rows (|...|).
_BARE_TABLE_PATTERN = re.compile(
    r"(?<!\<!-- TABLE_BLOCK --\>\n)"  # negative lookbehind for TABLE_BLOCK marker
    r"(^\|[^\n]+\|\s*\n"  # header row: starts with |, ends with |
    r"^\|[\s\-:|]+\|\s*\n"  # separator row: | with dashes/colons/spaces |
    r"(?:^\|[^\n]+\|\s*\n?)+)",  # one or more data rows
    flags=re.MULTILINE,
)

# Heading pattern: lines starting with 1-3 `#` followed by a space
_HEADING_PATTERN = re.compile(r"^(#{1,3})\s+(.+)$", flags=re.MULTILINE)

# Separator hierarchy for recursive splitting
_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " "]

# Sentence boundary for splitting (period followed by space)
_SENTENCE_BOUNDARY = re.compile(r"(?<=\. )")


# ─────────────────────────────────────────────────────────────────────────────
#  MarkdownChunker
# ─────────────────────────────────────────────────────────────────────────────


class MarkdownChunker:
    """Two-stage Markdown chunker: heading-split then semantic-split.

    Oversized sections are split using embedding-based semantic chunking:
    sentences are embedded, and adjacent sentences with high cosine similarity
    are grouped into the same chunk.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        max_table_size: int = 1500,
        similarity_threshold: float = 0.75,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_table_size = max_table_size
        self.similarity_threshold = similarity_threshold
        self._embed_fn = embed_fn

    # ─────────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────────

    def chunk(self, markdown: str) -> list[str]:
        """Split Markdown into chunks respecting structure.

        Algorithm:
            1. Wrap bare Markdown tables in TABLE_BLOCK markers
            2. Handle large tables (split oversized tables into sub-tables)
            3. Extract protected blocks, replacing them with placeholders
            4. Split on heading boundaries (H1, H2, H3)
            5. For oversized sections, apply recursive splitting
            6. Re-insert protected blocks as atomic units
            7. Apply overlap (excluding protected blocks from overlap window)
        """
        if not markdown or not markdown.strip():
            return []

        # Step 0: Wrap bare Markdown tables in TABLE_BLOCK markers
        markdown = self.wrap_bare_tables(markdown)

        # Step 1: Handle large tables before extraction
        markdown = self._handle_large_tables(markdown)

        # Step 2: Extract protected blocks and replace with placeholders
        protected_blocks, text_with_placeholders = self._extract_protected_blocks(markdown)

        # Step 3: Split on heading boundaries
        sections = self._split_on_headings(text_with_placeholders)

        # Step 4: For each section, produce chunks (recursive split if oversized)
        raw_chunks: list[str] = []
        for section in sections:
            section_text = section.content.strip()
            if not section_text:
                continue

            # Calculate effective size accounting for protected block content
            effective_size = self._effective_length(section_text, protected_blocks)

            if effective_size <= self.chunk_size:
                # Section fits in one chunk — emit as-is
                raw_chunks.append(section_text)
            else:
                # Split section into parts, keeping protected blocks as atomic units
                sub_chunks = self._split_section_with_protected(
                    section_text, protected_blocks
                )
                raw_chunks.extend(sub_chunks)

        # Step 5: Apply overlap between adjacent chunks
        overlapped_chunks = self._apply_overlap(raw_chunks, protected_blocks)

        # Step 6: Re-insert protected blocks
        final_chunks = self._reinsert_protected_blocks(overlapped_chunks, protected_blocks)

        # Filter empty chunks
        final_chunks = [c.strip() for c in final_chunks if c.strip()]

        log.debug(
            "chunking_complete",
            input_length=len(markdown),
            num_chunks=len(final_chunks),
            num_protected_blocks=len(protected_blocks),
        )

        return final_chunks

    # ─────────────────────────────────────────────────────────────────────
    #  Bare Table Detection and XLSX Splitting
    # ─────────────────────────────────────────────────────────────────────

    def wrap_bare_tables(self, markdown: str) -> str:
        """Find bare Markdown tables and wrap them in TABLE_BLOCK markers.

        A bare table is one that is NOT already wrapped in
        <!-- TABLE_BLOCK --> / <!-- /TABLE_BLOCK --> markers.
        Tables are detected as sequences of lines starting with `|` that
        follow the pattern: header row, separator row (|---|), data rows.
        """

        def _is_already_wrapped(match: re.Match, text: str) -> bool:
            """Check if the matched table is already inside a TABLE_BLOCK."""
            # Look backwards from match start for an opening marker without
            # a corresponding close marker in between
            before = text[: match.start()]
            last_open = before.rfind("<!-- TABLE_BLOCK -->")
            last_close = before.rfind("<!-- /TABLE_BLOCK -->")
            if last_open == -1:
                return False
            # If the last TABLE_BLOCK open is after the last close, we're inside
            return last_open > last_close

        new_parts: list[str] = []
        last_end = 0

        for match in _BARE_TABLE_PATTERN.finditer(markdown):
            if _is_already_wrapped(match, markdown):
                continue

            table_text = match.group(0).rstrip("\n")
            wrapped = f"<!-- TABLE_BLOCK -->\n{table_text}\n<!-- /TABLE_BLOCK -->"

            new_parts.append(markdown[last_end : match.start()])
            new_parts.append(wrapped)
            last_end = match.end()

        if not new_parts:
            return markdown

        new_parts.append(markdown[last_end:])
        return "".join(new_parts)

    def split_xlsx_tables(self, markdown: str, rows_per_block: int) -> str:
        """Split tables in XLSX-converted Markdown by row count.

        For XLSX documents, tables with more than `rows_per_block` rows are
        split into sub-tables, each preserving the original column headers.
        This is called when the document is known to be XLSX format.

        Args:
            markdown: The Markdown content containing tables.
            rows_per_block: Maximum number of data rows per sub-table.

        Returns:
            Markdown with large tables split into sub-tables.
        """
        if rows_per_block <= 0:
            return markdown

        def _split_table_by_rows(match: re.Match) -> str:
            block_content = match.group(0)
            inner = block_content[len("<!-- TABLE_BLOCK -->") : -len("<!-- /TABLE_BLOCK -->")]

            lines = [line for line in inner.strip().split("\n") if line.strip()]
            if len(lines) < 3:
                return block_content

            header_line = lines[0]
            separator_line = lines[1]
            data_lines = lines[2:]

            if len(data_lines) <= rows_per_block:
                return block_content

            # Split data rows into groups of rows_per_block
            sub_tables: list[str] = []
            for i in range(0, len(data_lines), rows_per_block):
                chunk_rows = data_lines[i : i + rows_per_block]
                table_str = "\n".join([header_line, separator_line] + chunk_rows)
                sub_tables.append(
                    f"<!-- TABLE_BLOCK -->\n{table_str}\n<!-- /TABLE_BLOCK -->"
                )

            return "\n\n".join(sub_tables)

        return _TABLE_BLOCK_PATTERN.sub(_split_table_by_rows, markdown)

    def table_to_row_chunks(self, markdown: str) -> list[str]:
        """Convert Markdown tables into row-level natural language chunks.

        Each row becomes a separate chunk with column names as keys,
        producing text like: "Segment: Government, Country: Canada, Product: Carretera, ..."

        This gives each row a unique semantic fingerprint for better
        vector search retrieval on tabular data.

        Args:
            markdown: Markdown content containing TABLE_BLOCK-wrapped tables.

        Returns:
            List of natural language row descriptions. Non-table content
            is returned as-is.
        """
        chunks: list[str] = []
        last_end = 0

        for match in _TABLE_BLOCK_PATTERN.finditer(markdown):
            # Capture any non-table text before this table
            before_text = markdown[last_end:match.start()].strip()
            if before_text:
                chunks.append(before_text)

            # Parse the table
            block_content = match.group(0)
            inner = block_content[len("<!-- TABLE_BLOCK -->"):-len("<!-- /TABLE_BLOCK -->")]
            lines = [line for line in inner.strip().split("\n") if line.strip()]

            if len(lines) < 3:
                # Not a valid table — keep as-is
                chunks.append(block_content)
                last_end = match.end()
                continue

            # Parse header columns
            header_line = lines[0]
            data_lines = lines[2:]  # Skip separator line

            headers = [h.strip() for h in header_line.strip("|").split("|")]

            # Convert each row to natural language
            for row_line in data_lines:
                cells = [c.strip() for c in row_line.strip("|").split("|")]
                # Pair headers with cell values, skip empty values
                pairs = []
                for col, val in zip(headers, cells):
                    if col and val and val.strip():
                        pairs.append(f"{col}: {val}")
                if pairs:
                    chunks.append(", ".join(pairs))

            last_end = match.end()

        # Capture any trailing non-table text
        trailing = markdown[last_end:].strip()
        if trailing:
            chunks.append(trailing)

        return chunks

    # ─────────────────────────────────────────────────────────────────────
    #  Protected Block Extraction
    # ─────────────────────────────────────────────────────────────────────

    def _extract_protected_blocks(
        self, markdown: str
    ) -> tuple[list[ProtectedBlock], str]:
        """Find and extract protected blocks, replacing with unique placeholders."""
        blocks: list[ProtectedBlock] = []
        text = markdown

        patterns = [
            (_IMAGE_REF_PATTERN, "image_ref"),
            (_TABLE_BLOCK_PATTERN, "table"),
            (_SPEAKER_NOTES_PATTERN, "speaker_notes"),
        ]

        for pattern, block_type in patterns:
            offset = 0
            new_text = ""
            for match in pattern.finditer(text):
                placeholder = f"__PROTECTED_{len(blocks)}__"
                block = ProtectedBlock(
                    content=match.group(0),
                    block_type=block_type,
                    placeholder=placeholder,
                )
                blocks.append(block)
                new_text += text[offset : match.start()] + placeholder
                offset = match.end()
            new_text += text[offset:]
            text = new_text

        return blocks, text

    # ─────────────────────────────────────────────────────────────────────
    #  Heading Split
    # ─────────────────────────────────────────────────────────────────────

    def _split_on_headings(self, markdown: str) -> list[Section]:
        """Split Markdown at heading boundaries (H1, H2, H3).

        Each section includes its heading text as part of the content.
        """
        sections: list[Section] = []
        matches = list(_HEADING_PATTERN.finditer(markdown))

        if not matches:
            # No headings — entire text is one section
            return [Section(heading=None, level=0, content=markdown)]

        # Content before the first heading (if any)
        if matches[0].start() > 0:
            pre_content = markdown[: matches[0].start()]
            if pre_content.strip():
                sections.append(Section(heading=None, level=0, content=pre_content))

        # Each heading starts a new section
        for i, match in enumerate(matches):
            level = len(match.group(1))
            heading_text = match.group(2).strip()
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
            content = markdown[start:end]
            sections.append(Section(heading=heading_text, level=level, content=content))

        return sections

    # ─────────────────────────────────────────────────────────────────────
    #  Section Splitting with Protected Block Awareness
    # ─────────────────────────────────────────────────────────────────────

    def _effective_length(
        self, text: str, protected_blocks: list[ProtectedBlock]
    ) -> int:
        """Calculate the effective length of text, replacing placeholder lengths with actual content lengths."""
        length = len(text)
        for block in protected_blocks:
            if block.placeholder in text:
                length += len(block.content) - len(block.placeholder)
        return length

    def _split_section_with_protected(
        self, text: str, protected_blocks: list[ProtectedBlock]
    ) -> list[str]:
        """Split a section that contains protected blocks into appropriately sized chunks.

        Protected blocks are emitted as their own chunks. Text between protected
        blocks is recursively split if oversized.

        Table context adjacency: When text immediately precedes a protected block
        (especially tables), and their combined size fits within chunk_size, they
        are kept together in the same chunk.
        """
        # Find which protected blocks are in this text and their positions
        block_positions: list[tuple[int, int, ProtectedBlock]] = []
        for block in protected_blocks:
            pos = text.find(block.placeholder)
            if pos != -1:
                block_positions.append((pos, pos + len(block.placeholder), block))

        # Sort by position
        block_positions.sort(key=lambda x: x[0])

        if not block_positions:
            # No protected blocks — just recursive split
            return self._recursive_split(text)

        # Split text into segments: text, protected, text, protected, ...
        chunks: list[str] = []
        cursor = 0

        for start, end, block in block_positions:
            # Handle text before this protected block
            before_text = text[cursor:start].strip()
            if before_text:
                # Table context adjacency: check if preceding text + protected block
                # fit together within chunk_size
                combined_size = len(before_text) + 1 + len(block.content)  # +1 for separator
                if combined_size <= self.chunk_size:
                    # Keep context adjacent to the protected block
                    chunks.append(before_text + "\n" + block.placeholder)
                    cursor = end
                    continue
                else:
                    # Text is too large to combine — split it separately
                    if len(before_text) <= self.chunk_size:
                        chunks.append(before_text)
                    else:
                        chunks.extend(self._recursive_split(before_text))

            # Emit the protected block placeholder as its own chunk
            chunks.append(block.placeholder)
            cursor = end

        # Handle text after the last protected block
        after_text = text[cursor:].strip()
        if after_text:
            if len(after_text) <= self.chunk_size:
                chunks.append(after_text)
            else:
                chunks.extend(self._recursive_split(after_text))

        return [c for c in chunks if c.strip()]

    # ─────────────────────────────────────────────────────────────────────
    #  Semantic Split (embedding-based)
    # ─────────────────────────────────────────────────────────────────────

    def _recursive_split(self, text: str) -> list[str]:
        """Split oversized text using embedding-based semantic chunking.

        Splits text into sentences, computes embeddings, and groups adjacent
        sentences whose embeddings have high cosine similarity. Falls back to
        simple sentence-based splitting if no embed function is available.
        """
        if self._embed_fn is not None:
            return self._semantic_split(text)
        return self._fallback_split(text)

    def _semantic_split(self, text: str) -> list[str]:
        """Split text by grouping semantically similar adjacent sentences.

        Algorithm:
        1. Split text into sentences
        2. Compute embeddings for each sentence
        3. Calculate cosine similarity between adjacent sentence embeddings
        4. Identify breakpoints where similarity drops below threshold
        5. Group sentences between breakpoints into chunks
        6. Merge small groups or split large ones to respect chunk_size
        """
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            if len(sentences[0]) <= self.chunk_size:
                return sentences
            return self._hard_split(sentences[0])

        # Compute embeddings for all sentences
        try:
            embeddings = self._embed_fn(sentences)
        except Exception as exc:
            log.warning(
                "semantic_split_embed_failed",
                error=str(exc),
                fallback="sentence_split",
            )
            return self._fallback_split(text)

        if not embeddings or len(embeddings) != len(sentences):
            return self._fallback_split(text)

        # Calculate cosine similarities between adjacent sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # Find breakpoints where similarity drops below threshold
        breakpoints = [0]  # Always start at the beginning
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                breakpoints.append(i + 1)
        breakpoints.append(len(sentences))  # End marker

        # Group sentences between breakpoints
        groups: list[str] = []
        for i in range(len(breakpoints) - 1):
            start_idx = breakpoints[i]
            end_idx = breakpoints[i + 1]
            group_text = " ".join(sentences[start_idx:end_idx]).strip()
            if group_text:
                groups.append(group_text)

        # Merge small groups and split large ones to respect chunk_size
        chunks: list[str] = []
        current = ""

        for group in groups:
            if not current:
                current = group
            elif len(current) + 1 + len(group) <= self.chunk_size:
                current = current + " " + group
            else:
                if current:
                    chunks.append(current)
                current = group

        if current:
            chunks.append(current)

        # Handle any chunks that still exceed chunk_size
        final_chunks: list[str] = []
        for chunk in chunks:
            if len(chunk) <= self.chunk_size:
                final_chunks.append(chunk)
            else:
                final_chunks.extend(self._hard_split(chunk))

        return [c for c in final_chunks if c.strip()]

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences using regex-based sentence boundary detection."""
        # Split on sentence-ending punctuation followed by whitespace
        raw = re.split(r'(?<=[.!?])\s+', text.strip())
        # Filter empty and merge very short fragments
        sentences: list[str] = []
        for s in raw:
            s = s.strip()
            if not s:
                continue
            # Merge very short fragments (< 20 chars) with previous sentence
            if sentences and len(s) < 20 and not s[0].isupper():
                sentences[-1] = sentences[-1] + " " + s
            else:
                sentences.append(s)
        return sentences

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _fallback_split(self, text: str) -> list[str]:
        """Fallback: split on sentence boundaries and merge greedily by size."""
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        chunks: list[str] = []
        current = ""

        for sentence in sentences:
            candidate = (current + " " + sentence) if current else sentence
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(sentence) <= self.chunk_size:
                    current = sentence
                else:
                    chunks.extend(self._hard_split(sentence))
                    current = ""

        if current:
            chunks.append(current)

        return [c for c in chunks if c.strip()]

    def _hard_split(self, text: str) -> list[str]:
        """Last resort: split text at chunk_size boundaries."""
        chunks: list[str] = []
        while len(text) > self.chunk_size:
            # Try to find a space near the boundary to avoid mid-word splits
            split_pos = text.rfind(" ", 0, self.chunk_size)
            if split_pos == -1:
                split_pos = self.chunk_size
            chunks.append(text[:split_pos])
            text = text[split_pos:].lstrip()
        if text:
            chunks.append(text)
        return chunks

    # ─────────────────────────────────────────────────────────────────────
    #  Large Table Handling
    # ─────────────────────────────────────────────────────────────────────

    def _handle_large_tables(self, markdown: str) -> str:
        """Split tables exceeding max_table_size into sub-tables preserving headers.

        Operates on TABLE_BLOCK protected regions. If a table within a
        TABLE_BLOCK exceeds max_table_size, it is split into multiple
        TABLE_BLOCK regions, each with the original column headers.
        """

        def _split_table_block(match: re.Match) -> str:
            block_content = match.group(0)
            # Extract the table content between markers
            inner = block_content[len("<!-- TABLE_BLOCK -->") : -len("<!-- /TABLE_BLOCK -->")]

            if len(block_content) <= self.max_table_size:
                return block_content

            # Parse table lines
            lines = [line for line in inner.strip().split("\n") if line.strip()]
            if len(lines) < 3:
                # Not enough lines to split (header + separator + at least 1 row)
                return block_content

            header_line = lines[0]
            separator_line = lines[1]
            data_lines = lines[2:]

            if not data_lines:
                return block_content

            # Split data rows into sub-tables
            sub_tables: list[str] = []
            current_rows: list[str] = []
            # Calculate overhead: markers + header + separator + newlines
            header_overhead = (
                len("<!-- TABLE_BLOCK -->\n")
                + len(header_line)
                + 1  # newline
                + len(separator_line)
                + 1  # newline
                + len("\n<!-- /TABLE_BLOCK -->")
            )

            for row in data_lines:
                # Check if adding this row would exceed max_table_size
                current_content_len = header_overhead + sum(
                    len(r) + 1 for r in current_rows
                ) + len(row) + 1

                if current_rows and current_content_len > self.max_table_size:
                    # Flush current sub-table
                    table_str = "\n".join(
                        [header_line, separator_line] + current_rows
                    )
                    sub_tables.append(
                        f"<!-- TABLE_BLOCK -->\n{table_str}\n<!-- /TABLE_BLOCK -->"
                    )
                    current_rows = []

                current_rows.append(row)

            # Flush remaining rows
            if current_rows:
                table_str = "\n".join([header_line, separator_line] + current_rows)
                sub_tables.append(
                    f"<!-- TABLE_BLOCK -->\n{table_str}\n<!-- /TABLE_BLOCK -->"
                )

            return "\n\n".join(sub_tables)

        return _TABLE_BLOCK_PATTERN.sub(_split_table_block, markdown)

    # ─────────────────────────────────────────────────────────────────────
    #  Overlap Application
    # ─────────────────────────────────────────────────────────────────────

    def _apply_overlap(
        self, chunks: list[str], protected_blocks: list[ProtectedBlock]
    ) -> list[str]:
        """Apply overlap between adjacent chunks, excluding protected blocks.

        Protected block placeholders are not included in the overlap text.
        """
        if self.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks

        placeholder_set = {b.placeholder for b in protected_blocks}

        result: list[str] = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                result.append(chunk)
                continue

            # Get overlap text from the end of the previous chunk,
            # excluding any protected block placeholders
            prev_chunk = chunks[i - 1]
            overlap_text = self._get_overlap_text(prev_chunk, placeholder_set)

            if overlap_text and not self._is_protected_placeholder(chunk, placeholder_set):
                # Only prepend overlap if current chunk doesn't start with a protected block
                combined = overlap_text + " " + chunk
                # Don't exceed chunk_size with overlap (unless chunk already exceeds)
                if len(combined) <= self.chunk_size or len(chunk) > self.chunk_size:
                    result.append(combined)
                else:
                    result.append(chunk)
            else:
                result.append(chunk)

        return result

    def _get_overlap_text(
        self, chunk: str, placeholder_set: set[str]
    ) -> str:
        """Extract up to chunk_overlap characters from end of chunk, skipping placeholders."""
        # Remove any protected block placeholders from the text for overlap purposes
        clean_text = chunk
        for placeholder in placeholder_set:
            clean_text = clean_text.replace(placeholder, "")

        clean_text = clean_text.strip()
        if not clean_text:
            return ""

        # Take the last chunk_overlap characters
        if len(clean_text) <= self.chunk_overlap:
            return clean_text

        # Try to break at a word boundary
        tail = clean_text[-self.chunk_overlap :]
        space_pos = tail.find(" ")
        if space_pos != -1 and space_pos < len(tail) - 1:
            return tail[space_pos + 1 :]
        return tail

    def _is_protected_placeholder(self, chunk: str, placeholder_set: set[str]) -> bool:
        """Check if a chunk starts with a protected block placeholder."""
        stripped = chunk.strip()
        for placeholder in placeholder_set:
            if stripped.startswith(placeholder):
                return True
        return False

    # ─────────────────────────────────────────────────────────────────────
    #  Protected Block Re-insertion
    # ─────────────────────────────────────────────────────────────────────

    def _reinsert_protected_blocks(
        self, chunks: list[str], protected_blocks: list[ProtectedBlock]
    ) -> list[str]:
        """Replace placeholders with original protected block content."""
        if not protected_blocks:
            return chunks

        result: list[str] = []
        for chunk in chunks:
            for block in protected_blocks:
                chunk = chunk.replace(block.placeholder, block.content)
            result.append(chunk)

        return result
