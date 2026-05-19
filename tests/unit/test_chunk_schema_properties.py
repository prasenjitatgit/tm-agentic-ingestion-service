"""Property-based tests for Output Chunk Schema Completeness (Simplified).

# Feature: document-schema-redesign

For any document processed through the pipeline, every output chunk SHALL
contain only the simplified fields: content and page_no. The chunker no longer
computes or stores normalized_content, chunk_hash, or chunk_simhash.

**Validates: Requirements 5.1, 12.1, 12.2**
"""

from __future__ import annotations

from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from app.graph.nodes.markdown_chunker import MarkdownChunker


# ─────────────────────────────────────────────────────────────────────────────
#  Strategies
# ─────────────────────────────────────────────────────────────────────────────

# Strategy for generating heading levels (H1, H2, H3)
heading_level = st.sampled_from(["#", "##", "###"])

# Strategy for generating non-empty text lines
text_line = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "S", "Zs"),
        blacklist_characters="#<>!\n\r|",
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
        alphabet=st.characters(
            whitelist_categories=("L", "N", "Zs"),
            blacklist_characters="\n\r#<>|",
        ),
        min_size=3,
        max_size=30,
    ).filter(lambda t: t.strip()),
)

# Strategy for generating a section (heading + paragraphs)
section_strategy = st.builds(
    lambda h, ps: f"{h}\n\n" + "\n\n".join(ps),
    heading_line,
    st.lists(paragraph, min_size=1, max_size=3),
)

# Strategy for generating a Markdown table
table_strategy = st.builds(
    lambda cols, rows: (
        "| " + " | ".join(f"Col{i}" for i in range(1, cols + 1)) + " |\n"
        + "|" + "|".join("---" for _ in range(cols)) + "|\n"
        + "\n".join(
            "| " + " | ".join(f"r{r}c{c}" for c in range(1, cols + 1)) + " |"
            for r in range(1, rows + 1)
        )
    ),
    st.integers(min_value=2, max_value=4),
    st.integers(min_value=1, max_value=5),
)

# Strategy for generating random Markdown content with headings, paragraphs,
# and tables
markdown_content = st.builds(
    lambda elements: "\n\n".join(elements),
    st.lists(
        st.one_of(section_strategy, paragraph, table_strategy),
        min_size=1,
        max_size=6,
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
#  Property Test
# ─────────────────────────────────────────────────────────────────────────────


class TestOutputChunkSchemaCompleteness:
    """For any document processed through the chunker, every output chunk
    SHALL contain non-empty content. The chunker no longer produces
    normalized_content, chunk_hash, or chunk_simhash fields.

    **Validates: Requirements 5.1, 12.1, 12.2**
    """

    @given(md=markdown_content)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_all_chunks_have_non_empty_content(self, md: str) -> None:
        """Every chunk produced by MarkdownChunker yields non-empty content."""
        chunker = MarkdownChunker(chunk_size=800, chunk_overlap=100)
        chunks = chunker.chunk(md)

        # Only test when we get actual chunks
        assume(len(chunks) > 0)

        for chunk_content in chunks:
            # content must be a non-empty string
            assert chunk_content is not None, "content must not be None"
            assert isinstance(chunk_content, str), (
                f"content must be str, got {type(chunk_content)}"
            )
            assert len(chunk_content) > 0, "content must be non-empty"
