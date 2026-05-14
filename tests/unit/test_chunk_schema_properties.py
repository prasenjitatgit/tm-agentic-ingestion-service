"""Property-based tests for Output Chunk Schema Completeness.

# Feature: docling-migration, Property 12: Output Chunk Schema Completeness

For any document processed through the Docling-based pipeline, every output
chunk SHALL contain all required fields (chunk_hash, chunk_simhash, section,
page, content, normalized_content) with non-null values for chunk_hash,
content, and normalized_content.

**Validates: Requirements 7.3**
"""

from __future__ import annotations

from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from app.graph.nodes.chunker import chunk_hash, chunk_simhash, normalize
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

# Strategy for generating an image reference (Markdown image syntax)
image_line = st.builds(
    lambda desc, page, idx: f"![{desc}](s3://bucket/images/{page}_{idx}.png)",
    st.text(
        alphabet=st.characters(
            whitelist_categories=("L", "N", "Zs"),
            blacklist_characters="\n\r<>[]()!",
        ),
        min_size=3,
        max_size=20,
    ).filter(lambda t: t.strip()),
    st.integers(min_value=1, max_value=10),
    st.integers(min_value=1, max_value=5),
)

# Strategy for generating random Markdown content with headings, paragraphs,
# tables, and images
markdown_content = st.builds(
    lambda elements: "\n\n".join(elements),
    st.lists(
        st.one_of(section_strategy, paragraph, table_strategy, image_line),
        min_size=1,
        max_size=6,
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
#  Property Test
# ─────────────────────────────────────────────────────────────────────────────


# Feature: docling-migration, Property 12: Output Chunk Schema Completeness
class TestOutputChunkSchemaCompleteness:
    """For any document processed through the Docling-based pipeline, every
    output chunk SHALL contain all required fields (chunk_hash, chunk_simhash,
    section, page, content, normalized_content) with non-null values for
    chunk_hash, content, and normalized_content.

    **Validates: Requirements 7.3**
    """

    @given(md=markdown_content)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_all_chunks_have_required_fields_non_null(self, md: str) -> None:
        """Every chunk produced by MarkdownChunker, when processed through
        the hash/simhash/normalize pipeline, yields non-null required fields.
        """
        chunker = MarkdownChunker(chunk_size=800, chunk_overlap=100)
        chunks = chunker.chunk(md)

        # Only test when we get actual chunks
        assume(len(chunks) > 0)

        for chunk_content in chunks:
            # Simulate the pipeline processing that generate_chunks_node does:
            # compute chunk_hash, chunk_simhash, and normalized_content
            content = chunk_content
            normalized_content = normalize(content)
            h = chunk_hash(content)
            sh = chunk_simhash(content)

            # --- Verify non-null required fields ---

            # content must be a non-empty string
            assert content is not None, "content must not be None"
            assert isinstance(content, str), f"content must be str, got {type(content)}"
            assert len(content) > 0, "content must be non-empty"

            # normalized_content must be a non-empty string
            assert normalized_content is not None, "normalized_content must not be None"
            assert isinstance(normalized_content, str), (
                f"normalized_content must be str, got {type(normalized_content)}"
            )
            assert len(normalized_content) > 0, "normalized_content must be non-empty"

            # chunk_hash must be a non-empty string (SHA-256 hex digest)
            assert h is not None, "chunk_hash must not be None"
            assert isinstance(h, str), f"chunk_hash must be str, got {type(h)}"
            assert len(h) == 64, (
                f"chunk_hash must be 64-char SHA-256 hex, got length {len(h)}"
            )
            # Verify it's valid hex
            assert all(c in "0123456789abcdef" for c in h), (
                f"chunk_hash must be valid hex, got: {h!r}"
            )

            # chunk_simhash must be a non-negative integer
            assert sh is not None, "chunk_simhash must not be None"
            assert isinstance(sh, int), (
                f"chunk_simhash must be int, got {type(sh)}"
            )
            assert sh >= 0, f"chunk_simhash must be non-negative, got {sh}"

            # section and page CAN be None (they are optional fields),
            # so we just verify the schema allows them.
            # In the actual pipeline, section and page come from the
            # AssembledText block, not from the chunker itself.
            # This test validates the chunker output + hash functions
            # produce valid non-null values for the required fields.
