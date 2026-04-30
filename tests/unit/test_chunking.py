"""Tests for the tag-aware chunker."""

from __future__ import annotations

import re

from app.graph.nodes.chunker import (
    PROTECTED_TAGS,
    chunk_hash,
    chunk_simhash,
    normalize,
    split_text,
)


def _tag_intact(chunk: str, tag: str) -> bool:
    """A chunk that opens `tag` must also close it (no split mid-block)."""
    opens = len(re.findall(rf"<{tag}(?:\s[^>]*)?>", chunk))
    closes = len(re.findall(rf"</{tag}>", chunk))
    return opens == closes


def test_image_reference_never_split():
    img = (
        '<IMAGE_REFERENCE src="s3://b/k" page="1" index="1">\n'
        + ("X" * 600)
        + "\n</IMAGE_REFERENCE>"
    )
    text = "Hello world. " * 100 + img + "Tail."
    chunks = split_text(text, chunk_size=300, chunk_overlap=50)
    for tag in PROTECTED_TAGS:
        for c in chunks:
            assert _tag_intact(c, tag), f"tag {tag} split across chunks: {c!r}"


def test_oversized_table_kept_whole():
    huge = "<TABLE_REFERENCE>" + ("a," * 5000) + "</TABLE_REFERENCE>"
    chunks = split_text(huge, chunk_size=800, chunk_overlap=100)
    assert any("<TABLE_REFERENCE>" in c and "</TABLE_REFERENCE>" in c for c in chunks)


def test_speaker_notes_preserved():
    text = (
        "Slide title. Some bullet text. "
        "<SPEAKER_NOTES>Notes start " + ("y " * 200) + "</SPEAKER_NOTES> Tail."
    )
    chunks = split_text(text, chunk_size=400, chunk_overlap=50)
    for c in chunks:
        assert _tag_intact(c, "SPEAKER_NOTES")


def test_chunk_hash_normalizes_whitespace():
    assert chunk_hash("Hello   World") == chunk_hash("Hello World")


def test_simhash_positive_64bit():
    val = chunk_simhash("the quick brown fox jumps over the lazy dog")
    assert 0 <= val < 2**63


def test_normalize_lowercases_and_collapses():
    assert normalize("Hello\tWORLD\n  hi") == "hello world hi"


def test_empty_input_returns_no_chunks():
    assert split_text("", chunk_size=800, chunk_overlap=100) == []
    assert split_text("   ", chunk_size=800, chunk_overlap=100) == []
