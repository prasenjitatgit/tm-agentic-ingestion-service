"""Document download and verification node.

The `select_parser` node downloads the document from S3, verifies its
SHA-256 hash, and stores the local path in state. The actual conversion
is handled by the `docling_convert` node downstream.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from typing import Any

from app.graph.nodes.status import node
from app.graph.state import AgentState
from app.services.s3_service import S3Client

_s3 = S3Client()

# Map document type → file extension on disk.
_EXT_BY_TYPE = {"PDF": ".pdf", "DOCX": ".docx", "PPT": ".pptx", "EXCEL": ".xlsx"}


def _sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@node("select_parser")
def select_parser_node(state: AgentState) -> dict[str, Any]:
    """Validate `doc_type`, download binary from S3, and verify SHA-256.

    The actual conversion is handled by `docling_convert` downstream.
    """
    meta = state["doc_metadata"]
    if meta.doc_type not in _EXT_BY_TYPE:
        raise ValueError(f"Unsupported doc_type: {meta.doc_type!r}")

    suffix = _EXT_BY_TYPE[meta.doc_type]
    fd, dest_path = tempfile.mkstemp(prefix=f"ingest-{meta.doc_id}-", suffix=suffix)
    os.close(fd)
    _s3.download_to_path(meta.s3_url, dest_path)

    actual = _sha256_of_file(dest_path)
    if actual.lower() != meta.doc_hash.lower():
        raise ValueError(
            f"doc_hash mismatch: expected={meta.doc_hash} actual={actual} for {meta.s3_url}"
        )

    return {"local_path": dest_path}
