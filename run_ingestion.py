"""Standalone ingestion script.

Processes all pending documents (status = TO BE INGESTED, CHUNKED, or EMBEDDED)
sequentially through the RAG ingestion pipeline without requiring a running
FastAPI server.

Usage:
    python run_ingestion.py [--max-documents N] [--knowledge-base-type TYPE]

Examples:
    python run_ingestion.py                          # Process all pending (up to 100)
    python run_ingestion.py --max-documents 5        # Process up to 5 documents
    python run_ingestion.py --knowledge-base-type Maintenance  # Only Maintenance docs
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid

from app.core.config import get_settings
from app.graph import run_ingestion_job
from app.utils.logger import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)


def main() -> int:
    """Run the ingestion pipeline as a standalone process."""
    parser = argparse.ArgumentParser(
        description="Run the RAG ingestion pipeline for pending documents."
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=0,
        help="Maximum number of documents to process (default: 0 = unlimited, process all pending)",
    )
    parser.add_argument(
        "--knowledge-base-type",
        choices=["Maintenance", "Construction", "BusinessIntelligence"],
        default=None,
        help="Filter by knowledge base type (default: all)",
    )
    args = parser.parse_args()

    settings = get_settings()
    job_id = str(uuid.uuid4())

    request_filter: dict[str, str] = {}
    if args.knowledge_base_type:
        request_filter["knowledge_base_type"] = args.knowledge_base_type

    # 0 means unlimited — use a large number; the loop breaks when no docs remain
    max_docs = args.max_documents if args.max_documents > 0 else 999_999

    log.info(
        "ingestion_script_started",
        job_id=job_id,
        max_documents=args.max_documents,
        knowledge_base_type=args.knowledge_base_type,
    )

    summary = run_ingestion_job(
        request_filter=request_filter,
        max_documents=max_docs,
        job_id=job_id,
    )

    # Print summary to stdout
    print(json.dumps(summary, indent=2, default=str))

    # Exit with non-zero if any failures occurred
    if summary.get("status") == "CRASHED":
        return 1
    if summary.get("failed"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
