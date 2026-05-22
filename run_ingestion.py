"""Standalone ingestion and deletion script.

Processes pending documents through the RAG ingestion pipeline or deletes
documents marked for deletion, without requiring a running FastAPI server.

Usage:
    python run_ingestion.py [--mode ingest|delete] [--max-documents N] [--knowledge-base-type TYPE]

Examples:
    python run_ingestion.py                          # Ingest all pending documents
    python run_ingestion.py --max-documents 5        # Ingest up to 5 documents
    python run_ingestion.py --mode delete            # Delete all TO BE DELETED documents
    python run_ingestion.py --knowledge-base-type Maintenance  # Only Maintenance docs
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid

from app.core.config import get_settings
from app.graph.workflow import run_ingestion_job, run_deletion_job
from app.utils.logger import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)


def main() -> int:
    """Run the ingestion or deletion pipeline as a standalone process."""
    parser = argparse.ArgumentParser(
        description="Run the RAG ingestion/deletion pipeline for pending documents."
    )
    parser.add_argument(
        "--mode",
        choices=["ingest", "delete"],
        default="ingest",
        help="Pipeline mode: 'ingest' processes pending documents, 'delete' removes TO BE DELETED documents (default: ingest)",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=0,
        help="Maximum number of documents to process (default: 0 = unlimited)",
    )
    parser.add_argument(
        "--knowledge-base-type",
        choices=["Maintenance", "Construction", "BusinessIntelligence"],
        default=None,
        help="Filter by knowledge base type (default: all, ingest mode only)",
    )
    args = parser.parse_args()

    job_id = str(uuid.uuid4())
    # 0 means unlimited — use a large number; the loop breaks when no docs remain
    max_docs = args.max_documents if args.max_documents > 0 else 999_999

    if args.mode == "delete":
        log.info(
            "deletion_script_started",
            job_id=job_id,
            max_documents=args.max_documents,
        )

        summary = run_deletion_job(
            max_documents=max_docs,
            job_id=job_id,
        )
    else:
        request_filter: dict[str, str] = {}
        if args.knowledge_base_type:
            request_filter["knowledge_base_type"] = args.knowledge_base_type

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
