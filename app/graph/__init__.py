"""LangGraph orchestration package.

Public re-exports keep call sites stable:

    from app.graph import build_graph, run_ingestion_job, initial_state
"""

from app.graph.state import (
    AgentState,
    AssembledText,
    Chunk,
    DocMetadata,
    ErrorInfo,
    ImageItem,
    RetryContext,
    TextBlock,
    initial_state,
)
from app.graph.workflow import build_graph, run_ingestion_job

__all__ = [
    "AgentState",
    "AssembledText",
    "Chunk",
    "DocMetadata",
    "ErrorInfo",
    "ImageItem",
    "RetryContext",
    "TextBlock",
    "build_graph",
    "initial_state",
    "run_ingestion_job",
]
