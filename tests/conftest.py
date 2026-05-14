"""Pytest fixtures: ensure required env vars exist before app modules import."""

import os

os.environ.setdefault("USE_PARAMETER_STORE", "false")
os.environ.setdefault("PG_DSN", "postgresql+psycopg://test:test@localhost:5432/test")
os.environ.setdefault("S3_BUCKET", "charter-rag-test")
os.environ.setdefault("NVIDIA_API_KEY", "test")
os.environ.setdefault("LOG_FORMAT", "console")
