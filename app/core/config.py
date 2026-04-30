"""Centralized settings.

Secrets (PostgreSQL DSN, NVIDIA API key) are sourced from **AWS Parameter
Store** in production. Set `USE_PARAMETER_STORE=true` and provide
`*_PARAM` paths; values fetched from SSM override the corresponding
environment variables. In local dev (USE_PARAMETER_STORE=false) the
plain env-var values from `.env` are used directly.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)


def _fetch_ssm(name: str, region: str) -> str | None:
    """Fetch a SecureString parameter from AWS Parameter Store.

    Returns None on any failure (caller should fall back to env var).
    """
    try:
        import boto3  # local import keeps boto3 optional in unit tests

        ssm = boto3.client("ssm", region_name=region)
        resp = ssm.get_parameter(Name=name, WithDecryption=True)
        return resp["Parameter"]["Value"]
    except Exception as exc:  # noqa: BLE001
        log.warning("ssm_fetch_failed", extra={"name": name, "error": str(exc)})
        return None


class Settings(BaseSettings):
    """Service configuration loaded from env / .env (+ optional SSM)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Service ─────────────────────────────────────────────────────────────
    service_name: str = "tm-agentic-ingestion-service"
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"

    # ── AWS / Parameter Store ───────────────────────────────────────────────
    use_parameter_store: bool = False
    aws_region: str = "us-east-1"
    pg_dsn_param: str = "/charter/rag/ingestion/PG_DSN"
    nvidia_api_key_param: str = "/charter/rag/ingestion/NVIDIA_API_KEY"

    # ── PostgreSQL ──────────────────────────────────────────────────────────
    # The default is a placeholder for fresh checkouts; real DSNs come from
    # `.env` (gitignored) or AWS SSM in production. Note the `+psycopg`
    # driver scheme — required by SQLAlchemy 2.x.
    pg_dsn: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"
    pg_pool_size: int = 10
    pg_pool_max_overflow: int = 10

    # ── S3 ──────────────────────────────────────────────────────────────────
    # Bucket layout: s3://{s3_bucket}/raw/{kb}/{documents|images}/...
    s3_bucket: str = "rag-496359311344-us-east-1-an"

    # ── NVIDIA NIM ──────────────────────────────────────────────────────────
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    vlm_model: str = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
    embed_model: str = "nvidia/llama-nemotron-embed-1b-v2"
    embed_dim: int = 1536
    embed_batch_size: int = 16
    vlm_max_image_px: int = 1024
    vlm_timeout_seconds: float = 120.0
    embed_timeout_seconds: float = 60.0

    # ── Pipeline ────────────────────────────────────────────────────────────
    chunk_size: int = 800
    chunk_overlap: int = 100
    max_retries: int = 3
    retry_backoff_base_seconds: float = 1.0
    retry_backoff_cap_seconds: float = 30.0

    # ── Circuit Breakers (per spec: fail_max=3, reset_timeout=60) ───────────
    cb_vlm_fail_max: int = 3
    cb_vlm_reset_timeout: int = 60
    cb_embed_fail_max: int = 3
    cb_embed_reset_timeout: int = 60

    # ── KB → S3 prefix mapping ──────────────────────────────────────────────
    KB_PREFIX_MAP: dict[str, str] = Field(
        default_factory=lambda: {
            "Maintenance": "maintenance",
            "Construction": "construction",
            "BusinessIntelligence": "businessintelligence",
        }
    )

    # ── Helpers ─────────────────────────────────────────────────────────────
    def s3_documents_prefix(self, knowledge_base_type: str) -> str:
        """Return `raw/{kb}/documents` for the given KB."""
        return f"raw/{self.KB_PREFIX_MAP[knowledge_base_type]}/documents"

    def s3_images_prefix(self, knowledge_base_type: str) -> str:
        """Return `raw/{kb}/images` for the given KB."""
        return f"raw/{self.KB_PREFIX_MAP[knowledge_base_type]}/images"

    def hydrate_from_parameter_store(self) -> None:
        """Replace secrets with SSM values when `USE_PARAMETER_STORE=true`."""
        if not self.use_parameter_store:
            return

        for attr, param_name in (
            ("pg_dsn", self.pg_dsn_param),
            ("nvidia_api_key", self.nvidia_api_key_param),
        ):
            value = _fetch_ssm(param_name, self.aws_region)
            if value:
                object.__setattr__(self, attr, value)
            else:
                log.warning(
                    "ssm_value_missing_falling_back_to_env",
                    extra={"attr": attr, "param": param_name},
                )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached `Settings` instance (after SSM hydration)."""
    settings = Settings()
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return settings
    settings.hydrate_from_parameter_store()
    return settings
