"""S3Client — partitioned uploads/downloads keyed by `knowledge_base_type`.

Layout in S3:
    s3://{bucket}/raw/{kb}/documents/<doc_hash or filename>
    s3://{bucket}/raw/{kb}/images/<doc_id>/<page>_<index>.<ext>
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from app.core.config import get_settings

# Re-exported tuple for callers that need to classify exceptions.
S3TransientErrors = (BotoCoreError, ClientError)


@dataclass(frozen=True)
class S3Object:
    """Lightweight (bucket, key) pair with `s3://` URL helper."""

    bucket: str
    key: str

    @property
    def url(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


def parse_s3_url(s3_url: str) -> S3Object:
    """Parse an `s3://bucket/key` URL into an `S3Object`."""
    parsed = urlparse(s3_url)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid s3 url: {s3_url!r}")
    return S3Object(bucket=parsed.netloc, key=parsed.path.lstrip("/"))


class S3Client:
    """Thin partitioned wrapper over boto3 S3."""

    def __init__(self, bucket: str | None = None, region: str | None = None) -> None:
        settings = get_settings()
        self._bucket = bucket or settings.s3_bucket
        # Build credential kwargs from settings (loaded from .env).
        # If empty, boto3 falls back to its standard credential chain.
        cred_kwargs: dict = {}
        if settings.aws_access_key_id:
            cred_kwargs["aws_access_key_id"] = settings.aws_access_key_id
        if settings.aws_secret_access_key:
            cred_kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
        if settings.aws_session_token:
            cred_kwargs["aws_session_token"] = settings.aws_session_token
        self._client = boto3.client(
            "s3",
            region_name=region or settings.aws_region,
            config=Config(
                retries={"max_attempts": 5, "mode": "standard"},
                connect_timeout=10,
                read_timeout=60,
            ),
            **cred_kwargs,
        )
        self._settings = settings

    # ── URL/key helpers ─────────────────────────────────────────────────────
    @property
    def bucket(self) -> str:
        return self._bucket

    def documents_prefix(self, knowledge_base_type: str) -> str:
        return self._settings.s3_documents_prefix(knowledge_base_type)

    def images_prefix(self, knowledge_base_type: str) -> str:
        return self._settings.s3_images_prefix(knowledge_base_type)

    def image_key(
        self,
        knowledge_base_type: str,
        doc_id: str,
        page: int,
        image_index: int,
        ext: str = "png",
    ) -> str:
        return f"{self.images_prefix(knowledge_base_type)}/{doc_id}/{page}_{image_index}.{ext}"

    # ── I/O ─────────────────────────────────────────────────────────────────
    def download_to_path(self, s3_url: str, dest_path: str) -> None:
        """Download an `s3://` URL to a local file path."""
        obj = parse_s3_url(s3_url)
        self._client.download_file(obj.bucket, obj.key, dest_path)

    def download_bytes(self, s3_url: str) -> bytes:
        """Download an `s3://` URL into memory."""
        obj = parse_s3_url(s3_url)
        buf = io.BytesIO()
        self._client.download_fileobj(obj.bucket, obj.key, buf)
        return buf.getvalue()

    def upload_image(
        self,
        knowledge_base_type: str,
        doc_id: str,
        page: int,
        image_index: int,
        data: bytes,
        ext: str = "png",
    ) -> str:
        """Upload an image into the KB-partitioned image prefix and return its `s3://` URL."""
        key = self.image_key(knowledge_base_type, doc_id, page, image_index, ext=ext)
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=data,
            ContentType=f"image/{ext}",
        )
        return f"s3://{self._bucket}/{key}"

    def list_objects(self, prefix: str) -> Iterable[str]:
        """Yield all keys under `prefix` in this client's bucket."""
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for item in page.get("Contents", []) or []:
                yield item["Key"]

    def ping(self) -> bool:
        """Cheap connectivity check used by `/readyz`."""
        self._client.head_bucket(Bucket=self._bucket)
        return True
