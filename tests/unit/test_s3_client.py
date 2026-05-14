"""Tests for S3 url parsing and KB-partitioned key resolution."""

from __future__ import annotations

import pytest

from app.core.config import get_settings
from app.services.s3_service import S3Client, parse_s3_url


def test_kb_prefix_maintenance():
    s = get_settings()
    assert s.s3_documents_prefix("Maintenance") == "raw/maintenance/documents"
    assert s.s3_images_prefix("Maintenance") == "raw/maintenance/images"


def test_kb_prefix_construction():
    s = get_settings()
    assert s.s3_documents_prefix("Construction") == "raw/construction/documents"
    assert s.s3_images_prefix("Construction") == "raw/construction/images"


def test_kb_prefix_business_intelligence():
    s = get_settings()
    assert s.s3_documents_prefix("BusinessIntelligence") == "raw/businessintelligence/documents"
    assert s.s3_images_prefix("BusinessIntelligence") == "raw/businessintelligence/images"


def test_image_key_format():
    client = S3Client.__new__(S3Client)  # bypass __init__ so boto3 isn't required
    client._settings = get_settings()
    client._bucket = "charter-rag-test"
    key = client.image_key("Construction", "doc-123", page=4, image_index=2, ext="png")
    assert key == "raw/construction/images/doc-123/4_2.png"


def test_parse_s3_url():
    obj = parse_s3_url("s3://bucket/path/to/file.pdf")
    assert obj.bucket == "bucket"
    assert obj.key == "path/to/file.pdf"
    assert obj.url == "s3://bucket/path/to/file.pdf"


def test_parse_s3_url_invalid():
    with pytest.raises(ValueError):
        parse_s3_url("https://nope.example.com/x")
