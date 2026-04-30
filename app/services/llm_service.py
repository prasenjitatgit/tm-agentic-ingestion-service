"""NVIDIA NIM clients (OpenAI-compatible) with circuit-breaker wrapping.

* `summarize_image(image_bytes)` — VLM summary; on `CircuitBreakerError`
  returns a sentinel fallback so the pipeline can continue.
* `embed_texts(texts)` — batched embeddings; raises on breaker open
  so the document is failed (no acceptable fallback for vectors).
"""

from __future__ import annotations

import base64
import io

import pybreaker
from openai import APIError, APITimeoutError, OpenAI
from PIL import Image

from app.core.circuit_breaker import embed_breaker, vlm_breaker
from app.core.config import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)
_settings = get_settings()

VLM_FALLBACK_SUMMARY = "[image summary unavailable]"

_VLM_PROMPT = (
    "You are an expert technical document analyst. Describe this image in 2-4 concise "
    "sentences. Focus on objects, text, charts, tables, diagrams, and any technical "
    "detail useful for retrieval. Do not speculate beyond what is visible."
)

_vlm_client = OpenAI(
    api_key=_settings.nvidia_api_key or "missing",
    base_url=_settings.nvidia_base_url,
    timeout=_settings.vlm_timeout_seconds,
)
_embed_client = OpenAI(
    api_key=_settings.nvidia_api_key or "missing",
    base_url=_settings.nvidia_base_url,
    timeout=_settings.embed_timeout_seconds,
)

# Re-exported for the retry classifier.
TransientLLMErrors = (APIError, APITimeoutError)


# ── VLM ──────────────────────────────────────────────────────────────────────
def _resize(image_bytes: bytes, max_px: int) -> bytes:
    """Downscale an image so the longest side ≤ `max_px`; re-encode as PNG."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        if max(img.size) > max_px:
            img.thumbnail((max_px, max_px))
        out = io.BytesIO()
        img.save(out, format="PNG", optimize=True)
        return out.getvalue()


def _to_data_url(image_bytes: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(image_bytes).decode('ascii')}"


@vlm_breaker
def _vlm_call(data_url: str) -> str:
    """Single VLM invocation guarded by `vlm_breaker`."""
    resp = _vlm_client.chat.completions.create(
        model=_settings.vlm_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _VLM_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        max_tokens=300,
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


def summarize_image(image_bytes: bytes) -> str:
    """Return a VLM summary for a single image; falls back when breaker is open."""
    resized = _resize(image_bytes, _settings.vlm_max_image_px)
    try:
        return _vlm_call(_to_data_url(resized))
    except pybreaker.CircuitBreakerError:
        log.warning("vlm_circuit_open_fallback")
        return VLM_FALLBACK_SUMMARY


# ── Embeddings ───────────────────────────────────────────────────────────────
@embed_breaker
def _embed_call(texts: list[str]) -> list[list[float]]:
    """Single embeddings batch guarded by `embed_breaker`."""
    resp = _embed_client.embeddings.create(
        model=_settings.embed_model,
        input=texts,
        encoding_format="float",
        extra_body={"input_type": "passage", "truncate": "END"},
    )
    return [d.embedding for d in resp.data]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings in deterministic batches; preserves input order.

    Raises `pybreaker.CircuitBreakerError` when the embed breaker is open —
    the pipeline treats this as a non-retryable failure for the document.
    """
    if not texts:
        return []
    out: list[list[float]] = []
    bs = _settings.embed_batch_size
    for i in range(0, len(texts), bs):
        out.extend(_embed_call(texts[i : i + bs]))
    return out
