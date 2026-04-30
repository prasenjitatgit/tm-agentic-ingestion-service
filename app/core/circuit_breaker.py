"""Module-level `pybreaker` circuit breakers for VLM and embedding calls.

Per spec the breakers are *configured* (not just present):
    fail_max=3, reset_timeout=60
Both breakers also exclude `pybreaker.CircuitBreakerError` itself from
counting toward `fail_max` and emit structured state-transition logs.
"""

from __future__ import annotations

import pybreaker

from app.core.config import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)


class _StateLogger(pybreaker.CircuitBreakerListener):
    """Structured logger for breaker state changes."""

    def __init__(self, breaker_name: str) -> None:
        self.name = breaker_name

    def state_change(
        self,
        cb: pybreaker.CircuitBreaker,
        old_state,  # noqa: ANN001 — pybreaker uses untyped dynamic state objects
        new_state,  # noqa: ANN001
    ) -> None:
        log.warning(
            "circuit_breaker_state_change",
            breaker=self.name,
            old_state=getattr(old_state, "name", str(old_state)),
            new_state=getattr(new_state, "name", str(new_state)),
        )


_settings = get_settings()

vlm_breaker: pybreaker.CircuitBreaker = pybreaker.CircuitBreaker(
    fail_max=_settings.cb_vlm_fail_max,
    reset_timeout=_settings.cb_vlm_reset_timeout,
    name="vlm",
    listeners=[_StateLogger("vlm")],
    exclude=[pybreaker.CircuitBreakerError],
)

embed_breaker: pybreaker.CircuitBreaker = pybreaker.CircuitBreaker(
    fail_max=_settings.cb_embed_fail_max,
    reset_timeout=_settings.cb_embed_reset_timeout,
    name="embed",
    listeners=[_StateLogger("embed")],
    exclude=[pybreaker.CircuitBreakerError],
)
