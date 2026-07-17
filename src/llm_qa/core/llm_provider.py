"""LangChain-compatible Cloudflare Workers AI LLM wrapper.
* credentials and tuning come from Settings (no module-level globals),
* transient provider failures are retried with exponential backoff,
* all upstream failures surface as a typed ``LLMProviderError``.
"""
from __future__ import annotations

from typing import Any

import httpx
from langchain_core.language_models.llms import LLM
from pydantic import Field, PrivateAttr
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from llm_qa.config.settings import Settings
from llm_qa.core.exceptions import LLMProviderError, QuotaExhaustedError
from llm_qa.core.logging_config import get_logger

logger = get_logger(__name__)

_CLOUDFLARE_API_BASE = "https://api.cloudflare.com/client/v4/accounts"

# 5xx (transient server-side failure) and 429 (rate limit) are worth retrying;
# other 4xx status codes (bad request, bad auth, unknown model) are permanent
# and would fail identically on every attempt.
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# Cloudflare's own error code for "daily free-tier neuron allocation used
# up". This is a 429, but NOT a transient rate limit - it will fail
# identically on every attempt until the provider's daily reset, so
# retrying it (or pacing requests further apart) cannot help at all.
_QUOTA_EXHAUSTED_ERROR_CODE = 4006


def _is_quota_exhausted(response: httpx.Response) -> bool:
    try:
        errors = response.json().get("errors", [])
    except ValueError:
        return False
    return any(e.get("code") == _QUOTA_EXHAUSTED_ERROR_CODE for e in errors)


def _is_retryable(exc: BaseException) -> bool:
    """Retry transient failures only, not permanent client errors.

    Network-level failures (timeouts, connection errors) and retryable HTTP
    status codes are worth another attempt. A well-formed response that
    Cloudflare flagged as failed, or an empty completion, is also retried -
    that's a response we did receive, just not a usable one, and it's
    plausibly a transient provider glitch rather than a config error we can
    diagnose from the response alone.

    The one deliberate exception: a 429 caused by exhausting the daily free
    quota looks like a rate limit but isn't one - see _is_quota_exhausted.
    """
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        if exc.response.status_code == 429 and _is_quota_exhausted(exc.response):
            return False
        return exc.response.status_code in _RETRYABLE_STATUS_CODES
    return isinstance(exc, LLMProviderError)


class CloudflareWorkersAILLM(LLM):
    """A minimal LangChain ``LLM`` backed by the Cloudflare Workers AI REST API."""

    model_name: str = Field(...)
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2000)
    max_retries: int = Field(default=3)

    # The HTTP client is runtime state, not a serialisable field.
    _client: httpx.Client = PrivateAttr()

    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        super().__init__(  # type: ignore[call-arg]  # LangChain pydantic base
            model_name=settings.model_name,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            max_retries=settings.max_retries,
            **kwargs,
        )
        self._client = httpx.Client(
            base_url=f"{_CLOUDFLARE_API_BASE}/{settings.cloudflare_account_id}/ai/run",
            headers={"Authorization": f"Bearer {settings.cloudflare_api_key}"},
            # Connect should fail fast - a slow handshake means connectivity
            # trouble, not a model that needs more time. Read gets the full
            # configured budget since that's the leg that waits on generation.
            timeout=httpx.Timeout(
                connect=5.0,
                read=settings.request_timeout_seconds,
                write=settings.request_timeout_seconds,
                pool=5.0,
            ),
        )

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        self._client.close()

    @property
    def _llm_type(self) -> str:
        return "cloudflare-workers-ai"

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> str:
        """Send a single prompt to Cloudflare Workers AI and return the completion text."""

        @retry(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            reraise=True,
        )
        def _invoke() -> str:
            response = self._client.post(
                f"/{self.model_name}",
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
            )
            response.raise_for_status()
            payload = response.json()
            if not payload.get("success", False):
                raise LLMProviderError(
                    f"Cloudflare Workers AI error: {payload.get('errors')}"
                )
            content = payload.get("result", {}).get("response")
            if not content:
                raise LLMProviderError("LLM returned an empty response.")
            return content

        try:
            return _invoke()
        except LLMProviderError as exc:
            # _invoke already raises this type directly (empty response,
            # success: false) - don't wrap an LLMProviderError in another one.
            logger.error("Cloudflare Workers AI request failed: %s", exc)
            raise
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429 and _is_quota_exhausted(exc.response):
                message = (
                    "Cloudflare Workers AI daily free-tier neuron allocation "
                    "(10,000/day) is exhausted. This is a hard quota, not a "
                    "rate limit - retrying or pacing requests cannot help. "
                    "Wait for the daily reset, or upgrade to the Workers AI "
                    "paid plan."
                )
                logger.error(message)
                raise QuotaExhaustedError(message) from exc
            logger.error("Cloudflare Workers AI request failed: %s", exc)
            raise LLMProviderError(f"LLM provider request failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - typed re-raise
            logger.error("Cloudflare Workers AI request failed: %s", exc)
            raise LLMProviderError(f"LLM provider request failed: {exc}") from exc
