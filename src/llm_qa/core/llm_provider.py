"""LangChain-compatible Cloudflare Workers AI LLM wrapper.
* credentials and tuning come from Settings (no module-level globals),
* transient provider failures are retried with exponential backoff,
* all upstream failures surface as a typed ``LLMProviderError``.
"""
from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class TokenUsage:
    """Token counts for one completed LLM call, for cost/observability.

    Cloudflare Workers AI includes a ``usage`` block on most (not all)
    text-generation models' responses - when it's absent we still need to
    account for the call happening, so every field is optional rather than
    dropping the call from aggregate totals silently.
    """

    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None

    def __add__(self, other: TokenUsage) -> TokenUsage:
        def _sum(a: int | None, b: int | None) -> int | None:
            if a is None and b is None:
                return None
            return (a or 0) + (b or 0)

        return TokenUsage(
            prompt_tokens=_sum(self.prompt_tokens, other.prompt_tokens),
            completion_tokens=_sum(self.completion_tokens, other.completion_tokens),
            total_tokens=_sum(self.total_tokens, other.total_tokens),
        )


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
    # Usage from the most recent _call - LangChain's LLM._call signature
    # returns only the completion text, so token counts ride along here
    # instead, read back via get_last_usage() right after invoke().
    _last_usage: TokenUsage | None = PrivateAttr(default=None)

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

    def get_last_usage(self) -> TokenUsage | None:
        """Token usage from the most recent completed call, if reported.

        None either means no call has been made yet, or Cloudflare didn't
        include a usage block for this model - callers should treat that as
        "unknown", not "zero".
        """
        return self._last_usage

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
        """Send a single prompt to Cloudflare Workers AI and return the completion."""

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
            result = payload.get("result", {})
            content = result.get("response")
            if not content:
                raise LLMProviderError("LLM returned an empty response.")
            usage = result.get("usage") or {}
            self._last_usage = TokenUsage(
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
            )
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
