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
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llm_qa.config.settings import Settings
from llm_qa.core.exceptions import LLMProviderError
from llm_qa.core.logging_config import get_logger

logger = get_logger(__name__)

_CLOUDFLARE_API_BASE = "https://api.cloudflare.com/client/v4/accounts"


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
            timeout=settings.request_timeout_seconds,
        )

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
            retry=retry_if_exception_type(Exception),
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
        except Exception as exc:  # noqa: BLE001 - typed re-raise
            logger.error("Cloudflare Workers AI request failed: %s", exc)
            raise LLMProviderError(f"LLM provider request failed: {exc}") from exc
