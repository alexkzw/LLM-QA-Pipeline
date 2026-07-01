"""LangChain-compatible Together AI LLM wrapper.
* credentials and tuning come from Settings (no module-level globals),
* transient provider failures are retried with exponential backoff,
* all upstream failures surface as a typed ``LLMProviderError``.
"""
from __future__ import annotations

from typing import Any

import together
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


class TogetherAILLM(LLM):
    """A minimal LangChain ``LLM`` backed by the Together AI chat API."""

    model_name: str = Field(...)
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2000)
    max_retries: int = Field(default=3)

    # The SDK client is runtime state, not a serialisable field.
    _client: together.Together = PrivateAttr()

    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        super().__init__(  # type: ignore[call-arg]  # LangChain pydantic base
            model_name=settings.model_name,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            max_retries=settings.max_retries,
            **kwargs,
        )
        self._client = together.Together(api_key=settings.together_api_key)

    @property
    def _llm_type(self) -> str:
        return "together-ai"

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> str:
        """Send a single prompt to Together AI and return the completion text."""

        @retry(
            retry=retry_if_exception_type(Exception),
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            reraise=True,
        )
        def _invoke() -> str:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            message = response.choices[0].message
            content = message.content if message is not None else None
            if not content:
                raise LLMProviderError("LLM returned an empty response.")
            return content

        try:
            return _invoke()
        except Exception as exc:  # noqa: BLE001 - typed re-raise
            logger.error("Together AI request failed: %s", exc)
            raise LLMProviderError(f"LLM provider request failed: {exc}") from exc
