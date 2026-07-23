"""Typed exceptions for the QA pipeline.

Using specific exception types, lets callers and the API layer handle 
each failure mode appropriately and return meaningful error responses.
"""
from __future__ import annotations


class LLMQAError(Exception):
    """Base class for all errors raised by this package."""


class ConfigurationError(LLMQAError):
    """Raised when configuration is missing or invalid."""


class DocumentError(LLMQAError):
    """Raised when a reference document cannot be read or is invalid."""


class DocumentTooLargeError(DocumentError):
    """Raised when a reference document exceeds the configured size limit."""


class LLMProviderError(LLMQAError):
    """Raised when the upstream LLM provider fails (network, auth, rate limit)."""


class QuotaExhaustedError(LLMProviderError):
    """Raised when a hard usage quota (e.g. a daily free-tier allocation) is
    exhausted. Unlike an ordinary rate limit, this will not recover within
    the same run - retrying or waiting seconds/minutes cannot help, only
    waiting for the provider's reset window (or upgrading the plan) can."""


class RetrievalError(LLMQAError):
    """Raised when the retrieval subsystem fails: querying before the index
    is built, or the underlying vector store rejecting/failing an
    index/search operation."""


class EmbeddingError(LLMQAError):
    """Raised when the embedding model fails to load (bad model name, no
    network on first download, corrupted cache, etc.)."""
