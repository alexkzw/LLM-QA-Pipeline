"""Centralised, validated configuration loaded from environment variables.

This replaces the hard-coded API key and magic numbers scattered through the
original notebook. Every tunable lives here, is type-checked by pydantic, and
can be overridden via environment variables or a local ``.env`` file.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, populated from environment or a .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="LLMQA_",
        extra="ignore",
    )

    # --- Secrets (never hard-code; supplied via env) -------------------
    cloudflare_api_key: str = Field(
        ...,
        description="Cloudflare API key. Set via LLMQA_CLOUDFLARE_API_KEY.",
    )
    cloudflare_account_id: str = Field(
        ...,
        description="Cloudflare Account ID. Set via LLMQA_CLOUDFLARE_ACCOUNT_ID.",
    )

    # --- Model configuration ------------------------------------------
    model_name: str = Field(
        default="@cf/meta/llama-3.1-8b-instruct",
        description="Cloudflare Workers AI model identifier.",
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, gt=0)
    request_timeout_seconds: int = Field(default=60, gt=0)
    max_retries: int = Field(default=3, ge=0)

    # --- Pipeline behaviour -------------------------------------------
    max_refinement_iterations: int = Field(default=5, gt=0)
    max_reference_chars: int = Field(default=131_072, gt=0)

    # --- Retrieval (RAG) ----------------------------------------------
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)
    retrieval_top_k: int = Field(default=5, gt=0)
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Local sentence-transformers model for embeddings.",
    )
    vector_store_dir: str = Field(
        default=".chroma",
        description="Directory where the Chroma index is persisted.",
    )

    # --- Observability -------------------------------------------------
    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=False, description="Emit logs as JSON lines.")


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (read once per process)."""
    return Settings()  # type: ignore[call-arg]
