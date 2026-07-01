"""Local embedding model wrapper.

Uses a sentence-transformers model that runs locally with no API cost. This
keeps the whole RAG pipeline free to operate: embeddings are computed on the
machine rather than via a paid embeddings API.

The model is loaded lazily and cached so repeated calls are cheap.
"""
from __future__ import annotations

from functools import lru_cache

from llm_qa.core.logging_config import get_logger

logger = get_logger(__name__)

# Small, fast, widely-used embedding model. 384-dimensional vectors; good
# quality-for-size and CPU-friendly, so it runs on free infrastructure.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    """Load and cache a SentenceTransformer model (imported lazily)."""
    # Imported here so the dependency is only needed when embeddings are used,
    # keeping import time low for callers that don't touch retrieval.
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model: %s", model_name)
    return SentenceTransformer(model_name)


class EmbeddingModel:
    """Thin wrapper over a sentence-transformers model."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model = _load_model(model_name)

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vectors."""
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,  # cosine similarity via dot product
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text into one vector."""
        return self.embed([text])[0]
