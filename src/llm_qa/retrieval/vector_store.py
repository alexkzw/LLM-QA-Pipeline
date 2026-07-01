"""Vector store backed by ChromaDB.

ChromaDB is free, runs embedded (no separate server), and persists to disk, so
the index survives restarts. This is the component that lets us retrieve only
the most relevant chunks for a question instead of passing the whole document.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llm_qa.core.logging_config import get_logger
from llm_qa.retrieval.chunking import Chunk
from llm_qa.retrieval.embeddings import EmbeddingModel

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk returned from a similarity search, with its distance score."""

    chunk_id: int
    text: str
    score: float  # similarity score (higher = more relevant)


class VectorStore:
    """A persistent Chroma collection of document chunks."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        persist_dir: str | Path = ".chroma",
        collection_name: str = "documents",
    ) -> None:
        import chromadb

        self._embedder = embedding_model
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def __len__(self) -> int:
        return self._collection.count()

    def index_chunks(self, chunks: list[Chunk], batch_size: int = 128) -> None:
        """Embed and store chunks. Idempotent on chunk_id (upsert)."""
        if not chunks:
            logger.warning("No chunks to index.")
            return

        logger.info("Indexing %d chunks...", len(chunks))
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = self._embedder.embed([c.text for c in batch])
            self._collection.upsert(
                ids=[str(c.chunk_id) for c in batch],
                embeddings=vectors,
                documents=[c.text for c in batch],
                metadatas=[
                    {"char_start": c.char_start, "char_end": c.char_end}
                    for c in batch
                ],
            )
        logger.info("Index now holds %d chunks.", len(self))

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return the ``top_k`` chunks most similar to ``query``."""
        query_vec = self._embedder.embed_one(query)
        results = self._collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
        )

        ids = results["ids"][0]
        docs = results["documents"][0]
        distances = results["distances"][0]

        retrieved: list[RetrievedChunk] = []
        for chunk_id, text, distance in zip(ids, docs, distances, strict=False):
            # Chroma returns cosine *distance*; convert to a similarity score.
            retrieved.append(
                RetrievedChunk(
                    chunk_id=int(chunk_id),
                    text=text,
                    score=1.0 - float(distance),
                )
            )
        return retrieved
