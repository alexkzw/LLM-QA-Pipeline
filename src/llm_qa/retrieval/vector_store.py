"""Vector store backed by ChromaDB.

ChromaDB is free, runs embedded (no separate server), and persists to disk, so
the index survives restarts. This is the component that lets us retrieve only
the most relevant chunks for a question instead of passing the whole document.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llm_qa.core.exceptions import LLMQAError, RetrievalError
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
        self._collection_name = collection_name

        try:
            # need persistence because indexing is meant to happen once
            self._client = chromadb.PersistentClient(path=str(persist_dir))
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                # explicitly use cosine similarity - Chroma's default is
                # squared L2. HNSW - Hierarchical Navigable Small World - an
                # approximate nearest-neighbour graph, suitable if the
                # corpus gets too large for brute-force search.
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:  # noqa: BLE001 - typed re-raise at the boundary
            raise RetrievalError(f"Failed to open vector store: {exc}") from exc

    def __len__(self) -> int:
        return self._collection.count()

    def clear(self) -> None:
        """Delete all chunks, leaving an empty collection ready for re-indexing.

        ``index_chunks`` upserts by id, so it never removes chunks left over
        from a *previous* document that the new one doesn't overwrite (e.g.
        re-indexing a shorter document leaves the old tail's chunks orphaned in
        the store, silently polluting future retrieval). Call this before
        rebuilding from a different document.
        """
        try:
            self._client.delete_collection(name=self._collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:  # noqa: BLE001 - typed re-raise at the boundary
            raise RetrievalError(f"Failed to clear vector store: {exc}") from exc

    def index_chunks(self, chunks: list[Chunk], batch_size: int = 128) -> None:
        """Embed and store chunks. Idempotent on chunk_id (upsert)."""
        if not chunks:
            logger.warning("No chunks to index.")
            return

        logger.info("Indexing %d chunks...", len(chunks))
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = self._embedder.embed([c.text for c in batch])
            try:
                self._collection.upsert(
                    ids=[str(c.chunk_id) for c in batch],
                    # Chroma's stub type is stricter than what it actually
                    # accepts at runtime (plain list[list[float]] works
                    # fine, verified extensively) - the stub just doesn't
                    # model it.
                    embeddings=vectors,  # type: ignore[arg-type]
                    documents=[c.text for c in batch],
                    metadatas=[
                        {"char_start": c.char_start, "char_end": c.char_end}
                        for c in batch
                    ],
                )
            except LLMQAError:
                raise
            except Exception as exc:  # noqa: BLE001 - typed re-raise
                raise RetrievalError(f"Failed to index chunks: {exc}") from exc
        logger.info("Index now holds %d chunks.", len(self))

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return the ``top_k`` chunks most similar to ``query``."""
        query_vec = self._embedder.embed_one(query)
        try:
            results = self._collection.query(
                query_embeddings=[query_vec],  # type: ignore[arg-type]
                n_results=top_k,
            )
        except Exception as exc:  # noqa: BLE001 - typed re-raise at the boundary
            raise RetrievalError(f"Failed to search vector store: {exc}") from exc

        # Chroma types these as optional (None only if the query itself
        # failed, which would have raised already) - assert narrows the
        # type and documents the invariant rather than silently indexing
        # a value mypy considers possibly-None.
        assert results["ids"] is not None
        assert results["documents"] is not None
        assert results["distances"] is not None
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
                    # encapsulate so that we normalise to one consistent semantic
                    # i.e., higher is better
                    score=1.0 - float(distance),
                )
            )
        return retrieved
