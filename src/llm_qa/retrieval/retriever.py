"""High-level retriever: the public face of the retrieval subsystem.

Combines chunking, embedding, and the vector store into two operations:
  * ``index_document`` - chunk a document and build/refresh the index (once).
  * ``retrieve`` - get the most relevant chunks for a question (per query).

Downstream, the pipeline calls ``retrieve`` and passes only those chunks to the
LLM, instead of the entire document.
"""
from __future__ import annotations

from pathlib import Path

from llm_qa.core.logging_config import get_logger
from llm_qa.retrieval.chunking import chunk_text
from llm_qa.retrieval.embeddings import EmbeddingModel
from llm_qa.retrieval.vector_store import RetrievedChunk, VectorStore

logger = get_logger(__name__)


class Retriever:
    """Builds an index over a document and retrieves relevant chunks."""

    def __init__(
        self,
        persist_dir: str | Path = ".chroma",
        embedding_model: EmbeddingModel | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ) -> None:
        self._embedder = embedding_model or EmbeddingModel()
        self._store = VectorStore(self._embedder, persist_dir=persist_dir)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    @property
    def is_indexed(self) -> bool:
        return len(self._store) > 0

    def index_document(self, text: str, force: bool = False) -> int:
        """Chunk and index a document. Skips work if already indexed.

        Returns the number of chunks in the index.
        """
        if self.is_indexed and not force:
            logger.info(
                "Index already populated (%d chunks); skipping. "
                "Pass force=True to rebuild.",
                len(self._store),
            )
            return len(self._store)

        chunks = chunk_text(
            text,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        self._store.index_chunks(chunks)
        return len(self._store)

    def retrieve(self, question: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return the most relevant chunks for a question."""
        if not self.is_indexed:
            raise RuntimeError("Retriever has no index; call index_document first.")
        return self._store.search(question, top_k=top_k)

    @staticmethod
    def format_context(chunks: list[RetrievedChunk]) -> str:
        """Render retrieved chunks into a numbered context block for the prompt.

        Numbering enables the LLM to cite sources as [1], [2], etc., supporting
        the provenance/auditability goal of the pipeline.
        """
        blocks = []
        for i, chunk in enumerate(chunks, start=1):
            blocks.append(f"[{i}] (relevance {chunk.score:.2f})\n{chunk.text}")
        return "\n\n".join(blocks)
