"""High-level retriever: the public face of the retrieval subsystem.

Combines chunking, embedding, and the vector store into two operations:
  * ``index_document`` - chunk a document and build/refresh the index (once).
  * ``retrieve`` - get the most relevant chunks for a question (per query).

Downstream, the pipeline calls ``retrieve`` and passes only those chunks to the
LLM, instead of the entire document.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from llm_qa.core.exceptions import RetrievalError
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
        # Sidecar file (not Chroma metadata - see index_document's docstring)
        # recording what produced the current index, so staleness is
        # detectable instead of silently trusted.
        self._fingerprint_path = Path(persist_dir) / "index_fingerprint.json"

    @property
    def embedder(self) -> EmbeddingModel:
        """The embedder backing this index - reused by anything else that
        needs vectors comparable to it (e.g. dynamic few-shot selection)."""
        return self._embedder

    @property
    def is_indexed(self) -> bool:
        return len(self._store) > 0

    def _fingerprint(self, text: str) -> str:
        """Fingerprint everything that determines what the index *should* hold.

        Content, chunking parameters, and the embedding model name: if any of
        these differ from what's on disk, the persisted vectors no longer
        correspond to this document/config (or, for the embedding model, may
        even live in a different, incompatible vector space).
        """
        payload = {
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "chunk_size": self._chunk_size,
            "chunk_overlap": self._chunk_overlap,
            "embedding_model": self._embedder.model_name,
        }
        return json.dumps(payload, sort_keys=True)

    def _stored_fingerprint(self) -> str | None:
        if not self._fingerprint_path.exists():
            return None
        return self._fingerprint_path.read_text(encoding="utf-8")

    def _write_fingerprint(self, fingerprint: str) -> None:
        self._fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
        self._fingerprint_path.write_text(fingerprint, encoding="utf-8")

    def index_document(self, text: str, force: bool = False) -> int:
        """Chunk and index a document. Skips work if already indexed.

        "Already indexed" is verified, not assumed: a fingerprint of the
        document text, chunk settings, and embedding model name is persisted
        alongside the index. If the index is non-empty but the fingerprint
        doesn't match (different document, different chunk settings, or a
        swapped embedding model), the existing index is stale and gets
        rebuilt automatically rather than silently serving wrong results. A
        non-empty index with no recorded fingerprint (e.g. built before this
        check existed) is ambiguous, not assumed stale - it's kept as-is, with
        a warning that correctness can't be verified.

        Returns the number of chunks in the index.
        """
        fingerprint = self._fingerprint(text)

        if self.is_indexed and not force:
            stored = self._stored_fingerprint()
            if stored is None:
                logger.warning(
                    "Index has %d chunk(s) but no recorded fingerprint "
                    "(built before this check existed); correctness against "
                    "the current document can't be verified. Skipping. "
                    "Pass force=True to rebuild safely.",
                    len(self._store),
                )
                return len(self._store)
            if stored == fingerprint:
                logger.info(
                    "Index already populated (%d chunks) and matches this "
                    "document/config; skipping.",
                    len(self._store),
                )
                return len(self._store)
            logger.warning(
                "Existing index doesn't match this document, chunk settings, "
                "or embedding model; it is stale. Rebuilding automatically."
            )
        elif self.is_indexed and force:
            logger.info(
                "Rebuilding index: clearing %d existing chunk(s) first.",
                len(self._store),
            )

        if self.is_indexed:
            self._store.clear()

        chunks = chunk_text(
            text,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        self._store.index_chunks(chunks)
        self._write_fingerprint(fingerprint)
        return len(self._store)

    def retrieve(self, question: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return the most relevant chunks for a question."""
        if not self.is_indexed:
            raise RetrievalError("Retriever has no index; call index_document first.")
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
