"""Tests for the retrieval layer.

The chunker is tested directly (pure function, no dependencies). The
retrieval-augmented pipeline is tested with a fake retriever and fake LLM, so
no embedding model download or network call is needed.
"""
from __future__ import annotations

from llm_qa.chains.pipeline import QAPipeline
from llm_qa.config.settings import Settings
from llm_qa.retrieval.chunking import chunk_text
from llm_qa.retrieval.vector_store import RetrievedChunk


def _settings() -> Settings:
    return Settings(together_api_key="test", max_refinement_iterations=3)


# --- Chunking -----------------------------------------------------------
def test_chunking_produces_chunks() -> None:
    text = " ".join(f"Sentence number {i}." for i in range(200))
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    assert all(len(c.text) <= 260 for c in chunks)  # size + overlap headroom
    # chunk ids are sequential
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))


def test_chunking_overlap_preserves_context() -> None:
    text = "Alpha one. Bravo two. Charlie three. Delta four. Echo five."
    chunks = chunk_text(text, chunk_size=30, chunk_overlap=15)
    # With overlap, consecutive chunks should share some text.
    assert len(chunks) >= 2


def test_chunking_rejects_bad_overlap() -> None:
    import pytest

    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=100, chunk_overlap=100)


# --- Retrieval pipeline -------------------------------------------------
class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    def retrieve(self, question: str, top_k: int = 5) -> list[RetrievedChunk]:
        return self._chunks[:top_k]

    @staticmethod
    def format_context(chunks: list[RetrievedChunk]) -> str:
        return "\n\n".join(f"[{i + 1}] {c.text}" for i, c in enumerate(chunks))


def test_answer_with_retrieval_grounds_in_chunks(monkeypatch) -> None:
    chunks = [
        RetrievedChunk(
            chunk_id=7, text="Global growth slows to 2.8% in 2026.", score=0.9
        ),
        RetrievedChunk(
            chunk_id=8, text="Inflation rises to 4.0% in 2026.", score=0.8
        ),
    ]
    pipeline = QAPipeline(
        llm=None,  # type: ignore[arg-type]
        settings=_settings(),
        retriever=FakeRetriever(chunks),  # type: ignore[arg-type]
    )

    # Script the LLM: first an answer, then a clean validation.
    scripted = iter(
        ["Global growth slows to 2.8% in 2026 [1].", "All claims SUPPORTED."]
    )
    monkeypatch.setattr(
        QAPipeline, "_run_chain", lambda self, t, i: next(scripted)
    )

    result = pipeline.answer_with_retrieval("What is 2026 growth?")

    assert result.fully_grounded is True
    assert result.retrieved_chunk_ids == [7, 8]
    assert "2.8%" in result.final_answer


def test_answer_with_retrieval_requires_retriever() -> None:
    import pytest

    pipeline = QAPipeline(llm=None, settings=_settings())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="without a retriever"):
        pipeline.answer_with_retrieval("anything?")
