"""Tests for the retrieval layer.

The chunker is tested directly (pure function, no dependencies). The
retrieval-augmented pipeline is tested with a fake retriever and fake LLM, so
no embedding model download or network call is needed.
"""
from __future__ import annotations

from llm_qa.chains.ensemble_validator import ValidatorVote
from llm_qa.chains.pipeline import QAPipeline
from llm_qa.config.settings import Settings
from llm_qa.core.exceptions import ConfigurationError
from llm_qa.retrieval.chunking import chunk_text
from llm_qa.retrieval.vector_store import RetrievedChunk


class FakeEnsembleValidator:
    """Scripted stand-in for EnsembleValidator - no real model calls."""

    def __init__(self, results: list[tuple[bool, str]]) -> None:
        self._results = list(results)

    def validate(self, reference, response):  # noqa: ARG002
        grounded, text = (
            self._results.pop(0) if self._results else (True, "SUPPORTED.")
        )
        vote = ValidatorVote(model_name="fake-model", grounded=grounded, text=text)
        return grounded, [vote]

    def close(self) -> None:
        pass


class FakeEmbedder:
    """Cheap deterministic stand-in for EmbeddingModel - no real inference."""

    model_name = "fake-embedder"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t) % 7), float(sum(map(ord, t)) % 13)] for t in texts]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


def _settings() -> Settings:
    return Settings(
        cloudflare_api_key="test",
        cloudflare_account_id="test-account",
        max_refinement_iterations=3,
    )


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


def test_chunking_hard_splits_oversized_sentence() -> None:
    # A dense table has no sentence-ending punctuation for pages at a time -
    # without a hard-split fallback, this becomes one arbitrarily large chunk.
    oversized_blob = " ".join(f"col{i}" for i in range(2000))
    assert len(oversized_blob) > 1000

    chunks = chunk_text(oversized_blob, chunk_size=1000, chunk_overlap=150)
    assert all(len(c.text) <= 1000 for c in chunks)
    assert len(chunks) > 1


def test_chunking_hard_split_coexists_with_normal_sentences() -> None:
    oversized_blob = " ".join(f"col{i}" for i in range(2000))
    text = (
        "Normal sentence one. Normal sentence two. "
        + oversized_blob
        + " Normal sentence three. Normal sentence four."
    )
    chunks = chunk_text(text, chunk_size=1000, chunk_overlap=150)
    assert all(len(c.text) <= 1000 for c in chunks)
    assert chunks[0].text.startswith("Normal sentence one.")
    assert chunks[-1].text.endswith("Normal sentence four.")


# --- Retrieval pipeline -------------------------------------------------
class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    def retrieve(self, question: str, top_k: int = 5) -> list[RetrievedChunk]:
        return self._chunks[:top_k]

    @staticmethod
    def format_context(chunks: list[RetrievedChunk]) -> str:
        return "\n\n".join(f"[{i + 1}] {c.text}" for i, c in enumerate(chunks))

    @property
    def embedder(self) -> FakeEmbedder:
        return FakeEmbedder()


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
        ensemble_validator=FakeEnsembleValidator([(True, "All claims SUPPORTED.")]),  # type: ignore[arg-type]
    )

    # Script the LLM: just the initial answer (validation no longer runs
    # through _run_chain - it's the FakeEnsembleValidator above).
    scripted = iter(["Global growth slows to 2.8% in 2026 [1]."])
    monkeypatch.setattr(
        QAPipeline, "_run_chain", lambda self, t, i: next(scripted)
    )

    result = pipeline.answer_with_retrieval("What is 2026 growth?")

    assert result.fully_grounded is True
    assert result.retrieved_chunk_ids == [7, 8]
    assert "2.8%" in result.final_answer


def test_answer_with_retrieval_requires_retriever() -> None:
    import pytest

    pipeline = QAPipeline(
        llm=None,  # type: ignore[arg-type]
        settings=_settings(),
        ensemble_validator=FakeEnsembleValidator([]),  # type: ignore[arg-type]
    )
    with pytest.raises(ConfigurationError, match="without a retriever"):
        pipeline.answer_with_retrieval("anything?")
