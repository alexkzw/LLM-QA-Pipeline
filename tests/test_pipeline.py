"""Unit tests for the QA pipeline using a fake LLM (no network calls).

These tests demonstrate the production habit of testing business logic in
isolation: the grounding/refinement loop is verified deterministically by
injecting scripted LLM responses.
"""
from __future__ import annotations

import pytest

from llm_qa.chains.pipeline import QAPipeline, _is_fully_grounded
from llm_qa.config.settings import Settings


class FakeLLM:
    """A scripted stand-in for the real LLM.

    Returns queued responses in order, so we can simulate a validator that
    first reports an unsupported claim and then reports a clean answer.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def invoke(self, _inputs) -> str:  # matches the runnable interface
        self.calls += 1
        if self._responses:
            return self._responses.pop(0)
        return "SUPPORTED. Overall verdict: all claims supported."


def _settings() -> Settings:
    return Settings(together_api_key="test-key", max_refinement_iterations=5)


def _patch_runnable(monkeypatch, fake: FakeLLM) -> None:
    """Make ``template | llm`` resolve to our FakeLLM regardless of template."""
    monkeypatch.setattr(
        QAPipeline,
        "_run_chain",
        lambda self, template, inputs: fake.invoke(inputs),
    )


@pytest.mark.parametrize(
    "validation_text,expected",
    [
        ("All claims SUPPORTED. Verdict: supported.", True),
        ("Claim 1: UNSUPPORTED", False),
        ("Claim 1: PARTIALLY SUPPORTED", False),
        ("supported", True),  # case-insensitive
    ],
)
def test_is_fully_grounded(validation_text: str, expected: bool) -> None:
    assert _is_fully_grounded(validation_text) is expected


def test_pipeline_returns_immediately_when_grounded(monkeypatch) -> None:
    # initial answer, then a clean validation -> no refinement needed
    fake = FakeLLM(
        [
            "Initial grounded answer.",
            "All claims SUPPORTED. Verdict: supported.",
        ]
    )
    pipeline = QAPipeline(llm=None, settings=_settings())  # type: ignore[arg-type]
    _patch_runnable(monkeypatch, fake)

    result = pipeline.answer("reference text", "a question?")

    assert result.fully_grounded is True
    assert result.iterations_used == 1
    assert result.final_answer == "Initial grounded answer."


def test_pipeline_refines_then_succeeds(monkeypatch) -> None:
    fake = FakeLLM(
        [
            "Initial answer with a bad claim.",       # initial generation
            "Claim 1: UNSUPPORTED",                   # validation #1 -> refine
            "Refined, fully supported answer.",       # refinement #1
            "All claims SUPPORTED. Verdict: ok.",     # validation #2 -> stop
        ]
    )
    pipeline = QAPipeline(llm=None, settings=_settings())  # type: ignore[arg-type]
    _patch_runnable(monkeypatch, fake)

    result = pipeline.answer("reference text", "a question?")

    assert result.fully_grounded is True
    assert result.iterations_used == 2
    assert result.final_answer == "Refined, fully supported answer."


def test_pipeline_stops_at_max_iterations(monkeypatch) -> None:
    # Always returns UNSUPPORTED -> should exhaust the iteration budget.
    fake = FakeLLM(["initial"] + ["Claim: UNSUPPORTED", "refined"] * 10)
    settings = Settings(together_api_key="k", max_refinement_iterations=3)
    pipeline = QAPipeline(llm=None, settings=settings)  # type: ignore[arg-type]
    _patch_runnable(monkeypatch, fake)

    result = pipeline.answer("reference", "q?")

    assert result.fully_grounded is False
    assert result.iterations_used == 3
