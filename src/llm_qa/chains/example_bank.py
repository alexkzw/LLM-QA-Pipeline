"""Curated few-shot example bank for the RAG prompt, selected dynamically per
question rather than injected as one fixed set on every call.

Small enough (a handful of examples) that brute-force cosine similarity over
an in-memory list is the right tool - reusing Chroma's ANN index here would
add dependency surface to solve a scale problem this doesn't have (the same
reasoning behind vector_store.py's own note that HNSW is overkill below a
few thousand items).
"""
from __future__ import annotations

from dataclasses import dataclass

from llm_qa.retrieval.embeddings import EmbeddingModel


@dataclass(frozen=True)
class FewShotExample:
    """One worked RAG example: retrieved-style context, question, and answer."""

    question: str
    context: str
    answer: str


# Demonstrate the two behaviours plain instructions are weakest at teaching:
# multi-passage citation, and the exact decline phrasing for unanswerable
# questions (the eval harness scores abstention_accuracy on the latter, so
# showing the exact wording rather than only describing it should help).
RAG_FEW_SHOT_EXAMPLES: list[FewShotExample] = [
    FewShotExample(
        question="What is the projected global GDP growth rate for 2026?",
        context=(
            "[1] (relevance 0.91)\n"
            "Global growth is projected to slow from 3.4% in 2025 to 2.8% in "
            "2026, before recovering to 3.1% in 2027 under the time-limited "
            "disruption scenario.\n\n"
            "[2] (relevance 0.77)\n"
            "The prolonged disruption scenario projects weaker growth of "
            "2.3% in 2026 and 2.6% in 2027."
        ),
        answer=(
            "Under the time-limited disruption scenario, global growth is "
            "projected to slow to 2.8% in 2026 [1]. Under the more severe "
            "prolonged disruption scenario, growth would be weaker at 2.3% "
            "in 2026 [2]."
        ),
    ),
    FewShotExample(
        question="How does inflation affect interest rate decisions in 2026?",
        context=(
            "[1] (relevance 0.88)\n"
            "Headline inflation is expected to rise to 4.0% in 2026 due to "
            "renewed supply-chain disruption.\n\n"
            "[2] (relevance 0.81)\n"
            "Central banks are expected to delay planned rate cuts if "
            "inflation remains above target, prioritising price stability "
            "over growth support."
        ),
        answer=(
            "Inflation is projected to rise to 4.0% in 2026 [1], and central "
            "banks are expected to delay planned interest rate cuts as a "
            "result, prioritising price stability over supporting growth [2]."
        ),
    ),
    FewShotExample(
        question=(
            "What event does the editorial identify as the dominant force "
            "shaping the global outlook?"
        ),
        context=(
            "[1] (relevance 0.93)\n"
            "The conflict in the Middle East has become the dominant force "
            "shaping the global economic outlook, according to the editorial."
        ),
        answer=(
            "The conflict in the Middle East has become the dominant force "
            "shaping the global economic outlook [1]."
        ),
    ),
    FewShotExample(
        question="What is the population of Japan mentioned in the reference?",
        context=(
            "[1] (relevance 0.42)\n"
            "Japan's monetary policy remains accommodative relative to other "
            "advanced economies.\n\n"
            "[2] (relevance 0.38)\n"
            "Wage growth in Japan is expected to support a modest recovery "
            "in private consumption."
        ),
        answer=(
            "The reference does not provide enough information to answer "
            "this question."
        ),
    ),
    FewShotExample(
        question="What was the stock market's reaction on the day of publication?",
        context=(
            "[1] (relevance 0.35)\n"
            "The scenario-based approach captures a range of possible "
            "outcomes rather than a single point forecast.\n\n"
            "[2] (relevance 0.31)\n"
            "Trade policy uncertainty remains a key downside risk across "
            "all scenarios considered."
        ),
        answer=(
            "The reference does not provide enough information to answer "
            "this question."
        ),
    ),
]


class RagExampleSelector:
    """Picks the few-shot examples most relevant to a given question.

    Reuses the same EmbeddingModel as retrieval - it must be the same model
    instance/weights the rest of the RAG path uses, for the same reason
    indexing and querying must share an embedder (see Retriever): vectors
    from two different models aren't comparable.
    """

    def __init__(
        self,
        embedder: EmbeddingModel,
        examples: list[FewShotExample] | None = None,
    ) -> None:
        self._embedder = embedder
        self._examples = (
            examples if examples is not None else RAG_FEW_SHOT_EXAMPLES
        )
        # embed() normalises to unit vectors, so a plain dot product below
        # already equals cosine similarity - no extra normalisation needed.
        self._vectors = self._embedder.embed(
            [e.question for e in self._examples]
        )

    def select(self, question: str, k: int = 2) -> list[FewShotExample]:
        """Return the ``k`` examples whose questions are most similar to ``question``."""
        if not self._examples:
            return []
        query_vec = self._embedder.embed_one(question)
        scored = [
            (sum(q * v for q, v in zip(query_vec, vec, strict=True)), example)
            for vec, example in zip(self._vectors, self._examples, strict=True)
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [example for _, example in scored[:k]]
