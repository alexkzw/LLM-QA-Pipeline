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
#
# Every context passage and figure below was pulled from a real, live query
# against the actual indexed OECD PDF (verified, not invented) - an earlier
# version of this bank used illustrative-but-invented figures, which risks
# the model treating fictional example content as available fact. Building
# from real retrieved passages removes that risk entirely. None of these
# questions overlap with data/evaluation_set.json's questions/topics, to
# avoid the prompt design leaking information about the held-out eval set.
RAG_FEW_SHOT_EXAMPLES: list[FewShotExample] = [
    FewShotExample(
        question=(
            "What is Canada's economic outlook for 2026, in terms of growth "
            "and inflation?"
        ),
        context=(
            "[1] (relevance 0.74)\n"
            "Real GDP growth in the United States and Canada will benefit "
            "from stronger energy-sector exports, but this will be offset by "
            "the negative impact of higher energy prices on inflation and "
            "household purchasing power. In the United States, growth is "
            "projected to slow from 2.1% in 2025 to 2.0% in 2026 and 1.8% in "
            "2027, while Canada's growth is expected to decline from 1.7% in "
            "2025 to 1.2% in 2026 before rebounding to 1.7% in 2027 as "
            "domestic demand recovers.\n\n"
            "[2] (relevance 0.68)\n"
            "Headline inflation rose to 2.8% in April 2026, from 2.4% in "
            "March, driven by higher energy prices linked to the conflict in "
            "the Middle East. Core inflation continued to moderate, easing "
            "to 1.5% in April."
        ),
        answer=(
            "Canada's growth is expected to decline from 1.7% in 2025 to "
            "1.2% in 2026, before rebounding to 1.7% in 2027 as domestic "
            "demand recovers [1]. Headline inflation rose to 2.8% in April "
            "2026, up from 2.4% in March, driven by higher energy prices "
            "linked to the Middle East conflict, while core inflation eased "
            "to 1.5% in April [2]."
        ),
    ),
    FewShotExample(
        question="How are export volumes in China expected to grow in 2026?",
        context=(
            "[1] (relevance 0.71)\n"
            "In 2026, export volumes in China and the dynamic Asian "
            "economies are expected to grow by over 6.5%, amongst the "
            "fastest rates globally. Aggregate trade growth is projected to "
            "slow in the second and third quarters of this year, reflecting "
            "a sharp decline in trade with the Gulf economies, and "
            "increasing energy and transport costs."
        ),
        answer=(
            "Export volumes in China and the dynamic Asian economies are "
            "expected to grow by over 6.5% in 2026, among the fastest rates "
            "globally [1]."
        ),
    ),
    FewShotExample(
        question="What has the Bank of Japan done regarding its policy interest rate?",
        context=(
            "[1] (relevance 0.68)\n"
            "The Bank of Japan has taken steps toward monetary policy "
            "normalisation by starting to reduce the size of its balance "
            "sheet and raising the policy interest rate, which reached "
            "around 0.75% in December 2025."
        ),
        answer=(
            "The Bank of Japan has taken steps toward monetary policy "
            "normalisation, including raising its policy interest rate to "
            "around 0.75% in December 2025 [1]."
        ),
    ),
    FewShotExample(
        question="What is the population of Japan mentioned in the reference?",
        context=(
            "[1] (relevance 0.43)\n"
            "Japan: Demand, output and prices. GDP at market prices 584.9 "
            "0.7 -0.2 1.1 0.6 0.8. Private consumption 319.0 0.1 -0.6 1.3 "
            "0.7 0.6. Government consumption 121.1 -0.2 1.6 1.0 1.7 1.9.\n\n"
            "[2] (relevance 0.36)\n"
            "Monetary policy is projected to tighten, while fiscal policy "
            "will support growth in 2026. The Bank of Japan has taken steps "
            "toward monetary policy normalisation by starting to reduce the "
            "size of its balance sheet and raising the policy interest rate, "
            "which reached around 0.75% in December 2025."
        ),
        answer=(
            "The reference does not provide enough information to answer "
            "this question."
        ),
    ),
    FewShotExample(
        question="What was the stock market's reaction on the day of publication?",
        context=(
            "[1] (relevance 0.39)\n"
            "Congressional Research Service (2026), \"Private credit funds "
            "redemption restrictions: Market context and policy issues\", "
            "Insight, IN12674, Washington, April. Conigrave, B. and Y-H. "
            "Shin (2026), \"Fiscal and macroeconomic impacts of defence "
            "spending\", OECD Economics Department Working Papers, No. 1861.\n\n"
            "[2] (relevance 0.38)\n"
            "S&P (2026a) \"J.P. Morgan global composite PMI: Global economic "
            "growth eases to 11-month low\", News release, S&P Global, 7 "
            "April. S&P (2026b) \"Global default rate forecast: Defaults to "
            "edge up...\", Default, Transition, and Recovery, S&P Global, "
            "26 May."
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
        """Return the ``k`` examples whose questions are closest to ``question``."""
        if not self._examples:
            return []
        query_vec = self._embedder.embed_one(question)
        scored = [
            (sum(q * v for q, v in zip(query_vec, vec, strict=True)), example)
            for vec, example in zip(self._vectors, self._examples, strict=True)
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [example for _, example in scored[:k]]
