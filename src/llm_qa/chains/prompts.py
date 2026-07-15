"""Prompt templates and few-shot examples.

All prompt engineering lives here so prompts can be version controlled, 
reviewed, and unit tested independently of execution logic.
"""
from __future__ import annotations

from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

from llm_qa.chains.example_bank import FewShotExample

# --- Few-shot examples (grounded in the reference document) -------------
# Deliberately simple factual snippets that demonstrate the desired answer
# style: grounded, attributed, and concise. These are drawn from the OECD
# Economic Outlook 2026 reference document.
FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    {
        "question": "What is projected global GDP growth for 2026?",
        "answer": (
            "The reference document projects global growth to slow from 3.4% in "
            "2025 to 2.8% in 2026, before recovering to 3.1% in 2027 under the "
            "time-limited disruption scenario."
        ),
    },
    {
        "question": (
            "What event does the editorial identify as the dominant force "
            "shaping the global outlook?"
        ),
        "answer": (
            "The conflict in the Middle East has become the dominant force "
            "shaping the global economic outlook, according to the editorial."
        ),
    },
    {
        "question": "What approach did the OECD use for its global projections?",
        "answer": (
            "A scenario-based approach rather than a single forecast, defining a "
            "'time-limited disruption' scenario and a 'prolonged disruption' "
            "scenario to capture the range of possible outcomes."
        ),
    },
]

EXAMPLE_PROMPT = PromptTemplate.from_template(
    "Question: {question}\nAnswer: {answer}"
)

_PREFIX = (
    "You are an assistant with access to the following reference document.\n"
    "Answer ONLY from this reference - cite nothing else and do not guess.\n\n"
    "Reference (verbatim extract):\n{reference}\n\n"
)

_SUFFIX = "\nQuestion: {prompt}\nAnswer:"


def build_initial_prompt() -> FewShotPromptTemplate:
    """Few-shot, grounded prompt used to generate the first answer."""
    return FewShotPromptTemplate(
        prefix=_PREFIX,
        suffix=_SUFFIX,
        examples=FEW_SHOT_EXAMPLES,
        example_prompt=EXAMPLE_PROMPT,
        input_variables=["reference", "prompt"],
    )


# --- RAG prompt (operates over retrieved chunks, not the whole document) ---
_RAG_EXAMPLE_PROMPT = PromptTemplate.from_template(
    "Context passages:\n{context}\n\nQuestion: {question}\nAnswer: {answer}"
)

_RAG_PREFIX = (
    "You are an assistant answering questions about a document. Use ONLY "
    "the numbered context passages below. Cite the passages you rely on "
    "using their bracket numbers, e.g. [1] or [2].\n\n"
    "If the context does not contain enough information to answer, reply "
    "exactly: 'The reference does not provide enough information to answer "
    "this question.' Do not guess or use outside knowledge.\n"
)

_RAG_SUFFIX = "Context passages:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def build_rag_prompt(examples: list[FewShotExample]) -> FewShotPromptTemplate:
    """RAG prompt with dynamically-selected few-shot examples.

    Unlike ``build_initial_prompt``'s fixed few-shot set, the examples here
    vary per question (see ``RagExampleSelector``), so this builds a fresh
    template per call instead of being a module-level constant.
    """
    return FewShotPromptTemplate(
        prefix=_RAG_PREFIX,
        suffix=_RAG_SUFFIX,
        examples=[
            {"context": e.context, "question": e.question, "answer": e.answer}
            for e in examples
        ],
        example_prompt=_RAG_EXAMPLE_PROMPT,
        input_variables=["context", "question"],
    )


VALIDATION_PROMPT = PromptTemplate(
    input_variables=["reference", "response"],
    template=(
        "You are a fact-checking assistant. For EACH factual claim in the "
        "response, state whether it is SUPPORTED, UNSUPPORTED, or PARTIALLY "
        "SUPPORTED by the reference.\n\nReference:\n{reference}\n\n"
        "Response:\n{response}\n\n"
        "Return a bullet list of labelled claims followed by an overall verdict."
    ),
)

REFINEMENT_PROMPT = PromptTemplate(
    input_variables=["reference", "response", "validation"],
    template=(
        "The previous answer was fact-checked against the reference, claim "
        "by claim, with this result:\n\n{validation}\n\n"
        "Using ONLY the reference below, rewrite the answer:\n"
        "1. For each claim marked UNSUPPORTED or PARTIALLY SUPPORTED above, "
        "identify exactly what the reference does or doesn't say about it.\n"
        "2. Rewrite the answer so every remaining statement is fully "
        "supported by the reference - if a claim cannot be supported, omit "
        "it rather than guessing.\n\n"
        "Reference:\n{reference}\n\nCurrent answer:\n{response}\n\n"
        "Revised answer:"
    ),
)