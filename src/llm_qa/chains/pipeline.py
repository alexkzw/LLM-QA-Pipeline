"""The grounded question-answering pipeline.

Refactored from the notebook's ``run_pipeline`` /
``iterative_refinement_chain`` functions into a class that:
  * returns structured results (dataclasses) instead of printing,
  * exposes refinement history for evaluation and auditability,
  * parses the validator output robustly instead of substring-matching.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.prompts import BasePromptTemplate

from llm_qa.chains.prompts import (
    RAG_PROMPT,
    REFINEMENT_PROMPT,
    VALIDATION_PROMPT,
    build_initial_prompt,
)
from llm_qa.config.settings import Settings
from llm_qa.core.llm_provider import TogetherAILLM
from llm_qa.core.logging_config import get_logger
from llm_qa.retrieval.retriever import Retriever

logger = get_logger(__name__)


@dataclass
class RefinementStep:
    """One validate-then-refine iteration, retained for auditability."""

    iteration: int
    validation_result: str
    is_grounded: bool
    answer_after_refinement: str | None = None


@dataclass
class QAResult:
    """The full, structured output of a single question."""

    question: str
    initial_answer: str
    final_answer: str
    iterations_used: int
    fully_grounded: bool
    history: list[RefinementStep] = field(default_factory=list)
    retrieved_chunk_ids: list[int] = field(default_factory=list)
    retrieved_context: str | None = None


def _is_fully_grounded(validation_text: str) -> bool:
    """Return True only if the validator flagged no unsupported claims.

    More robust than the notebook's raw substring check: normalises case and
    looks for the specific negative labels as whole tokens.
    """
    upper = validation_text.upper()
    return "UNSUPPORTED" not in upper and "PARTIALLY SUPPORTED" not in upper


class QAPipeline:
    """Grounded QA with iterative, fact-checked self-refinement."""

    def __init__(
        self,
        llm: TogetherAILLM,
        settings: Settings,
        retriever: Retriever | None = None,
    ) -> None:
        self._llm = llm
        self._settings = settings
        self._initial_prompt = build_initial_prompt()
        self._retriever = retriever

    def _run_chain(self, template: BasePromptTemplate, inputs: dict) -> str:
        """Pipe a prompt template into the LLM and return the text."""
        runnable = template | self._llm
        return runnable.invoke(inputs)

    def _refine_against(
        self, reference: str, initial_answer: str
    ) -> tuple[str, list[RefinementStep], bool]:
        """Run the validate->refine loop against a reference text.

        Shared by both the full-document and retrieval-based entry points.
        Returns (final_answer, history, fully_grounded).
        """
        answer = initial_answer
        history: list[RefinementStep] = []
        fully_grounded = False

        for i in range(1, self._settings.max_refinement_iterations + 1):
            validation = self._run_chain(
                VALIDATION_PROMPT,
                {"reference": reference, "response": answer},
            )
            grounded = _is_fully_grounded(validation)
            logger.info("Iteration %d grounded=%s", i, grounded)

            if grounded:
                history.append(
                    RefinementStep(
                        iteration=i,
                        validation_result=validation,
                        is_grounded=True,
                    )
                )
                fully_grounded = True
                break

            refined = self._run_chain(
                REFINEMENT_PROMPT,
                {"reference": reference, "response": answer},
            )
            history.append(
                RefinementStep(
                    iteration=i,
                    validation_result=validation,
                    is_grounded=False,
                    answer_after_refinement=refined,
                )
            )
            answer = refined

        if not fully_grounded:
            logger.warning("Max refinement depth reached; returning best attempt.")

        return answer, history, fully_grounded

    def answer(self, reference: str, question: str) -> QAResult:
        """Answer using the full reference document (no retrieval).

        Suitable for short documents. For long documents, prefer
        ``answer_with_retrieval``.
        """
        logger.info("Answering (full-document): %s", question)

        initial_answer = self._run_chain(
            self._initial_prompt,
            {"reference": reference, "prompt": question},
        )
        final, history, grounded = self._refine_against(reference, initial_answer)

        return QAResult(
            question=question,
            initial_answer=initial_answer,
            final_answer=final,
            iterations_used=len(history),
            fully_grounded=grounded,
            history=history,
        )

    def answer_with_retrieval(self, question: str) -> QAResult:
        """Answer using RAG: retrieve relevant chunks, then ground + refine.

        This is the scalable path for large documents - only the most relevant
        chunks are sent to the LLM, not the whole document.
        """
        if self._retriever is None:
            raise RuntimeError(
                "Pipeline was built without a retriever; "
                "use build_rag_pipeline()."
            )

        logger.info("Answering (retrieval): %s", question)

        retrieved = self._retriever.retrieve(
            question, top_k=self._settings.retrieval_top_k
        )
        context = self._retriever.format_context(retrieved)

        initial_answer = self._run_chain(
            RAG_PROMPT,
            {"context": context, "question": question},
        )
        # Refinement is validated against the retrieved context (the only
        # evidence the answer is permitted to use).
        final, history, grounded = self._refine_against(context, initial_answer)

        return QAResult(
            question=question,
            initial_answer=initial_answer,
            final_answer=final,
            iterations_used=len(history),
            fully_grounded=grounded,
            history=history,
            retrieved_chunk_ids=[c.chunk_id for c in retrieved],
            retrieved_context=context,
        )

    def answer_without_reference(self, question: str) -> str:
        """Baseline: answer with no grounding (for comparison/eval)."""
        return self._llm.invoke(question)
