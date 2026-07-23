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

from llm_qa.chains.ensemble_validator import EnsembleValidator, ValidatorVote
from llm_qa.chains.example_bank import RagExampleSelector
from llm_qa.chains.prompts import (
    REFINEMENT_PROMPT,
    build_initial_prompt,
    build_rag_prompt,
)
from llm_qa.config.settings import Settings
from llm_qa.core.cost import estimate_cost_usd
from llm_qa.core.exceptions import ConfigurationError
from llm_qa.core.llm_provider import CloudflareWorkersAILLM, TokenUsage
from llm_qa.core.logging_config import get_logger
from llm_qa.retrieval.retriever import Retriever

logger = get_logger(__name__)

_EMPTY_USAGE = TokenUsage(prompt_tokens=None, completion_tokens=None, total_tokens=None)


class _UsageTracker:
    """Accumulates token usage and $ cost across every LLM call in one run.

    A pipeline run fans out into many calls - one generation, plus 3
    validator calls and (on refinement) another generation per iteration -
    each potentially a different model with a different cost/1K rate (see
    core/cost.py). Cost is summed per-call rather than from the aggregate
    token total, since applying one blended rate to tokens from several
    differently-priced models would misstate the total.
    """

    def __init__(self) -> None:
        self.usage = _EMPTY_USAGE
        self._cost_usd = 0.0
        self._any_known_cost = False

    def record(self, model_name: str, usage: TokenUsage | None) -> None:
        if usage is None:
            return
        self.usage = self.usage + usage
        call_cost = estimate_cost_usd(model_name, usage)
        if call_cost is not None:
            self._cost_usd += call_cost
            self._any_known_cost = True

    @property
    def estimated_cost_usd(self) -> float | None:
        return round(self._cost_usd, 6) if self._any_known_cost else None


@dataclass
class RefinementStep:
    """One validate-then-refine iteration, retained for auditability."""

    iteration: int
    validation_result: str
    is_grounded: bool
    answer_after_refinement: str | None = None
    validator_votes: list[ValidatorVote] = field(default_factory=list)


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
    token_usage: TokenUsage = field(default_factory=lambda: _EMPTY_USAGE)
    estimated_cost_usd: float | None = None


class QAPipeline:
    """Grounded QA with iterative, fact-checked self-refinement."""

    def __init__(
        self,
        llm: CloudflareWorkersAILLM,
        settings: Settings,
        retriever: Retriever | None = None,
        ensemble_validator: EnsembleValidator | None = None,
        rag_example_selector: RagExampleSelector | None = None,
    ) -> None:
        self._llm = llm
        self._settings = settings
        self._initial_prompt = build_initial_prompt()
        self._retriever = retriever
        self._ensemble_validator = (
            ensemble_validator
            if ensemble_validator is not None
            else EnsembleValidator(settings)
        )
        self._rag_example_selector: RagExampleSelector | None
        if rag_example_selector is not None:
            self._rag_example_selector = rag_example_selector
        elif retriever is not None:
            self._rag_example_selector = RagExampleSelector(retriever.embedder)
        else:
            self._rag_example_selector = None

    def close(self) -> None:
        """Release resources held by the underlying LLM (e.g. HTTP connections)."""
        if self._llm is not None:
            self._llm.close()
        if self._ensemble_validator is not None:
            self._ensemble_validator.close()

    def _run_chain(self, template: BasePromptTemplate, inputs: dict) -> str:
        """Pipe a prompt template into the LLM and return the text."""
        runnable = template | self._llm
        return runnable.invoke(inputs)

    def _record_llm_usage(self, tracker: _UsageTracker) -> None:
        """Record the most recent _run_chain call's usage onto tracker.

        self._llm is None only in tests that stub out _run_chain entirely
        (see test_pipeline.py) - nothing to record in that case.
        """
        if self._llm is not None:
            tracker.record(self._llm.model_name, self._llm.get_last_usage())

    @staticmethod
    def _summarise_votes(votes: list[ValidatorVote]) -> str:
        def _label(vote: ValidatorVote) -> str:
            if vote.grounded is None:
                return "failed"
            return "grounded" if vote.grounded else "unsupported"

        return "; ".join(f"{v.model_name}={_label(v)}" for v in votes)

    def _refine_against(
        self, reference: str, initial_answer: str, tracker: _UsageTracker
    ) -> tuple[str, list[RefinementStep], bool]:
        """Run the validate->refine loop against a reference text.

        Shared by both the full-document and retrieval-based entry points.
        Validation runs across several independent models (see
        EnsembleValidator) rather than a single model checking its own
        output; by default accepting "grounded" requires unanimous
        agreement, not just a majority. Returns (final_answer, history,
        fully_grounded); token usage/cost across every call made here is
        recorded onto the caller-owned tracker.
        """
        answer = initial_answer
        history: list[RefinementStep] = []
        fully_grounded = False

        for i in range(1, self._settings.max_refinement_iterations + 1):
            grounded, votes = self._ensemble_validator.validate(reference, answer)
            for vote in votes:
                tracker.record(vote.model_name, vote.usage)
            summary = self._summarise_votes(votes)
            logger.info("Iteration %d accepted_grounded=%s (%s)", i, grounded, summary)

            if grounded:
                history.append(
                    RefinementStep(
                        iteration=i,
                        validation_result=summary,
                        is_grounded=True,
                        validator_votes=votes,
                    )
                )
                fully_grounded = True
                break

            # Feed the refiner every validator's claim-by-claim breakdown,
            # not just one - it can catch issues only one of them flagged.
            combined_validation = "\n\n".join(
                f"--- Validator ({v.model_name}) ---\n{v.text}"
                for v in votes
                if v.grounded is not None
            )
            refined = self._run_chain(
                REFINEMENT_PROMPT,
                {
                    "reference": reference,
                    "response": answer,
                    "validation": combined_validation,
                },
            )
            self._record_llm_usage(tracker)
            history.append(
                RefinementStep(
                    iteration=i,
                    validation_result=summary,
                    is_grounded=False,
                    answer_after_refinement=refined,
                    validator_votes=votes,
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

        tracker = _UsageTracker()
        initial_answer = self._run_chain(
            self._initial_prompt,
            {"reference": reference, "prompt": question},
        )
        self._record_llm_usage(tracker)
        final, history, grounded = self._refine_against(
            reference, initial_answer, tracker
        )

        return QAResult(
            question=question,
            initial_answer=initial_answer,
            final_answer=final,
            iterations_used=len(history),
            fully_grounded=grounded,
            history=history,
            token_usage=tracker.usage,
            estimated_cost_usd=tracker.estimated_cost_usd,
        )

    def answer_with_retrieval(self, question: str) -> QAResult:
        """Answer using RAG: retrieve relevant chunks, then ground + refine.

        This is the scalable path for large documents - only the most relevant
        chunks are sent to the LLM, not the whole document.
        """
        if self._retriever is None:
            raise ConfigurationError(
                "Pipeline was built without a retriever; "
                "use build_rag_pipeline()."
            )

        logger.info("Answering (retrieval): %s", question)

        retrieved = self._retriever.retrieve(
            question, top_k=self._settings.retrieval_top_k
        )

        # format matched chunks (plain strings) into a numbered text block
        context = self._retriever.format_context(retrieved)

        # Few-shot examples are selected per-question (most relevant to this
        # question specifically), not a fixed set injected every time.
        examples = (
            self._rag_example_selector.select(question, k=2)
            if self._rag_example_selector is not None
            else []
        )
        prompt = build_rag_prompt(examples)

        tracker = _UsageTracker()
        initial_answer = self._run_chain(
            prompt,
            {"context": context, "question": question},
        )
        self._record_llm_usage(tracker)
        # Refinement is validated against the retrieved context (the only
        # evidence the answer is permitted to use).
        final, history, grounded = self._refine_against(
            context, initial_answer, tracker
        )

        return QAResult(
            question=question,
            initial_answer=initial_answer,
            final_answer=final,
            iterations_used=len(history),
            fully_grounded=grounded,
            history=history,
            retrieved_chunk_ids=[c.chunk_id for c in retrieved],
            retrieved_context=context,
            token_usage=tracker.usage,
            estimated_cost_usd=tracker.estimated_cost_usd,
        )

    def answer_without_reference(self, question: str) -> str:
        """Baseline: answer with no grounding (for comparison/eval)."""
        return self._llm.invoke(question)
