"""Multi-model ensemble validation: majority-vote groundedness across several
independent Cloudflare Workers AI models.

A validator built from the same weights as the generator is prone to miss
exactly the mistakes that model tends to make - it's grading its own
homework. Using architecturally different models (different training data,
different developers) reduces that correlated blind spot. All three still
run on Cloudflare Workers AI (no new accounts/providers needed), because
diversity here comes from different model weights, not different hosting
companies - a different provider serving the same open-weight model would
add zero independence.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from llm_qa.chains.grounding import is_fully_grounded
from llm_qa.chains.prompts import VALIDATION_PROMPT
from llm_qa.config.settings import Settings
from llm_qa.core.exceptions import LLMProviderError
from llm_qa.core.llm_provider import CloudflareWorkersAILLM
from llm_qa.core.logging_config import get_logger

logger = get_logger(__name__)

# Deliberately different model families than the generator
# (@cf/meta/llama-3.1-8b-instruct) and from each other. Verified live against
# the project's Cloudflare account - not every model on Cloudflare's catalog
# is reachable on every account/token (several return 403/410/empty-response
# on this one), so these are the three confirmed working at time of writing.
DEFAULT_VALIDATOR_MODELS: tuple[str, ...] = (
    "@cf/mistralai/mistral-small-3.1-24b-instruct",
    "@cf/qwen/qwen2.5-coder-32b-instruct",
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
)


@dataclass
class ValidatorVote:
    """One validator model's verdict, kept for auditability even on disagreement."""

    model_name: str
    grounded: bool | None  # None means this validator's call failed
    text: str


class EnsembleValidator:
    """Runs VALIDATION_PROMPT against several models in parallel and majority-votes."""

    def __init__(
        self,
        settings: Settings,
        model_names: tuple[str, ...] = DEFAULT_VALIDATOR_MODELS,
    ) -> None:
        self._model_names = model_names
        self._llms: list[CloudflareWorkersAILLM] = []
        for name in model_names:
            llm = CloudflareWorkersAILLM(settings=settings)
            llm.model_name = name
            self._llms.append(llm)

    def close(self) -> None:
        for llm in self._llms:
            llm.close()

    def _run_one(
        self, llm: CloudflareWorkersAILLM, reference: str, response: str
    ) -> ValidatorVote:
        prompt = VALIDATION_PROMPT.format(reference=reference, response=response)
        try:
            text = llm.invoke(prompt)
            return ValidatorVote(
                model_name=llm.model_name,
                grounded=is_fully_grounded(text),
                text=text,
            )
        except LLMProviderError as exc:
            logger.warning("Validator %s failed: %s", llm.model_name, exc)
            return ValidatorVote(
                model_name=llm.model_name,
                grounded=None,
                text=f"[validator call failed: {exc}]",
            )

    def validate(self, reference: str, response: str) -> tuple[bool, list[ValidatorVote]]:
        """Return (majority_grounded, individual votes), for auditability.

        If more than half the panel fails to respond, there's no real
        quorum - fail safe (treat as not grounded) rather than decide on
        whatever minority did respond.
        """
        with ThreadPoolExecutor(max_workers=len(self._llms)) as pool:
            votes = list(
                pool.map(
                    lambda llm: self._run_one(llm, reference, response),
                    self._llms,
                )
            )

        successful = [v for v in votes if v.grounded is not None]
        if len(successful) <= len(self._llms) // 2:
            logger.warning(
                "Only %d/%d validators responded; treating as not grounded "
                "(fail-safe, no quorum).",
                len(successful),
                len(self._llms),
            )
            return False, votes

        grounded_count = sum(1 for v in successful if v.grounded)
        majority_grounded = grounded_count > len(successful) / 2
        return majority_grounded, votes
