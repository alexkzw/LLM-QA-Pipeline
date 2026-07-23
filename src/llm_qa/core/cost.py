"""Approximate USD cost estimation from token usage.

Cloudflare Workers AI bills text-generation usage in "Neurons" (a
compute-time unit), not a flat $-per-token rate the way OpenAI/Anthropic do -
the neuron cost per token varies by model size and isn't published as a
simple per-token figure. The rates below are blended $-per-1K-tokens
approximations (roughly derived from published per-model neuron costs at
the time of writing), good enough to catch a cost regression or compare
models in aggregate, but they are NOT what your Cloudflare invoice will say
line-by-line. Treat this as an observability signal, not a billing source
of truth - if exact cost matters, read it from the Cloudflare dashboard.
"""
from __future__ import annotations

from llm_qa.core.llm_provider import TokenUsage

# $ per 1,000 total tokens (prompt + completion treated the same, since
# Workers AI doesn't expose separate input/output neuron rates per call).
# Unlisted models fall back to _DEFAULT_COST_PER_1K.
_DEFAULT_COST_PER_1K = 0.01
MODEL_COST_PER_1K_TOKENS: dict[str, float] = {
    "@cf/meta/llama-3.1-8b-instruct": 0.01,
    "@cf/mistralai/mistral-small-3.1-24b-instruct": 0.03,
    "@cf/qwen/qwen2.5-coder-32b-instruct": 0.03,
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast": 0.05,
}


def estimate_cost_usd(model_name: str, usage: TokenUsage) -> float | None:
    """Best-effort $ estimate for one call. None if token counts are unknown."""
    if usage.total_tokens is None:
        return None
    rate = MODEL_COST_PER_1K_TOKENS.get(model_name, _DEFAULT_COST_PER_1K)
    return round((usage.total_tokens / 1000) * rate, 6)
