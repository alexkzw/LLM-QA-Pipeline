"""API request and response schemas.

Explicit, validated contracts for the HTTP layer. Pydantic enforces types and
generates OpenAPI docs automatically.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""

    reference: str = Field(
        ...,
        min_length=1,
        description="The reference text the answer must be grounded in.",
    )
    question: str = Field(..., min_length=1, description="The question to answer.")


class RefinementStepResponse(BaseModel):
    iteration: int
    is_grounded: bool


class TokenUsageResponse(BaseModel):
    """Total token usage across every LLM call this request made (generation
    + all ensemble validator calls + any refinement passes). Fields are None
    where the provider didn't report usage for a given model."""

    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


class AskResponse(BaseModel):
    """Response body for the /ask endpoint."""

    question: str
    final_answer: str
    fully_grounded: bool
    iterations_used: int
    token_usage: TokenUsageResponse
    estimated_cost_usd: float | None = Field(
        default=None,
        description=(
            "Approximate USD cost across all calls this request made "
            "(generation + ensemble validation + refinement). See "
            "core/cost.py - this is a blended estimate, not a Cloudflare "
            "invoice figure."
        ),
    )


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str


class AskJobCreated(BaseModel):
    """Response body for POST /ask/async - poll GET /ask/jobs/{job_id} next."""

    job_id: str
    status: str


class AskJobStatusResponse(BaseModel):
    """Response body for GET /ask/jobs/{job_id}."""

    job_id: str
    status: str
    result: AskResponse | None = None
    error: str | None = None
